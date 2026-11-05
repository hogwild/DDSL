"""
run_gru_pipeline.py
====================================
Runs the GRU learnability-probe train + test loop sequentially across DDSL
iterations, for a single seed -- structured like run_transformer_pipeline.py,
but scoped only to the GRU probe.

This script does NOT do labeling, dataset building, or logit export. It
assumes outputs/{dataset}/iter_k/dataset/{train,val,test}/ already exists,
built by the main DDSL pipeline (run_transformer_pipeline.py). The GRU
probe is trained and evaluated on those SAME labels; this script only chains
run_train_gru.py -> run_test_gru.py, once per iteration.

Any subprocess failure stops the script immediately (subprocess.run(check=True)
raises CalledProcessError, which is left unhandled here on purpose -- no
try/except swallows it) and prints the failing command and iteration.

Usage
-----
Unified mode (the same YAML as Transformer, with a gru section):
    python src/pipeline/run_gru_pipeline.py \\
        --config configs/pipeline/SSE50.yaml \\
        --input_root outputs/pipeline_runs/sse50_run \\
        --output_root outputs/gru_runs/sse50_probe --dry_run

The input root holds the existing DDSL datasets; the output root holds only
GRU artifacts. Omitting input_root retains the historical shared-tree layout.

Advanced Python mode accepts an explicitly supplied GRU stage YAML via
--train_config instead of --config. Stage YAMLs are generated inside a run
under gru_pipeline_config/seed_<S>/iter_<start>_<end>/train_gru.yaml;
no standalone training templates are bundled. See docs/experiments.md.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.gru_pipeline_config import GRURunConfig


@dataclass
class GRUPipelineArgs:
    dataset: str
    train_config: str
    output_root: str
    start_iter: int
    end_iter: int
    seed: int
    adjust: str
    python_exec: str
    skip_test: bool = False
    dry_run: bool = False
    input_root: str | None = None
    unified_config: GRURunConfig | None = None


def parse_args() -> GRUPipelineArgs:
    p = argparse.ArgumentParser(
        description=(
            "Run the GRU learnability-probe train+test loop sequentially "
            "across DDSL iterations, for a single seed. Does not build "
            "labels/datasets and never exports logits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", default=None, help="Unified pipeline YAML with a gru section")
    p.add_argument("--input_root", default=None, help="Existing Transformer DDSL outputs; defaults to output_root")
    p.add_argument("--dataset", default=None)
    p.add_argument("--train_config", default=None)
    p.add_argument("--output_root", default=None)

    p.add_argument("--start_iter", type=int, default=None)
    p.add_argument("--end_iter", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--adjust", default=None)

    p.add_argument(
        "--python_exec",
        default=sys.executable,
        help="Python executable used to run subcommands.",
    )

    p.add_argument(
        "--skip_test",
        action="store_true", default=None,
        help="Skip run_test_gru.py after each train step.",
    )

    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them.",
    )

    args = p.parse_args()
    unified = None
    if args.config:
        if any(getattr(args, key) is not None for key in ("dataset", "adjust", "train_config")):
            p.error("--config cannot be combined with --dataset, --adjust or --train_config")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from pipeline.gru_pipeline_config import load_gru_config
        try:
            unified = load_gru_config(args.config, {key: getattr(args, key) for key in
                                      ("output_root", "start_iter", "end_iter", "seed", "skip_test")},
                                      input_root=args.input_root)
        except (ValueError, OSError) as exc:
            p.error(str(exc))
        for key in ("output_root", "start_iter", "end_iter", "seed", "skip_test"):
            setattr(args, key, unified.resolved["pipeline"][key])
        args.dataset = unified.resolved["dataset"]["name"]
        args.adjust = unified.resolved["dataset"]["adjust"]
        args.input_root = str(unified.input_root)
        args.train_config = str(unified.train_path)
    else:
        # Custom stage configurations remain supported, but no stage files are bundled.
        if not args.train_config:
            p.error("Use --config configs/pipeline/<DATASET>.yaml, or explicitly supply "
                    "--train_config with a GRU stage YAML.")
        defaults = {"dataset": "SSE50",
                    "output_root": "outputs", "start_iter": 0, "end_iter": 5,
                    "seed": 42, "adjust": "hfq", "skip_test": False}
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)

    if args.start_iter < 0 or args.end_iter < args.start_iter:
        raise ValueError(
            f"Invalid iteration range: start_iter={args.start_iter}, "
            f"end_iter={args.end_iter}"
        )

    return GRUPipelineArgs(
        dataset=args.dataset,
        train_config=args.train_config,
        output_root=args.output_root,
        start_iter=args.start_iter,
        end_iter=args.end_iter,
        seed=args.seed,
        adjust=args.adjust,
        python_exec=args.python_exec,
        skip_test=args.skip_test,
        dry_run=args.dry_run,
        input_root=args.input_root,
        unified_config=unified,
    )


# ---------------------------------------------------------------------------
# Precondition check
# ---------------------------------------------------------------------------
# This is a convenience early-exit only, using the same path convention that
# run_train_gru.py itself validates. It is not a second source of truth: if
# this check somehow disagreed with run_train_gru.py's own validation, the
# real one still runs and still fails loud. The point of checking here first
# is to fail before spending any GPU time on a doomed iteration, and to name
# the missing DDSL pipeline step explicitly instead of surfacing a generic
# FileNotFoundError from three subprocess layers down.


def dataset_ready(output_root: Path, dataset: str, iteration: int, adjust: str) -> bool:
    dataset_dir = output_root / dataset / f"iter_{iteration}" / "dataset"
    train_pq = dataset_dir / "train" / f"ALL_tokens_{adjust}.parquet"
    val_pq = dataset_dir / "val" / f"ALL_tokens_{adjust}.parquet"
    test_pq = dataset_dir / "test" / f"ALL_tokens_{adjust}.parquet"
    return train_pq.exists() and val_pq.exists() and test_pq.exists()


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

def build_train_cmd(args: GRUPipelineArgs, iteration: int) -> list[str]:
    return [
        args.python_exec, "src/pipeline/run_train_gru.py",
        "--config", args.train_config,
        "--dataset", args.dataset,
        "--iteration", str(iteration),
        "--output_root", args.output_root,
        "--seed", str(args.seed),
        "--adjust", args.adjust,
    ] + (["--input_root", args.input_root] if args.input_root else [])


def build_test_cmd(args: GRUPipelineArgs, iteration: int) -> list[str]:
    return [
        args.python_exec, "src/pipeline/run_test_gru.py",
        "--dataset", args.dataset,
        "--iteration", str(iteration),
        "--output_root", args.output_root,
        "--seed", str(args.seed),
        "--adjust", args.adjust,
    ] + (["--input_root", args.input_root] if args.input_root else [])


def run_step(cmd: list[str], *, dry_run: bool) -> None:
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n$ {printable}")
    if dry_run:
        return
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[INFO] step finished in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    input_root = Path(args.input_root or args.output_root)
    if args.unified_config is not None:
        args.unified_config.prepare(dry_run=args.dry_run)

    print("=" * 70)
    print("GRU learnability-probe pipeline (train + test only, no labeling)")
    print("=" * 70)
    print(f"dataset      : {args.dataset}")
    print(f"train_config : {args.train_config}")
    print(f"input_root   : {input_root}")
    print(f"output_root  : {args.output_root}")
    print(f"iterations   : {args.start_iter} .. {args.end_iter}")
    print(f"seed         : {args.seed}")
    print(f"adjust       : {args.adjust}")
    print(f"skip_test    : {args.skip_test}")
    print(f"dry_run      : {args.dry_run}")
    print("=" * 70)

    for iteration in range(args.start_iter, args.end_iter + 1):
        print(f"\n{'#' * 70}\n# iter_{iteration}\n{'#' * 70}")

        if not args.dry_run and not dataset_ready(input_root, args.dataset, iteration, args.adjust):
            raise FileNotFoundError(
                f"{input_root}/{args.dataset}/iter_{iteration}/dataset/{{train,val,test}}/"
                f"ALL_tokens_{args.adjust}.parquet not fully present.\n"
                f"Run the main DDSL pipeline (run_transformer_pipeline.py) "
                f"for iter_{iteration} first -- the GRU probe reuses the same "
                f"labels/dataset as the Transformer and does not build them."
            )

        run_step(build_train_cmd(args, iteration), dry_run=args.dry_run)

        if not args.skip_test:
            run_step(build_test_cmd(args, iteration), dry_run=args.dry_run)

    print("\n" + "=" * 70)
    print("GRU pipeline finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()
