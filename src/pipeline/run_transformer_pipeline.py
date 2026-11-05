from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.pipeline_config import CompiledPipelineConfig


@dataclass
class PipelineArgs:
    dataset: str
    label_config: str
    build_config: str
    train_config: str
    codebook_config: str
    output_root: str
    start_iter: int
    end_iter: int
    seed: int
    n_workers: int
    adjust: str
    labeler_type: str
    python_exec: str
    skip_test: bool = False
    skip_export_logits: bool = False
    force_rebuild_shared_logits_dataset: bool = False
    force_fit_codebook: bool = False
    force_regenerate_blacklist: bool = False
    dry_run: bool = False
    beta_model: float | None = None
    train_precision: str | None = None
    unified_config: CompiledPipelineConfig | None = None

def parse_args() -> PipelineArgs:
    p = argparse.ArgumentParser(
        description=(
            "Run the full DDSL iterative pipeline sequentially on a workstation or inside one Slurm job. "
            "This script does not call sbatch. It directly runs the pipeline programs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", default=None, help="Unified pipeline YAML. Otherwise explicitly supply all four stage YAML paths.")
    p.add_argument("--dataset", default=None)
    p.add_argument("--label_config", default=None)
    p.add_argument("--build_config", default=None)
    p.add_argument("--train_config", default=None)
    p.add_argument("--train_precision", choices=("fp32", "bf16", "fp16"), default=None,
                   help="Override training.precision; omitted uses the training YAML (default fp32). Applies only to training.")
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
        action="store_true",
        default=None,
        help="Skip run_test_transformer.py.",
    )
    p.add_argument(
        "--skip_export_logits",
        action="store_true",
        default=None,
        help="Skip run_export_transformer_logits.py. Only useful for debugging.",
    )
    p.add_argument(
        "--force_rebuild_shared_logits_dataset",
        action="store_true",
        default=None,
        help=(
            "Rebuild outputs/{dataset}/shared/logits_dataset even if it already exists. "
            "Normally this is built once after iter_0 labeling."
        ),
    )

    p.add_argument(
        "--codebook_config",
        default=None,
        help="YAML config for fitting K/Q/G/M codebooks.",
    )

    p.add_argument(
        "--force_fit_codebook",
        action="store_true",
        default=None,
        help="Refit codebook even if codebook artifacts already exist.",
    )

    p.add_argument(
        "--force_regenerate_blacklist",
        action="store_true",
        default=None,
        help="Regenerate blacklist even if blacklist already exist.",
    )

    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them.",
    )

    p.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="number of workers",
    )

    p.add_argument(
        "--labeler",
        default=None,
        help="choose the type of the labeler",
    )

    p.add_argument(
        "--beta_model",
        type=float,
        default=None,
        help="Override ddsl.beta_model in the label config. Omit to use the YAML value.",
    )

    a = p.parse_args()

    unified_config = None
    if a.config is not None:
        incompatible = [name for name in ("dataset", "adjust", "label_config", "build_config", "train_config", "codebook_config") if getattr(a, name) is not None]
        if incompatible:
            p.error("--config cannot be combined with " + ", ".join(f"--{name}" for name in incompatible) + "; set shared dataset settings in the unified YAML")
        # Keep legacy mode free of extra dependencies and preserve direct-script execution.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from pipeline.pipeline_config import RUN_DEFAULTS, load_pipeline_config

        try:
            unified_config = load_pipeline_config(
                a.config,
                {name: getattr(a, name) for name in [*RUN_DEFAULTS, "beta_model", "train_precision"]},
            )
        except (ValueError, OSError) as exc:
            p.error(str(exc))
        for name, value in unified_config.run.items():
            setattr(a, name, value)
        a.dataset = unified_config.resolved["dataset"]["name"]
        a.adjust = unified_config.resolved["dataset"]["adjust"]
        for stage, argument in [("codebook", "codebook_config"), ("label", "label_config"), ("build", "build_config"), ("train", "train_config")]:
            setattr(a, argument, unified_config.paths[stage])
        # Explicit overrides have already been applied and frozen in stage YAMLs.
        a.beta_model = None
        a.train_precision = None
    else:
        # Custom stage configurations remain supported, but no stage files are bundled.
        stage_options = ("codebook_config", "label_config", "build_config", "train_config")
        missing = [name for name in stage_options if not getattr(a, name)]
        if missing:
            p.error("Use --config configs/pipeline/<DATASET>.yaml, or explicitly supply all four "
                    "stage YAMLs. Missing: " + ", ".join(f"--{name}" for name in missing))
        defaults = {
            "dataset": "SSE50", "adjust": "hfq", "output_root": "outputs",
            "start_iter": 0, "end_iter": 3, "seed": 42, "n_workers": 8,
            "labeler": "auto", "skip_test": False, "skip_export_logits": False,
            "force_fit_codebook": False, "force_regenerate_blacklist": False,
            "force_rebuild_shared_logits_dataset": False,
        }
        for name, value in defaults.items():
            if getattr(a, name) is None:
                setattr(a, name, value)

    if a.start_iter < 0:
        raise ValueError("--start_iter must be >= 0")
    if a.end_iter < a.start_iter:
        raise ValueError("--end_iter must be >= --start_iter")
    if a.start_iter > 0:
        print(
            "[WARN] start_iter > 0. This assumes previous iteration logits already exist.",
            flush=True,
        )

    return PipelineArgs(
        dataset=a.dataset,
        label_config=a.label_config,
        build_config=a.build_config,
        train_config=a.train_config,
        codebook_config=a.codebook_config,
        output_root=a.output_root,
        start_iter=a.start_iter,
        end_iter=a.end_iter,
        beta_model=a.beta_model,
        train_precision=a.train_precision,
        seed=a.seed,
        n_workers=a.n_workers,
        adjust=a.adjust,
        labeler_type=a.labeler,
        python_exec=a.python_exec,
        skip_test=a.skip_test,
        skip_export_logits=a.skip_export_logits,
        force_rebuild_shared_logits_dataset=a.force_rebuild_shared_logits_dataset,
        force_fit_codebook=a.force_fit_codebook,
        force_regenerate_blacklist=a.force_regenerate_blacklist,
        dry_run=a.dry_run,
        unified_config=unified_config,
    )


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_command(cmd: list[str], *, dry_run: bool, log_file: Path | None = None) -> None:
    print("\n" + "=" * 80, flush=True)
    print(shell_join(cmd), flush=True)
    print("=" * 80, flush=True)

    if dry_run:
        return

    start = time.time()

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            result = subprocess.run(cmd, text=True, stdout=f, stderr=f)
    else:
        result = subprocess.run(cmd, text=True)

    elapsed = time.time() - start
    print(f"[INFO] Command finished in {elapsed / 60:.2f} minutes.", flush=True)

    if result.returncode != 0:
        if log_file is not None and log_file.exists():
            print(f"[ERROR] Command failed. Log file: {log_file}", flush=True)
            print(log_file.read_text(encoding="utf-8"), flush=True)
        raise RuntimeError(
            f"Command failed with return code {result.returncode}:\n"
            f"{shell_join(cmd)}"
        )

def _log_dir(args: PipelineArgs, iteration: int) -> Path:
    """Return the log directory for each iteration."""
    return Path(args.output_root) / args.dataset / f"iter_{iteration}" / "logs"


def path_exists_nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(path.iterdir())
    return False


def shared_logits_dataset_ready(args: PipelineArgs) -> bool:
    root = Path(args.output_root) / args.dataset / "shared" / "logits_dataset"

    expected = [
        root / "train" / f"ALL_tokens_{args.adjust}.parquet",
        root / "val" / f"ALL_tokens_{args.adjust}.parquet",
        root / "test" / f"ALL_tokens_{args.adjust}.parquet",
    ]

    return all(path_exists_nonempty(p) for p in expected)


def iter_logits_ready(args: PipelineArgs, iteration: int) -> bool:
    root = Path(args.output_root) / args.dataset / f"iter_{iteration}" / "logits"

    expected = [
        root / "train" / f"ALL_tokens_{args.adjust}.parquet",
        root / "val" / f"ALL_tokens_{args.adjust}.parquet",
        root / "test" / f"ALL_tokens_{args.adjust}.parquet",
    ]

    return all(path_exists_nonempty(p) for p in expected)


def run_check_price_quality(args: PipelineArgs) -> None:
    cmd = [
        args.python_exec,
        "src/pipeline/run_check_price_quality.py",
        "--dataset", args.dataset,
        "--config",  args.codebook_config,
        "--output_root", args.output_root,
        "--adjust",  args.adjust,
    ]
    if args.force_regenerate_blacklist:
        cmd.append("--force")

    log_file = Path(args.output_root) / args.dataset / "blacklist" / "logs" / "check_price_quality.log"
    run_command(cmd, dry_run=args.dry_run, log_file=log_file)
    

def run_fit_codebook(args: PipelineArgs) -> None:
    cmd = [
        args.python_exec,
        "src/pipeline/run_fit_codebook.py",
        "--config", args.codebook_config,
        "--dataset", args.dataset,
        "--output_root", args.output_root,
        "--adjust", args.adjust,
    ]
    if args.force_fit_codebook:
        cmd.append("--force")

    log_file = Path(args.output_root) / args.dataset / "codebook" / "logs" / "fit_codebook.log"
    run_command(cmd, dry_run=args.dry_run, log_file=log_file)


def run_labeling(args: PipelineArgs, iteration: int) -> None:
    cmd = [
        args.python_exec,
        "src/pipeline/run_labeling.py",
        "--config",
        args.label_config,
        "--dataset",
        args.dataset,
        "--iteration",
        str(iteration),
        "--output_root",
        args.output_root,
        "--adjust",
        args.adjust,
        "--codebook_dir", 
        str(Path(args.output_root) / args.dataset / "codebook"),
        "--n_workers",
        str(args.n_workers),
        "--labeler_type", 
        str(args.labeler_type),
    ]
    if args.beta_model is not None:
            cmd += ["--beta_model", str(args.beta_model)]
    log_file = _log_dir(args, iteration) / "labeling.log"
    run_command(cmd, dry_run=args.dry_run, log_file=log_file)

def run_single_iteration_label_analysis(args: PipelineArgs, iteration: int) -> None:
    cmd = [
        args.python_exec,
        "src/pipeline/run_analyze_single_iteration_labels.py",
        "--dataset",
        args.dataset,
        "--iteration",
        str(iteration),
        "--output_root",
        args.output_root,
        "--adjust",
        args.adjust,
        "--codebook_dir",   # Added
        str(Path(args.output_root) / args.dataset / "codebook"),
        "--splits", "test",   # Analyze only the test split; its CSV covers the full data history.
    ]
    log_file = _log_dir(args, iteration) / "label_analysis_single.log"
    run_command(cmd, dry_run=args.dry_run, log_file=log_file)


def run_compare_iteration_labels(args: PipelineArgs, base_iteration: int, new_iteration: int) -> None:
    cmd = [
        args.python_exec,
        "src/pipeline/run_compare_iteration_labels.py",
        "--dataset",
        args.dataset,
        "--base_iteration",
        str(base_iteration),
        "--new_iteration",
        str(new_iteration),
        "--output_root",
        args.output_root,
        "--adjust",
        args.adjust,
        "--codebook_dir",   # Added
        str(Path(args.output_root) / args.dataset / "codebook"),
        "--splits", "test",   # Analyze only the test split; its CSV covers the full data history.
    ]
    log_file = _log_dir(args, new_iteration) / "label_analysis_compare.log"
    run_command(cmd, dry_run=args.dry_run, log_file=log_file)


def run_build_dataset(args: PipelineArgs, iteration: int, dataset_kind: str) -> None:
    cmd = [
        args.python_exec,
        "src/pipeline/run_build_dataset.py",
        "--config",
        args.build_config,
        "--dataset",
        args.dataset,
        "--iteration",
        str(iteration),
        "--output_root",
        args.output_root,
        "--dataset_kind",
        dataset_kind,
    ]
    if dataset_kind == "shared_logits_dataset":
        log_file = Path(args.output_root) / args.dataset / "shared" / "logs" / "build_dataset.log"
    else:
        log_file = _log_dir(args, iteration) / "build_dataset.log"
    run_command(cmd, dry_run=args.dry_run, log_file=log_file)


def run_train(args: PipelineArgs, iteration: int) -> None:
    cmd = [
        args.python_exec,
        "src/pipeline/run_train_transformer.py",
        "--config",
        args.train_config,
        "--dataset",
        args.dataset,
        "--iteration",
        str(iteration),
        "--output_root",
        args.output_root,
        "--seed",
        str(args.seed),
        "--adjust",
        args.adjust,
    ]
    if args.train_precision is not None:
        cmd += ["--precision", args.train_precision]
    log_file = _log_dir(args, iteration) / "train.log"
    run_command(cmd, dry_run=args.dry_run, log_file=log_file)


def run_test(args: PipelineArgs, iteration: int) -> None:
    cmd = [
        args.python_exec,
        "src/pipeline/run_test_transformer.py",
        "--dataset",
        args.dataset,
        "--iteration",
        str(iteration),
        "--output_root",
        args.output_root,
        "--seed",
        str(args.seed),
        "--adjust",
        args.adjust,
    ]
    log_file = _log_dir(args, iteration) / "test.log"
    run_command(cmd, dry_run=args.dry_run, log_file=log_file)


def run_export_logits(args: PipelineArgs, iteration: int) -> None:
    cmd = [
        args.python_exec,
        "src/pipeline/run_export_transformer_logits.py",
        "--dataset",
        args.dataset,
        "--iteration",
        str(iteration),
        "--output_root",
        args.output_root,
        "--seed",
        str(args.seed),
        "--adjust",
        args.adjust,
        "--all_splits",
        "--num_workers", 
        str(args.n_workers),   

    ]
    log_file = _log_dir(args, iteration) / "export_logits.log"
    run_command(cmd, dry_run=args.dry_run, log_file=log_file)


def resolved_train_precision(args: PipelineArgs) -> str:
    if args.train_precision is not None:
        precision = args.train_precision
    elif args.unified_config is not None:
        precision = args.unified_config.stages["train"]["training"].get("precision", "fp32")
    else:
        import yaml

        config = yaml.safe_load(Path(args.train_config).read_text(encoding="utf-8"))
        precision = config.get("training", {}).get("precision", "fp32")
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("training.precision must be fp32, bf16 or fp16")
    return precision


def validate_legacy_resume_precision(args: PipelineArgs, precision: str) -> None:
    """Reject precision changes along the linked iteration history."""
    if args.unified_config is not None:
        return  # Its frozen training configuration is already part of run identity.
    for iteration in range(args.start_iter):
        recorded_path = (Path(args.output_root) / args.dataset / f"iter_{iteration}"
                         / "model" / f"seed_{args.seed}" / "config.resolved.json")
        if not recorded_path.is_file():
            if precision != "fp32":
                raise FileNotFoundError(f"Cannot verify resume training precision: {recorded_path}")
            continue  # Preserve historical manual FP32 resumes without model metadata.
        recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
        previous_precision = recorded.get("training", {}).get("precision", "fp32")
        if previous_precision != precision:
            raise ValueError(
                f"Training precision differs from iter_{iteration}: {previous_precision} -> {precision}; "
                "use a new output_root and iteration 0"
            )


def print_header(args: PipelineArgs, precision: str | None = None) -> None:
    print("=" * 80, flush=True)
    print("Sequential DDSL pipeline", flush=True)
    if args.unified_config is not None:
        print(f"pipeline YAML: {args.unified_config.source_path}", flush=True)
        print(f"stage inputs : {args.unified_config.config_dir}", flush=True)
        print(f"beta_model   : {args.unified_config.stages['label']['ddsl']['beta_model']}", flush=True)
    print(f"dataset      : {args.dataset}", flush=True)
    print(f"label_config : {args.label_config}", flush=True)
    print(f"build_config : {args.build_config}", flush=True)
    print(f"train_config : {args.train_config}", flush=True)
    print(f"train precision: {precision or resolved_train_precision(args)} (validation/test/export: fp32)", flush=True)
    print(f"output_root  : {args.output_root}", flush=True)
    print(f"iter range   : {args.start_iter} -> {args.end_iter}", flush=True)
    print(f"seed         : {args.seed}", flush=True)
    print(f"adjust       : {args.adjust}", flush=True)
    print(f"python_exec  : {args.python_exec}", flush=True)
    print(f"skip_test    : {args.skip_test}", flush=True)
    print(f"skip_logits  : {args.skip_export_logits}", flush=True)
    print(f"dry_run      : {args.dry_run}", flush=True)
    print("=" * 80, flush=True)


def run_pipeline(args: PipelineArgs) -> None:
    precision = resolved_train_precision(args)
    validate_legacy_resume_precision(args, precision)
    if args.unified_config is not None:
        args.unified_config.prepare(dry_run=args.dry_run)
        if args.start_iter > 0:
            codebook_dir = Path(args.output_root) / args.dataset / "codebook"
            for name in ("geom_kmeans.joblib", "bins.joblib", "fit_config.json", "symbol_codebook_info.json"):
                if not path_exists_nonempty(codebook_dir / name):
                    raise FileNotFoundError(f"Resume codebook artifact missing or empty: {codebook_dir / name}")
            if not shared_logits_dataset_ready(args):
                raise FileNotFoundError("Resume requires nonempty train/val/test shared logits datasets")
            if not iter_logits_ready(args, args.start_iter - 1):
                raise FileNotFoundError(f"Resume requires nonempty train/val/test logits from iter_{args.start_iter - 1}")
    print_header(args, precision)

    # 0a. Check data quality and generate the blacklist.
    if args.start_iter == 0:
        run_check_price_quality(args)
    # 0b. Fit codebook
    # build the codebook for tokenizing the price features into discrete values.
    # If force_fit_codebook, the codebook will be computed even there exists one.

    if args.start_iter == 0:
        run_fit_codebook(args)
    else:
        print(
            f"[INFO] start_iter={args.start_iter}; skip codebook fitting. "
            "Assume codebook already exists.",
            flush=True,
        )

    for iteration in range(args.start_iter, args.end_iter + 1):
        print("\n" + "#" * 80, flush=True)
        print(f"# ITERATION {iteration}", flush=True)
        print("#" * 80, flush=True)

        if iteration > 0 and not args.dry_run:
            prev_iter = iteration - 1
            if not iter_logits_ready(args, prev_iter):
                raise FileNotFoundError(
                    f"iter_{iteration} labeling needs previous logits, but they are missing:\n"
                    f"{Path(args.output_root) / args.dataset / f'iter_{prev_iter}' / 'logits'}"
                )

        # 1. Label
        #
        # iter_0:
        #   run_labeling.py should use DPLabeler.
        #
        # iter_1+:
        #   run_labeling.py should use DDSLLabeler and read iter_{k-1}/logits.
        run_labeling(args, iteration)

        # 2. Analyze labels
        # Analysis segment qualities of the labels:
        run_single_iteration_label_analysis(args, iteration)

        # Compare labels between two iteration:
        # This should be built after iter_0 labels exist.
        if iteration > 0:
            run_compare_iteration_labels(args, iteration - 1, iteration)

        # 3. Build shared logits dataset once.
        #
        # This should be built after iter_0 labels exist.
        # It is shared by all iterations.
        if iteration == 0:
            if args.force_rebuild_shared_logits_dataset or not shared_logits_dataset_ready(args):
                print("[INFO] Building shared logits dataset.", flush=True)
                run_build_dataset(args, iteration, dataset_kind="shared_logits_dataset")
            else:
                print("[INFO] Shared logits dataset already exists. Skip rebuilding.", flush=True)

        # 4. Build per-iteration dataset.
        #
        # This must run every iteration because labels change.
        run_build_dataset(args, iteration, dataset_kind="dataset")

        # 5. Train.
        run_train(args, iteration)

        # 6. Test.
        if not args.skip_test:
            run_test(args, iteration)
        else:
            print("[INFO] skip_test=True. Skip test.", flush=True)

        # 7. Export logits.
        #
        # This must run every iteration because model changes.
        # It reads shared/logits_dataset and writes iter_k/logits.
        if not args.skip_export_logits:
            if not args.dry_run and not shared_logits_dataset_ready(args):
                raise FileNotFoundError(
                    "Shared logits dataset is missing. Cannot export logits.\n"
                    f"Expected under: {Path(args.output_root) / args.dataset / 'shared' / 'logits_dataset'}"
                )
            run_export_logits(args, iteration)
        else:
            print("[INFO] skip_export_logits=True. Skip export logits.", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("Sequential DDSL pipeline finished successfully.", flush=True)
    print("=" * 80, flush=True)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
