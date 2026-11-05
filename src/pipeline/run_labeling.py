from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any
import math
import pandas as pd
import yaml

# Allow running from repository root without installing the package.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from labeling import DPLabeler, DDSLLabeler, PseudoLabeler, cfg_from_yaml
from common.config_utils import (
    get_nested,
    infer_dataset_name,
    infer_adjust,
    infer_market,
    apply_runtime_dataset_info,
)
from data.check_price_quality import load_blacklist
# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------


def parse_date_arg(s: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(s) if s else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run DP or DDSL 9-state labeling.")

    # Existing arguments
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--splits", nargs="*", default=None, help="Optional split names to run")

    # New pipeline arguments
    p.add_argument(
        "--dataset",
        default=None,
        help="Dataset name, e.g., SSE50. If omitted, try to infer from YAML.",
    )
    p.add_argument(
        "--iteration",
        type=int,
        default=None,
        help=(
            "DDSL iteration id. "
            "If provided, iter_0 uses DPLabeler and iter_1+ uses DDSLLabeler."
        ),
    )
    p.add_argument(
        "--output_root",
        default=None,
        help="Root output directory, e.g., outputs.",
    )
    p.add_argument(
        "--labeler_type",
        choices=["auto", "dp", "ddsl", "pseudo"],
        default="auto",
        help=(
            "Labeler override. "
            "auto: use iteration rule if --iteration is provided; otherwise use YAML labeler.type."
        ),
    )
    p.add_argument(
        "--adjust",
        default=None,
        help="Adjusted price suffix, e.g., hfq or qfq. Used to resolve logits file names.",
    )
    p.add_argument(
        "--codebook_dir",
        default=None,
        help="Path to codebook directory containing symbol_codebook_info.json.",
    )
    p.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for labeling. Default=1 (serial).",
    )

    p.add_argument(
        "--beta_model",
        type=float,
        default=None,
        help=(
            "Override ddsl.beta_model from the YAML. "
            "Omit to use the YAML value. Only meaningful for DDSLLabeler; "
            "DPLabeler ignores it."
        ),
    )

    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print resolved config information without running labeling.",
    )

    return p


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Labeler selection
# ---------------------------------------------------------------------


def resolve_labeler_type(y: dict[str, Any]) -> str:
    labeler_type = y.get("labeler", {}).get("type", None)

    if labeler_type is None:
        raise ValueError(
            "Missing labeler.type in YAML. "
            "Please set labeler.type to 'dp' or 'ddsl', "
            "or pass --iteration so the pipeline can set it automatically."
        )

    labeler_type = str(labeler_type).lower()

    if labeler_type not in {"dp", "ddsl", "pseudo"}:
        raise ValueError(
            f"Unknown labeler.type={labeler_type!r}. "
            "Expected 'dp', 'ddsl', or 'pseudo'."
        )

    return labeler_type


def build_labeler(y: dict[str, Any], cfg: Any):
    labeler_type = resolve_labeler_type(y)

    if labeler_type == "dp":
        return DPLabeler(cfg)

    if labeler_type == "ddsl":
        return DDSLLabeler(cfg)
    
    if labeler_type == "pseudo":
        return PseudoLabeler(cfg)

    raise RuntimeError(f"Unexpected labeler_type={labeler_type!r}")


# ---------------------------------------------------------------------
# Pipeline overrides
# ---------------------------------------------------------------------

def apply_pipeline_overrides(
    y: dict[str, Any],
    *,
    config_path: str,
    dataset: str | None,
    iteration: int | None,
    output_root: str | None,
    labeler_type_arg: str,
    adjust: str | None = None,
    codebook_dir_arg: str | None = None,   # Added
) -> dict[str, Any]:
    """
    Apply DDSL iteration overrides to the YAML dictionary in memory.

    Rules:
        - If --iteration is not provided:
            * Keep YAML paths and YAML labeler.type unless --labeler_type overrides it.

        - If --iteration is provided:
            * iter_0 uses DPLabeler
            * iter_1+ uses DDSLLabeler
            * output labels are written to:
                {output_root}/{dataset}/iter_{k}/labels/{split}
            * DDSL model scores are read from a concrete parquet file:
                {output_root}/{dataset}/iter_{k-1}/logits/{split}/ALL_tokens_{adjust}.parquet

    This function does not modify beta_model. beta_model remains an energy parameter.
    """
    y = copy.deepcopy(y)

    # Case 1: no iteration mode. Only optionally override labeler.type.
    if iteration is None:
        if labeler_type_arg != "auto":
            y.setdefault("labeler", {})
            y["labeler"]["type"] = labeler_type_arg

        # If neither YAML nor CLI provides labeler.type, fail early.
        resolve_labeler_type(y)
        return y

    if iteration < 0:
        raise ValueError("--iteration must be >= 0")

    dataset_name = infer_dataset_name(y, config_path, dataset)
    market_name = infer_market(y)
    adjust_name = infer_adjust(y, adjust)

    apply_runtime_dataset_info(
        y,
        dataset_name=dataset_name,
        market=market_name,
        adjust=adjust_name,
    )

    output_root_path = Path(output_root or "outputs")

    cur_iter_dir = output_root_path / dataset_name / f"iter_{iteration}"
    label_root = cur_iter_dir / "labels"

    if labeler_type_arg == "auto":
        effective_labeler_type = "dp" if iteration == 0 else "ddsl"
    elif labeler_type_arg == "pseudo":
        effective_labeler_type = "dp" if iteration == 0 else "pseudo"
    else:
        effective_labeler_type = labeler_type_arg

    if iteration == 0 and effective_labeler_type == "ddsl":
        raise ValueError(
            "iteration 0 should normally use DPLabeler, but labeler_type='ddsl'."
            "DDSLLabeler and PseudoLabeler need previous-iteration logits, which do not exist for iter_0."
        )
        
    codebook_dir = (
        Path(codebook_dir_arg)
        if codebook_dir_arg is not None
        else output_root_path / dataset_name / "codebook"
    )

    y["runtime"].update(
        {
            "iteration": iteration,
            "output_root": str(output_root_path),
            "label_root": str(label_root),
            "codebook_dir": str(codebook_dir),   
        }
    )

    y.setdefault("labeler", {})
    y["labeler"]["type"] = effective_labeler_type

    # Patch split-level output label dirs and score paths.
    if "splits" not in y or not isinstance(y["splits"], dict):
        raise ValueError("YAML must contain a 'splits' dictionary.")

    for split_name, split_cfg in y["splits"].items():
        if not isinstance(split_cfg, dict):
            raise ValueError(f"split config for {split_name!r} must be a dictionary.")

        split_out_dir = label_root / split_name
        split_cfg["out_dir"] = str(split_out_dir)

        if effective_labeler_type in {"ddsl", "pseudo"}:
            prev_logits_file = (
                output_root_path
                / dataset_name
                / f"iter_{iteration - 1}"
                / "logits"
                / split_name
                / f"ALL_tokens_{adjust_name}.parquet"
            )

            # Important:
            # load_model_log_probs() expects a concrete .parquet/.csv file,
            # not a directory.
            split_cfg["model_score_path"] = str(prev_logits_file)
            split_cfg.pop("model_score_dir", None)

            y.setdefault("ddsl", {})
            y["ddsl"]["model_score_path"] = str(prev_logits_file)

        else:
            split_cfg.pop("model_score_dir", None)
            split_cfg.pop("model_score_path", None)

    return y


def apply_beta_model_override(
    y: dict[str, Any],
    beta_model: float | None,
) -> dict[str, Any]:
    """
    Override ddsl.beta_model from the CLI.

    Strictly opt-in: if beta_model is None the YAML is returned untouched.

    Requires ddsl.beta_model to already exist in the YAML. A config without
    that key is not a labeling config (most likely the wrong --config was
    passed), and creating the key here would hide that until much later.
    """
    if beta_model is None:
        return y

    if not math.isfinite(beta_model):
        raise ValueError(f"--beta_model must be finite, got {beta_model!r}.")
    if beta_model < 0:
        raise ValueError(f"--beta_model must be >= 0, got {beta_model!r}.")

    if not isinstance(y.get("ddsl"), dict) or "beta_model" not in y["ddsl"]:
        raise ValueError(
            "--beta_model was given but the YAML has no 'ddsl.beta_model' key. "
            "Check that --config points at a labeling config."
        )

    y = copy.deepcopy(y)
    old = y["ddsl"]["beta_model"]
    y["ddsl"]["beta_model"] = beta_model
    print(f"[INFO] beta_model overridden by CLI: {old} -> {beta_model}", flush=True)
    return y


# ---------------------------------------------------------------------
# Split execution
# ---------------------------------------------------------------------


def resolve_model_score_path(split_cfg: dict[str, Any], y: dict[str, Any]) -> str | None:
    return split_cfg.get(
        "model_score_path",
        split_cfg.get(
            "model_score_dir",
            y.get("ddsl", {}).get(
                "model_score_path",
                y.get("ddsl", {}).get("model_score_dir"),
            ),
        ),
    )


def validate_split_before_run(
    *,
    split_name: str,
    labeler_type: str,
    out_dir: Path,
    model_score_path: str | None,
) -> None:
    if labeler_type in {"ddsl", "pseudo"} and model_score_path is None:
        raise ValueError(
            f"split={split_name!r}: labeler.type=={labeler_type!r}, "
            "but model_score_path/model_score_dir is missing."
        )

    ensure_dir(out_dir)


def run_split(split_name: str, split_cfg: dict[str, Any], y: dict[str, Any], dry_run: bool = False, n_workers: int = 1) -> None:
    cfg = cfg_from_yaml(y)
    labeler = build_labeler(y, cfg)
    labeler_type = resolve_labeler_type(y)

    io = y.get("io", {})
    if "in_dir" not in io:
        raise ValueError("YAML must contain io.in_dir")

    in_dir = Path(io["in_dir"])
    pattern = io.get("pattern", "*.csv")
    out_dir = Path(split_cfg["out_dir"])

    start_date = parse_date_arg(split_cfg.get("start_date"))
    end_date = parse_date_arg(split_cfg.get("end_date"))
    model_score_path = resolve_model_score_path(split_cfg, y)
    codebook_dir = y.get("runtime", {}).get("codebook_dir")  

    validate_split_before_run(
        split_name=split_name,
        labeler_type=labeler_type,
        out_dir=out_dir,
        model_score_path=model_score_path,
    )

    print("\n" + "=" * 72)
    print(f"SPLIT        : {split_name}")
    print(f"labeler.type : {labeler_type}")
    print(f"in_dir       : {in_dir}")
    print(f"pattern      : {pattern}")
    print(f"out_dir      : {out_dir}")
    print(f"date range   : {split_cfg.get('start_date')} -> {split_cfg.get('end_date')}")
    print(f"beta_model   : {getattr(cfg, 'beta_model', None)}")
    print(f"codebook_dir : {codebook_dir}")   
    if model_score_path is not None:
        print(f"scores       : {model_score_path}")
    print("=" * 72)

    blacklist_path_str = y.get("data", {}).get("blacklist_path")
    if blacklist_path_str:
        blacklist_path = Path(blacklist_path_str)
        blacklist = load_blacklist(blacklist_path) if blacklist_path.exists() else set()
        if blacklist:
            print(f"[INFO] Blacklist loaded: {len(blacklist)} symbols from {blacklist_path}")
    else:
        blacklist = set()

    if dry_run:
        return

    if isinstance(labeler, DDSLLabeler): # Covers both DDSLLabeler and PseudoLabeler.
        labeler.label_folder(
            in_dir=in_dir,
            out_dir=out_dir,
            pattern=pattern,
            start_date=start_date,
            end_date=end_date,
            model_score_path=model_score_path,
            codebook_dir=codebook_dir,   
            blacklist=blacklist,          
            n_workers=n_workers,         
        )
    elif isinstance(labeler, DPLabeler):
        labeler.label_folder(
            in_dir=in_dir,
            out_dir=out_dir,
            pattern=pattern,
            start_date=start_date,
            end_date=end_date,
            blacklist=blacklist,          
            n_workers=n_workers,         
        )
    else:
        raise RuntimeError(f"Unexpected labeler object: {type(labeler)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    if y is None:
        raise ValueError(f"Empty YAML config: {args.config}")

    y = apply_pipeline_overrides(
        y,
        config_path=args.config,
        dataset=args.dataset,
        iteration=args.iteration,
        output_root=args.output_root,
        labeler_type_arg=args.labeler_type,
        adjust=args.adjust,
        codebook_dir_arg=args.codebook_dir, 
    )
    
    if args.beta_model is not None and "beta_model_" in Path(args.config).name:
        raise ValueError(
            f"--beta_model={args.beta_model} was given together with a "
            f"beta-specific config ({args.config}). That is two sources of "
            f"truth for the same parameter. Use the plain label config "
            f"(for example, a generated pipeline_config/label.yaml), or drop --beta_model."
        )

    y = apply_beta_model_override(y, args.beta_model)

    splits = args.splits or list(y["splits"].keys())

    print("\nResolved labeling configuration")
    print("-" * 72)
    print(f"config       : {args.config}")
    print(f"market       : {y.get('runtime', {}).get('market', y.get('dataset', {}).get('market', 'cn'))}")
    print(f"dataset      : {y.get('runtime', {}).get('dataset', args.dataset)}")
    print(f"iteration    : {y.get('runtime', {}).get('iteration', args.iteration)}")
    print(f"labeler.type : {resolve_labeler_type(y)}")
    print(f"splits       : {splits}")
    print("-" * 72)

    for split_name in splits:
        if split_name not in y["splits"]:
            raise ValueError(
                f"Unknown split {split_name!r}. "
                f"Available: {list(y['splits'].keys())}"
            )
        run_split(split_name, y["splits"][split_name], y, dry_run=args.dry_run, n_workers=args.n_workers)

    if args.dry_run:
        print("\nDry run finished.")
    else:
        print("\nAll splits done.")


if __name__ == "__main__":
    main()
