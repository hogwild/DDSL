"""
CLI entry point for building K/Q/G/M token datasets from labeled CSVs.

Examples
--------
Build per-iteration training/evaluation datasets:

    python src/pipeline/run_build_dataset.py \
        --config configs/base/sse50_build_dataset.yaml \
        --dataset SSE50 \
        --iteration 0 \
        --output_root outputs \
        --dataset_kind dataset

Build the shared logits dataset once after iter_0 labeling:

    python src/pipeline/run_build_dataset.py \
        --config configs/base/sse50_build_dataset.yaml \
        --dataset SSE50 \
        --iteration 0 \
        --output_root outputs \
        --dataset_kind shared_logits_dataset

The shared logits dataset is used by run_export_transformer_logits.py for every iteration.
It contains the same K/Q/G/M token input sequences across iterations.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml


# Allow running from repository root without installing the package.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.build_dataset import run_split
from data.check_price_quality import load_blacklist

from common.config_utils import (
    get_nested,
    infer_dataset_name,
    infer_adjust,
    infer_market,
    apply_runtime_dataset_info,
)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build K/Q/G/M token parquet datasets via YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Existing arguments
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    ap.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Splits to run, e.g. --splits train val. Default: all splits in config.",
    )
    ap.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Override options.max_files for quick debugging.",
    )

    # Pipeline arguments
    ap.add_argument(
        "--dataset",
        default=None,
        help="Dataset name, e.g., SSE50. If omitted, try to infer from YAML.",
    )
    ap.add_argument(
        "--iteration",
        type=int,
        default=None,
        help=(
            "DDSL iteration id. Required when dataset_kind='dataset'. "
            "For shared_logits_dataset, this is ignored logically and iter_0 labels are used."
        ),
    )
    ap.add_argument(
        "--output_root",
        default=None,
        help="Root output directory, e.g., outputs.",
    )
    ap.add_argument(
        "--dataset_kind",
        choices=["dataset", "shared_logits_dataset"],
        default="dataset",
        help=(
            "dataset: per-iteration train/val/test dataset with current labels. "
            "shared_logits_dataset: one-time dataset for exporting logits across all iterations."
        ),
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print resolved split paths without building datasets.",
    )

    return ap.parse_args()


def apply_pipeline_overrides(
    cfg: dict[str, Any],
    *,
    config_path: str,
    dataset: str | None,
    iteration: int | None,
    output_root: str | None,
    dataset_kind: str,
) -> dict[str, Any]:
    """
    Patch YAML in memory for the iterative DDSL pipeline.

    There are two dataset kinds:

    1. dataset
       Per-iteration dataset used for training, validation, and testing.
       This dataset depends on the labels of the current iteration.

       Paths:
           raw_dir = outputs/{dataset}/iter_{iteration}/labels/{split}
           out_dir = outputs/{dataset}/iter_{iteration}/dataset/{split}

       Date range:
           Use split start_date/end_date from the base YAML.

    2. shared_logits_dataset
       One-time dataset used only for exporting logits.
       It is shared by all iterations because the model input tokens are the same
       across iterations. Only the model changes.

       Paths:
           raw_dir = outputs/{dataset}/iter_0/labels/{split}
           out_dir = outputs/{dataset}/shared/logits_dataset/{split}

       Date range:
           Force start_date = None for every split, while keeping end_date.
           This ensures that val/test logits contain data before the split start_date,
           so DDSL labeling has enough historical logits near the beginning of each split.

    Notes:
        - shared_logits_dataset is built from iter_0 labels mainly because the current
          build_dataset implementation expects labeled CSVs with joint_state.
        - The labels inside shared_logits_dataset are not used by export_logits.
          Exporting logits uses K/Q/G/M tokens plus symbol/date metadata.
    """
    cfg = copy.deepcopy(cfg)

    if dataset_kind not in {"dataset", "shared_logits_dataset"}:
        raise ValueError(
            f"Unknown dataset_kind={dataset_kind!r}. "
            "Expected 'dataset' or 'shared_logits_dataset'."
        )

    if "splits" not in cfg or not isinstance(cfg["splits"], dict):
        raise ValueError("YAML must contain a 'splits' dictionary.")

    dataset_name = infer_dataset_name(cfg, config_path=config_path, cli_dataset=dataset)
    adjust_name = infer_adjust(cfg)
    market_name = infer_market(cfg)
    apply_runtime_dataset_info(
        cfg,
        dataset_name=dataset_name,
        market=market_name,
        adjust=adjust_name,
    )
    # dataset_name = infer_dataset_name(cfg, config_path, dataset)
    output_root_path = Path(output_root or "outputs")
    
    # Also override tokenizer.dir.
    cfg.setdefault("tokenizer", {})
    cfg["tokenizer"]["dir"] = str(output_root_path / dataset_name / "codebook")

    if dataset_kind == "dataset":
        if iteration is None:
            raise ValueError("--iteration is required when dataset_kind='dataset'.")
        if iteration < 0:
            raise ValueError("--iteration must be >= 0")

        source_iteration = iteration
        labels_root = output_root_path / dataset_name / f"iter_{iteration}" / "labels"
        dataset_root = output_root_path / dataset_name / f"iter_{iteration}" / "dataset"

    else:
        # shared_logits_dataset is intentionally based on iter_0 labels and shared
        # across all iterations.
        source_iteration = 0
        labels_root = output_root_path / dataset_name / "iter_0" / "labels"
        dataset_root = output_root_path / dataset_name / "shared" / "logits_dataset"

    cfg["runtime"].update(
        {
            "iteration": iteration,
            "source_iteration": source_iteration,
            "output_root": str(output_root_path),
            "labels_root": str(labels_root),
            "dataset_root": str(dataset_root),
            "dataset_kind": dataset_kind,
        }
    )

    for split_name, split_cfg in cfg["splits"].items():
        if not isinstance(split_cfg, dict):
            raise ValueError(f"split config for {split_name!r} must be a dictionary.")

        split_cfg["raw_dir"] = str(labels_root / split_name)
        split_cfg["out_dir"] = str(dataset_root / split_name)

        if dataset_kind == "shared_logits_dataset":
            # Keep end_date but include all historical data before split start_date.
            split_cfg["start_date"] = None

    return cfg


def print_resolved_config(cfg: dict[str, Any], splits_to_run: list[str]) -> None:
    print("\nResolved build-dataset configuration")
    print("-" * 72)
    print(f"dataset        : {cfg.get('runtime', {}).get('dataset', None)}")
    print(f"iteration      : {cfg.get('runtime', {}).get('iteration', None)}")
    print(f"source_iter    : {cfg.get('runtime', {}).get('source_iteration', None)}")
    print(f"dataset_kind   : {cfg.get('runtime', {}).get('dataset_kind', None)}")
    print(f"output_root    : {cfg.get('runtime', {}).get('output_root', None)}")
    print(f"labels_root    : {cfg.get('runtime', {}).get('labels_root', None)}")
    print(f"dataset_root   : {cfg.get('runtime', {}).get('dataset_root', None)}")
    print(f"tokenizer_dir  : {cfg.get('tokenizer', {}).get('dir', None)}")
    print(f"adjust         : {cfg.get('dataset', {}).get('adjust', None)}")
    print(f"splits         : {splits_to_run}")
    print("-" * 72)

    for split_name in splits_to_run:
        split_cfg = cfg["splits"][split_name]
        print(f"[{split_name}]")
        print(f"  raw_dir   : {split_cfg.get('raw_dir')}")
        print(f"  out_dir   : {split_cfg.get('out_dir')}")
        print(f"  date range: {split_cfg.get('start_date')} -> {split_cfg.get('end_date')}")


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Empty YAML config: {args.config}")

    cfg = apply_pipeline_overrides(
        cfg,
        config_path=args.config,
        dataset=args.dataset,
        iteration=args.iteration,
        output_root=args.output_root,
        dataset_kind=args.dataset_kind,
    )

    splits_to_run = args.splits or list(cfg["splits"].keys())

    for split_name in splits_to_run:
        if split_name not in cfg["splits"]:
            raise ValueError(
                f"Split '{split_name}' not defined in {args.config}. "
                f"Available: {list(cfg['splits'].keys())}"
            )

    print_resolved_config(cfg, splits_to_run)

    blacklist_path_str = cfg.get("data", {}).get("blacklist_path")
    if blacklist_path_str:
        blacklist_path = Path(blacklist_path_str)
        blacklist = load_blacklist(blacklist_path) if blacklist_path.exists() else set()
        if blacklist:
            print(f"[INFO] Blacklist loaded: {len(blacklist)} symbols from {blacklist_path}")
    else:
        blacklist = set()

    if args.dry_run:
        print("\nDry run finished.")
        return

    for split_name in splits_to_run:
        run_split(
            split_name=split_name,
            split_cfg=cfg["splits"][split_name],
            shared=cfg,
            blacklist=blacklist,        # Added
            max_files_override=args.max_files,
        )

    print("\nAll splits done.")


if __name__ == "__main__":
    main()