from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.utils import get_best_device, load_yaml
from data.kqgm_dataset import KQGMJointDataset, collate_kqgm_joint
from models import TransformerEncoderClassifier
from states import NUM_STATES, STATE_NAMES
from training.evaluator import evaluate


# ---------------------------------------------------------------------
# Model / checkpoint
# ---------------------------------------------------------------------


def build_model(cfg: dict[str, Any]) -> TransformerEncoderClassifier:
    m = cfg["model"]

    return TransformerEncoderClassifier(
        num_K=int(m["num_K"]),
        num_Q=int(m["num_Q"]),
        num_G=int(m["num_G"]),
        num_M=int(m["num_M"]),
        num_classes=NUM_STATES,
        d_model=int(m.get("d_model", 64)),
        nhead=int(m.get("nhead", 4)),
        num_layers=int(m.get("num_layers", 3)),
        dim_feedforward=int(m.get("dim_feedforward", 128)),
        dropout=float(m.get("dropout", 0.3)),
        max_seq_len=int(m.get("max_seq_len", cfg["training"]["seq_len"])),
        pooling=str(m.get("pooling", "cls")),
    )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)

    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    return ckpt


# ---------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------


def metrics_to_json_safe(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v

    return out


def print_per_class_report(cm: np.ndarray) -> None:
    eps = 1e-12

    print("\n--- Per-class metrics ---")
    print(f"{'Class':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)

    for c, name in enumerate(STATE_NAMES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        print(
            f"{name:<24} "
            f"{precision:>10.4f} "
            f"{recall:>10.4f} "
            f"{f1:>10.4f} "
            f"{int(support):>10d}"
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate DDSL joint-state classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Existing / manual mode
    ap.add_argument("--config", default=None, help="Training YAML config.")
    ap.add_argument("--checkpoint", default=None, help="Path to best.pt.")
    ap.add_argument("--test_parquet", default=None, help="Test token parquet.")
    ap.add_argument("--out_json", default=None)

    # Pipeline mode
    ap.add_argument("--dataset", default=None, help="Dataset name, e.g., SSE50.")
    ap.add_argument("--iteration", type=int, default=None, help="DDSL iteration id.")
    ap.add_argument("--output_root", default="outputs", help="Root output directory.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adjust", default="qfq")

    # Evaluation options
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--checkpoint_name", default="best.pt")
    ap.add_argument("--dry_run", action="store_true")

    return ap.parse_args()


# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------


def resolve_pipeline_paths(
    *,
    dataset: str,
    iteration: int,
    output_root: str,
    seed: int,
    adjust: str,
    checkpoint_name: str,
) -> tuple[Path, Path, Path, Path]:
    """
    Returns:
        config_path, checkpoint_path, test_parquet, out_json

    Pipeline convention:
        model dir:
            outputs/{dataset}/iter_{iteration}/model/seed_{seed}

        test parquet:
            outputs/{dataset}/iter_{iteration}/dataset/test/ALL_tokens_{adjust}.parquet

        eval json:
            outputs/{dataset}/iter_{iteration}/eval/seed_{seed}/test_metrics.json
    """
    if iteration < 0:
        raise ValueError("--iteration must be >= 0")

    root = Path(output_root)
    iter_dir = root / dataset / f"iter_{iteration}"

    model_dir = iter_dir / "model" / f"seed_{seed}"
    config_path = model_dir / "config.resolved.json"
    checkpoint_path = model_dir / checkpoint_name
    test_parquet = iter_dir / "dataset" / "test" / f"ALL_tokens_{adjust}.parquet"
    out_json = iter_dir / "eval" / f"seed_{seed}" / "test_metrics.json"

    return config_path, checkpoint_path, test_parquet, out_json


def resolve_manual_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    missing = []

    if args.config is None:
        missing.append("--config")
    if args.checkpoint is None:
        missing.append("--checkpoint")
    if args.test_parquet is None:
        missing.append("--test_parquet")

    if missing:
        raise ValueError(
            "Manual mode requires: "
            + ", ".join(missing)
            + ". Or use pipeline mode with --dataset --iteration."
        )

    return (
        Path(args.config),
        Path(args.checkpoint),
        Path(args.test_parquet),
        Path(args.out_json) if args.out_json else None,
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    using_pipeline = args.dataset is not None or args.iteration is not None

    if using_pipeline:
        if args.dataset is None:
            raise ValueError("Pipeline mode requires --dataset.")
        if args.iteration is None:
            raise ValueError("Pipeline mode requires --iteration.")

        return resolve_pipeline_paths(
            dataset=args.dataset,
            iteration=args.iteration,
            output_root=args.output_root,
            seed=args.seed,
            adjust=args.adjust,
            checkpoint_name=args.checkpoint_name,
        )

    return resolve_manual_paths(args)


def validate_paths(
    *,
    config_path: Path,
    checkpoint_path: Path,
    test_parquet: Path,
) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not test_parquet.exists():
        raise FileNotFoundError(f"test_parquet not found: {test_parquet}")


def print_resolved_paths(
    *,
    config_path: Path,
    checkpoint_path: Path,
    test_parquet: Path,
    out_json: Path | None,
) -> None:
    print("\nResolved test configuration")
    print("-" * 72)
    print(f"config       : {config_path}")
    print(f"checkpoint   : {checkpoint_path}")
    print(f"test_parquet : {test_parquet}")
    print(f"out_json     : {out_json}")
    print("-" * 72)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    config_path, checkpoint_path, test_parquet, out_json = resolve_paths(args)

    print_resolved_paths(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        test_parquet=test_parquet,
        out_json=out_json,
    )

    validate_paths(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        test_parquet=test_parquet,
    )

    if args.dry_run:
        print("\nDry run finished.")
        return

    cfg = load_yaml(config_path)

    device = get_best_device()
    print(f"[INFO] Device: {device}")

    seq_len = int(cfg["training"]["seq_len"])
    batch_size = args.batch_size or int(
        cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    )

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(model, checkpoint_path, device)

    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    if "epoch" in ckpt:
        print(f"[INFO] Saved epoch: {ckpt['epoch']}")
    if "val_metrics" in ckpt:
        print(f"[INFO] Saved val metrics: {ckpt['val_metrics']}")

    test_df = pd.read_parquet(test_parquet)
    test_ds = KQGMJointDataset(test_df, seq_len=seq_len)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_kqgm_joint,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    print(f"[INFO] Test parquet rows: {len(test_df):,}")
    print(f"[INFO] Test samples after windowing: {len(test_ds):,}")

    # Dummy loss only for reporting test loss.
    loss_fn = torch.nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"joint_acc     : {metrics['joint_acc']:.4f}")
    print(f"joint_macroF1 : {metrics['joint_macro_f1']:.4f}")
    print(f"current_acc   : {metrics['current_acc_from_joint']:.4f}")
    print(f"future_acc    : {metrics['future_acc_from_joint']:.4f}")
    print(f"test_loss     : {metrics['loss']:.4f}")

    print_per_class_report(metrics["confusion_matrix"])

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)

        save_data: dict[str, Any] = {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "test_parquet": str(test_parquet),
            "seq_len": seq_len,
            "batch_size": batch_size,
        }
        save_data.update(metrics_to_json_safe(metrics))

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"\n[INFO] Saved test metrics -> {out_json}")


if __name__ == "__main__":
    main()