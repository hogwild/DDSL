from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader


# Allows running this file before installing the package.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.config_utils import (
    get_nested,
    infer_dataset_name,
    infer_adjust,
    infer_market,
    apply_runtime_dataset_info,
)
from common.utils import build_out_dir, get_best_device, load_yaml, save_json, set_seed
from data.kqgm_dataset import KQGMJointDataset, collate_kqgm_joint
from models import TransformerEncoderClassifier
from states import NUM_STATES
from training import build_loss, compute_class_weights_from_ids, evaluate, train_one_epoch
from training.trainer import resolve_amp_dtype



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train DDSL encoder-transformer 1-head classifier.")

    # Existing arguments
    ap.add_argument("--config", required=True, help="Training YAML config.")
    ap.add_argument("--seed", type=int, default=None, help="Override training.seed")
    ap.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    ap.add_argument("--out_dir", type=str, default=None, help="Override output.out_dir")
    ap.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None,
                    help="Override training.precision (default: fp32). Validation stays FP32.")

    # New pipeline arguments
    ap.add_argument(
        "--dataset",
        default=None,
        help="Dataset name, e.g., SSE50. If omitted, try to infer from YAML.",
    )
    ap.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="DDSL iteration id. If provided, dataset/model paths are patched.",
    )
    ap.add_argument(
        "--output_root",
        default=None,
        help="Root output directory, e.g., outputs.",
    )
    ap.add_argument(
        "--adjust",
        default=None,
        help="File suffix used by run_build_dataset.py, e.g., qfq. If omitted, read from YAML.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print resolved paths and exit without training.",
    )

    return ap.parse_args()


def apply_basic_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.out_dir is not None:
        cfg.setdefault("output", {})["out_dir"] = args.out_dir
    training = cfg.setdefault("training", {})
    if getattr(args, "precision", None) is not None:
        training["precision"] = args.precision
    training.setdefault("precision", "fp32")

    return cfg


def apply_pipeline_overrides(
    cfg: dict[str, Any],
    *,
    config_path: str,
    dataset: str | None,
    iteration: int | None,
    output_root: str | None,
    adjust: str | None,
) -> dict[str, Any]:
    """
    Patch training config for iterative DDSL pipeline.

    If --iteration is provided:
        train_parquet = outputs/{dataset}/iter_{k}/dataset/train/ALL_tokens_{adjust}.parquet
        val_parquet   = outputs/{dataset}/iter_{k}/dataset/val/ALL_tokens_{adjust}.parquet
        output.out_dir = outputs/{dataset}/iter_{k}/model/seed_{seed}

    The training core remains unchanged.
    """
    cfg = copy.deepcopy(cfg)

    if iteration is None:
        return cfg

    if iteration < 0:
        raise ValueError("--iteration must be >= 0")

    dataset_name = infer_dataset_name(cfg, config_path, dataset)
    adjust_name = infer_adjust(cfg, adjust)
    market_name = infer_market(cfg)
    output_root_path = Path(output_root or "outputs")

    iter_dir = output_root_path / dataset_name / f"iter_{iteration}"
    dataset_root = iter_dir / "dataset"
    model_root = iter_dir / "model"

    train_parquet = dataset_root / "train" / f"ALL_tokens_{adjust_name}.parquet"
    val_parquet = dataset_root / "val" / f"ALL_tokens_{adjust_name}.parquet"

    seed = int(cfg.get("training", {}).get("seed", 42))
    out_dir = model_root / f"seed_{seed}"

    apply_runtime_dataset_info(
        cfg,
        dataset_name=dataset_name,
        market=market_name,
        adjust=adjust_name,
    )

    cfg["runtime"].update(
        {
            "iteration": iteration,
            "output_root": str(output_root_path),
            "dataset_root": str(dataset_root),
            "model_root": str(model_root),
        }
    )

    cfg.setdefault("dataset", {})
    cfg["dataset"]["train_parquet"] = str(train_parquet)
    cfg["dataset"]["val_parquet"] = str(val_parquet)

    cfg.setdefault("output", {})
    cfg["output"]["out_dir"] = str(out_dir)

    return cfg


def validate_train_config(cfg: dict[str, Any]) -> None:
    precision = cfg.get("training", {}).get("precision", "fp32")
    if not isinstance(precision, str) or precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("training.precision must be fp32, bf16 or fp16")
    required = [
        ("dataset.train_parquet", ["dataset", "train_parquet"]),
        ("dataset.val_parquet", ["dataset", "val_parquet"]),
        ("training.seq_len", ["training", "seq_len"]),
        ("training.batch_size", ["training", "batch_size"]),
        ("training.epochs", ["training", "epochs"]),
        ("training.lr", ["training", "lr"]),
        ("model.num_K", ["model", "num_K"]),
        ("model.num_Q", ["model", "num_Q"]),
        ("model.num_G", ["model", "num_G"]),
        ("model.num_M", ["model", "num_M"]),
        ("model.pooling", ["model", "pooling"]),
        ("output.out_dir", ["output", "out_dir"]),
    ]

    missing = [name for name, keys in required if get_nested(cfg, keys) is None]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    train_path = Path(cfg["dataset"]["train_parquet"])
    val_path = Path(cfg["dataset"]["val_parquet"])

    if not train_path.exists():
        raise FileNotFoundError(f"train_parquet not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"val_parquet not found: {val_path}")


def validate_output_precision(cfg: dict[str, Any]) -> None:
    """Do not overwrite a different precision experiment via the standalone CLI."""
    recorded_path = Path(cfg["output"]["out_dir"]) / "config.resolved.json"
    if not recorded_path.is_file():
        return
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    previous_precision = recorded.get("training", {}).get("precision", "fp32")
    precision = cfg["training"].get("precision", "fp32")
    if previous_precision != precision:
        raise ValueError(
            f"Output already contains {previous_precision} training results; "
            f"refusing to overwrite them with {precision}. Use a new output directory."
        )


def validate_iteration_precision(cfg: dict[str, Any]) -> None:
    """Also guard direct run_train_transformer --iteration calls outside the pipeline runner."""
    runtime = cfg.get("runtime", {})
    iteration = runtime.get("iteration", 0)
    if iteration <= 0:
        return
    seed = cfg["training"].get("seed", 42)
    previous = (Path(runtime["output_root"]) / runtime["dataset"] / f"iter_{iteration - 1}"
                / "model" / f"seed_{seed}" / "config.resolved.json")
    precision = cfg["training"].get("precision", "fp32")
    if not previous.is_file():
        if precision != "fp32":
            raise ValueError(f"Cannot verify previous training precision: {previous}; use a new output_root and iteration 0")
        return  # Preserve legacy FP32 standalone usage.
    previous_cfg = json.loads(previous.read_text(encoding="utf-8"))
    previous_precision = previous_cfg.get("training", {}).get("precision", "fp32")
    if previous_precision != precision:
        raise ValueError(f"Training precision changed from {previous_precision} to {precision}; use a new output_root and iteration 0")


def print_resolved_config(cfg: dict[str, Any]) -> None:
    print("\nResolved training configuration")
    print("-" * 72)
    print(f"dataset       : {cfg.get('runtime', {}).get('dataset', None)}")
    print(f"iteration     : {cfg.get('runtime', {}).get('iteration', None)}")
    print(f"adjust        : {cfg.get('runtime', {}).get('adjust', None)}")
    print(f"seed          : {cfg.get('training', {}).get('seed', None)}")
    print(f"epochs        : {cfg.get('training', {}).get('epochs', None)}")
    print(f"precision     : {cfg.get('training', {}).get('precision', 'fp32')} (validation/test/logits: fp32)")
    print(f"train_parquet : {cfg.get('dataset', {}).get('train_parquet', None)}")
    print(f"val_parquet   : {cfg.get('dataset', {}).get('val_parquet', None)}")
    print(f"out_dir       : {cfg.get('output', {}).get('out_dir', None)}")
    print("-" * 72)


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer):
    name = cfg["training"].get("scheduler", "none")
    if name == "none":
        return None
    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.9, patience=3, threshold=1e-3,
            threshold_mode="abs", cooldown=1, min_lr=5e-5
        )
    if name == "CosineAnnealing":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=5e-5
        )
    raise ValueError(f"Unknown scheduler: {name}")


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_basic_overrides(cfg, args)
    cfg = apply_pipeline_overrides(
        cfg,
        config_path=args.config,
        dataset=args.dataset,
        iteration=args.iteration,
        output_root=args.output_root,
        adjust=args.adjust,
    )

    print_resolved_config(cfg)
    validate_train_config(cfg)
    validate_output_precision(cfg)
    validate_iteration_precision(cfg)

    if args.dry_run:
        print("\nDry run finished.")
        return

    seed = int(cfg["training"].get("seed", 42))
    set_seed(seed)
    device = get_best_device()
    is_cuda = device.type == "cuda"
    print(f"[INFO] device={device}")
    precision = cfg["training"]["precision"]
    amp_dtype = resolve_amp_dtype(precision, device)
    # Do not access the AMP API on the default FP32 path (older HPC environments).
    scaler = torch.amp.GradScaler(device.type) if amp_dtype == torch.float16 else None
    cfg.setdefault("runtime", {}).update({
        "evaluation_precision": "fp32",
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if is_cuda else str(device),
    })
    if device.type == "mps":
        cfg["runtime"]["macos_version"] = platform.mac_ver()[0]

    seq_len = int(cfg["training"]["seq_len"])
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 0))

    train_df = pd.read_parquet(cfg["dataset"]["train_parquet"])
    val_df = pd.read_parquet(cfg["dataset"]["val_parquet"])

    train_ds = KQGMJointDataset(train_df, seq_len=seq_len)
    val_ds = KQGMJointDataset(val_df, seq_len=seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_kqgm_joint, pin_memory=is_cuda, drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_kqgm_joint, pin_memory=is_cuda, drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    model_cfg = cfg["model"]
    model = TransformerEncoderClassifier(
        num_K=int(model_cfg["num_K"]),
        num_Q=int(model_cfg["num_Q"]),
        num_G=int(model_cfg["num_G"]),
        num_M=int(model_cfg["num_M"]),
        num_classes=NUM_STATES,
        d_model=int(model_cfg.get("d_model", 64)),
        nhead=int(model_cfg.get("nhead", 4)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 128)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        max_seq_len=int(model_cfg.get("max_seq_len", seq_len)),
        pooling=model_cfg["pooling"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] params total={total_params:,}, trainable={trainable_params:,}")

    class_weights = None
    if bool(cfg["training"].get("use_class_weights", False)):
        _, _, y_joint = train_ds.get_label_arrays()
        class_weights = compute_class_weights_from_ids(y_joint, NUM_STATES)

    loss_fn = build_loss(
        loss_type=cfg["training"].get("loss_type", "cross_entropy"),
        class_weights=class_weights,
        focal_gamma=cfg["loss"].get("focal_gamma", 2.0),
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    scheduler = build_scheduler(cfg, optimizer)

    out_dir = build_out_dir(cfg)
    save_json(cfg, out_dir / "config.resolved.json")

    best_metric = -1.0
    best_path = out_dir / "best.pt"
    history: list[dict] = []
    epochs = int(cfg["training"]["epochs"])
    grad_clip = float(cfg["training"].get("grad_clip", 0.0))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            loss_fn=loss_fn, device=device, grad_clip=grad_clip,
            amp_dtype=amp_dtype, scaler=scaler,
        )
        metrics = evaluate(model=model, loader=val_loader, loss_fn=loss_fn, device=device)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metrics["joint_macro_f1"])
        elif scheduler is not None:
            scheduler.step(epoch)

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "joint_acc": metrics["joint_acc"],
            "joint_macro_f1": metrics["joint_macro_f1"],
            "current_acc_from_joint": metrics["current_acc_from_joint"],
            "future_acc_from_joint": metrics["future_acc_from_joint"],
            "lr": lr,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
            f"joint_acc={metrics['joint_acc']:.4f} "
            f"joint_macroF1={metrics['joint_macro_f1']:.4f} "
            f"current_acc={metrics['current_acc_from_joint']:.4f} "
            f"future_acc={metrics['future_acc_from_joint']:.4f} "
            f"lr={lr:.2e}"
        )

        if metrics["joint_macro_f1"] > best_metric:
            best_metric = metrics["joint_macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "grad_scaler": scaler.state_dict() if scaler is not None else None,
                    "val_metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
                    "config": cfg,
                },
                best_path,
            )
            print(f"  -> saved best to {best_path} (joint_macro_f1={best_metric:.4f})")

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    print(f"Done. best_joint_macro_f1={best_metric:.4f}; checkpoint={best_path}")


if __name__ == "__main__":
    main()
