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

# Allow running from repo root without installing package.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.utils import get_best_device
from data.kqgm_dataset import KQGMJointDataset, collate_kqgm_joint
from models.transformer_encoder import TransformerEncoderClassifier
from states import STATE_NAMES, NUM_STATES
from training.evaluator import predict_logits
from common.symbols import normalize_symbol_series, validate_symbol_series


# ---------------------------------------------------------------------
# Config / model loading
# ---------------------------------------------------------------------
def load_config(model_dir: Path) -> dict[str, Any]:
    cfg_path = model_dir / "config.resolved.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_from_config(cfg: dict[str, Any]) -> TransformerEncoderClassifier:
    model_cfg = cfg["model"]

    return TransformerEncoderClassifier(
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
        max_seq_len=int(model_cfg.get("max_seq_len", cfg["training"]["seq_len"])),
        pooling=str(model_cfg.get("pooling", "last")),
    )


def load_model(
    model_dir: Path,
    checkpoint_name: str,
    device: torch.device,
) -> tuple[TransformerEncoderClassifier, dict[str, Any]]:
    cfg = load_config(model_dir)
    ckpt_path = model_dir / checkpoint_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = build_model_from_config(cfg).to(device)

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"[INFO] Loaded config: {model_dir / 'config.resolved.json'}")

    return model, cfg

def infer_market_from_config(cfg: dict[str, Any]) -> str:
    dataset_cfg = cfg.get("dataset", {})

    if isinstance(dataset_cfg, dict):
        return str(dataset_cfg.get("market", cfg.get("market", "cn")))

    return str(cfg.get("market", "cn"))

def check_logits_vs_codebook(
    out_df: pd.DataFrame,
    *,
    output_root: Path,
    dataset: str | None,
    split: str,
    out_path: Path,
) -> None:
    """
    Check that logits cover every stock in symbols_with_codebook.
    Raise if any codebook stock has no logits.
    Run this check only in pipeline mode, where dataset is not None.
    """
    if dataset is None:
        return  # Manual mode: the codebook directory cannot be located.

    info_path = output_root / dataset / "codebook" / "symbol_codebook_info.json"
    if not info_path.exists():
        print(f"[INFO] {info_path} not found, skipping coverage check.")
        return

    with open(info_path, "r", encoding="utf-8") as f:
        symbol_info = json.load(f)

    symbols_with_codebook = set(symbol_info.get("symbols_with_codebook", []))
    symbols_in_logits = set(out_df["symbol"].unique())

    missing = symbols_with_codebook - symbols_in_logits
    if missing:
        raise ValueError(
            f"[{split}] export_logits coverage check FAILED.\n"
            f"{len(missing)} symbol(s) are in symbols_with_codebook "
            f"but produced no logits (likely fewer rows than seq_len):\n"
            f"  {sorted(missing)}\n"
            f"Resolution: add these symbols to the blacklist and re-run "
            f"from codebook fitting.\n"
            f"Logits saved to: {out_path}"
        )

    print(
        f"[INFO] Coverage check passed: all {len(symbols_with_codebook)} "
        f"codebook symbols have logits in [{split}] split."
    )

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export model logits for DDSL relabeling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Manual mode arguments.
    # Use these only when you want to export one file manually.
    ap.add_argument("--model_dir", type=str, default=None)
    ap.add_argument("--data_parquet", type=str, default=None)
    ap.add_argument("--out_path", type=str, default=None)

    # Pipeline mode arguments.
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name, e.g., SSE50.",
    )
    ap.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="DDSL iteration id.",
    )
    ap.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root output directory.",
    )
    ap.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Split to export.",
    )
    ap.add_argument(
        "--all_splits",
        action="store_true",
        help="Export logits for train, val, and test splits.",
    )
    ap.add_argument(
        "--adjust",
        type=str,
        default="hfq",
        help="Token parquet suffix, e.g., ALL_tokens_hfq.parquet.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed. Used to locate iter_k/model/seed_{seed}.",
    )

    # Inference options.
    ap.add_argument("--checkpoint_name", type=str, default="best.pt")
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--date_col", type=str, default="date")
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
    split: str,
    seed: int,
    adjust: str,
) -> tuple[Path, Path, Path]:
    """
    Resolve paths for the iterative DDSL pipeline.

    Model:
        outputs/{dataset}/iter_{iteration}/model/seed_{seed}

    Input tokens:
        outputs/{dataset}/shared/logits_dataset/{split}/ALL_tokens_{adjust}.parquet

    Output logits:
        outputs/{dataset}/iter_{iteration}/logits/{split}/ALL_tokens_{adjust}.parquet
    """
    if iteration < 0:
        raise ValueError("--iteration must be >= 0")

    root = Path(output_root)
    iter_dir = root / dataset / f"iter_{iteration}"

    model_dir = iter_dir / "model" / f"seed_{seed}"

    data_parquet = (
        root
        / dataset
        / "shared"
        / "logits_dataset"
        / split
        / f"ALL_tokens_{adjust}.parquet"
    )

    out_path = (
        iter_dir
        / "logits"
        / split
        / f"ALL_tokens_{adjust}.parquet"
    )

    return model_dir, data_parquet, out_path


def resolve_manual_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    missing = []
    if args.model_dir is None:
        missing.append("--model_dir")
    if args.data_parquet is None:
        missing.append("--data_parquet")
    if args.out_path is None:
        missing.append("--out_path")

    if missing:
        raise ValueError(
            "Manual mode requires: "
            + ", ".join(missing)
            + ". Or use pipeline mode with --dataset --iteration --split/--all_splits."
        )

    return Path(args.model_dir), Path(args.data_parquet), Path(args.out_path)


def resolve_jobs(args: argparse.Namespace) -> list[tuple[str, Path, Path, Path]]:
    """
    Returns:
        List of (split_name, model_dir, data_parquet, out_path)
    """
    using_pipeline = args.dataset is not None or args.iteration is not None

    if using_pipeline:
        if args.dataset is None:
            raise ValueError("Pipeline mode requires --dataset.")
        if args.iteration is None:
            raise ValueError("Pipeline mode requires --iteration.")

        if args.all_splits:
            splits = ["train", "val", "test"]
        elif args.split is not None:
            splits = [args.split]
        else:
            raise ValueError("Pipeline mode requires --split or --all_splits.")

        jobs: list[tuple[str, Path, Path, Path]] = []

        for split in splits:
            model_dir, data_parquet, out_path = resolve_pipeline_paths(
                dataset=args.dataset,
                iteration=args.iteration,
                output_root=args.output_root,
                split=split,
                seed=args.seed,
                adjust=args.adjust,
            )
            jobs.append((split, model_dir, data_parquet, out_path))

        return jobs

    model_dir, data_parquet, out_path = resolve_manual_paths(args)
    split_name = args.split or "manual"
    return [(split_name, model_dir, data_parquet, out_path)]


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def validate_paths(
    *,
    split: str,
    model_dir: Path,
    data_parquet: Path,
    checkpoint_name: str,
) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"[{split}] model_dir not found: {model_dir}")

    ckpt = model_dir / checkpoint_name
    if not ckpt.exists():
        raise FileNotFoundError(f"[{split}] checkpoint not found: {ckpt}")

    cfg_path = model_dir / "config.resolved.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"[{split}] config.resolved.json not found: {cfg_path}")

    if not data_parquet.exists():
        raise FileNotFoundError(f"[{split}] data_parquet not found: {data_parquet}")


def check_logits_dataset_columns(df: pd.DataFrame, data_parquet: Path) -> None:
    required = ["symbol", "date", "K", "Q", "G", "M"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{data_parquet}: missing required columns for logits export: {missing}"
        )


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def export_one_split(
    *,
    split: str,
    model_dir: Path,
    data_parquet: Path,
    out_path: Path,
    checkpoint_name: str,
    batch_size: int,
    num_workers: int,
    date_col: str,
    device: torch.device,
    output_root: Path,   
    dataset: str | None,
) -> None:
    print("\n" + "=" * 72)
    print(f"SPLIT        : {split}")
    print(f"model_dir    : {model_dir}")
    print(f"data_parquet : {data_parquet}")
    print(f"out_path     : {out_path}")
    print("=" * 72)

    validate_paths(
        split=split,
        model_dir=model_dir,
        data_parquet=data_parquet,
        checkpoint_name=checkpoint_name,
    )

    model, cfg = load_model(
        model_dir=model_dir,
        checkpoint_name=checkpoint_name,
        device=device,
    )

    market = infer_market_from_config(cfg)
    print(f"[INFO] Market: {market}")

    df = pd.read_parquet(data_parquet)
    check_logits_dataset_columns(df, data_parquet)

    print(f"[INFO] Input parquet: {data_parquet}")
    print(f"[INFO] Rows before windowing: {len(df):,}")

    seq_len = int(cfg["training"]["seq_len"])

    ds = KQGMJointDataset(
        df,
        seq_len=seq_len,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_kqgm_joint,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    print(f"[INFO] Seq len: {seq_len}")
    print(f"[INFO] Samples after windowing: {len(ds):,}")

    logits, symbols, dates = predict_logits(
        model=model,
        loader=loader,
        device=device,
    )

    if logits.shape[1] != len(STATE_NAMES):
        raise ValueError(
            f"Logit dimension mismatch: logits={logits.shape[1]}, "
            f"states={len(STATE_NAMES)}"
        )

    out_df = pd.DataFrame(
        {
            "symbol": symbols,
            date_col: pd.to_datetime(dates).astype(str),
        }
    )

    out_df["symbol"] = normalize_symbol_series(
        out_df["symbol"],
        market=market,
    )
    
    validate_symbol_series(
        out_df["symbol"],
        market=market,
        name=f"{out_path.name}:symbol",
    )

    for i, state_name in enumerate(STATE_NAMES):
        out_df[f"logit_{state_name}"] = logits[:, i]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Before saving out_df
    check_logits_vs_codebook(
        out_df,
        output_root=output_root,
        dataset=dataset,
        split=split,
        out_path=out_path,
    )
    out_df.to_parquet(out_path, index=False)

    print(f"[INFO] Saved logits -> {out_path}")
    print(f"[INFO] Rows: {len(out_df):,}")
    print("[INFO] Logit column order:")
    for i, state_name in enumerate(STATE_NAMES):
        print(f"  {i}: logit_{state_name}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    jobs = resolve_jobs(args)

    print("\nResolved export-logits jobs")
    print("-" * 72)
    for split, model_dir, data_parquet, out_path in jobs:
        print(f"[{split}]")
        print(f"  model_dir    : {model_dir}")
        print(f"  data_parquet : {data_parquet}")
        print(f"  out_path     : {out_path}")
    print("-" * 72)

    if args.dry_run:
        for split, model_dir, data_parquet, _ in jobs:
            validate_paths(
                split=split,
                model_dir=model_dir,
                data_parquet=data_parquet,
                checkpoint_name=args.checkpoint_name,
            )
        print("\nDry run finished.")
        return

    device = get_best_device()
    print(f"[INFO] Device = {device}")

    for split, model_dir, data_parquet, out_path in jobs:
        export_one_split(
            split=split,
            model_dir=model_dir,
            data_parquet=data_parquet,
            out_path=out_path,
            checkpoint_name=args.checkpoint_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            date_col=args.date_col,
            device=device,
            output_root=Path(args.output_root),  # Added
            dataset=args.dataset,                # A string in pipeline mode, or None in manual mode
        )

    print("\nAll logits exported.")


if __name__ == "__main__":
    main()