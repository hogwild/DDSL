"""
run_dump_points.py
==================
Lightweight per-point dump for the event study and the causal online Viterbi
re-decoder. Writes exactly the columns those tools consume and nothing else:

  symbol, date, y_dC, y_slope, signal_bull_minus_bear,
  pred_future, pred_joint, p_joint_{STATE} x9

This entry point never touches the train split: the event study and the
Viterbi decoder need only the model's predictions and realized targets on
the evaluation window.

Uses load_label_split / load_predictions / realized_targets / apply_stride
from anchor_common for consistent target and prediction computation.

Output (one file per iteration):
  {out_prefix}_points_iter{k}.parquet

Usage:
  python src/analysis/run_dump_points.py \
      --output_root outputs/outputs_seed42 --dataset CSI500 \
      --iterations 0,1,2,3,4,5 --adjust hfq --horizon 20 \
      --split test --stride 1 --eval_start_date 2023-01-01 \
      --out_prefix outputs/outputs_seed42/CSI500/external_anchor/seed42_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from anchor_common import (
    STATE_NAMES, TREND_NAMES, DATE_COL, SYMBOL_COL,
    load_label_split, load_predictions, realized_targets, apply_stride,
)


def dump_iteration(
    *,
    output_root: Path,
    dataset: str,
    iteration: int,
    split: str,
    adjust: str,
    horizon: int,
    stride: int,
    eval_start_date: pd.Timestamp,
    eval_end_date: pd.Timestamp | None,
    out_path: Path,
) -> int:
    print(f"\n[INFO] ===== iter_{iteration} =====")

    eval_df = load_label_split(output_root, dataset, iteration, split, adjust)
    eval_targets, n_tail = realized_targets(eval_df, horizon)

    n_full = len(eval_targets)
    in_window = eval_targets[DATE_COL] >= eval_start_date
    if eval_end_date is not None:
        in_window &= eval_targets[DATE_COL] <= eval_end_date
    eval_targets = eval_targets.loc[in_window].reset_index(drop=True)
    n_outside = n_full - len(eval_targets)
    if eval_targets.empty:
        raise ValueError(
            f"iter_{iteration} [{split}]: no eligible target points inside the "
            f"evaluation window starting {eval_start_date.date()}. "
            f"Check --eval_start_date against the split definition."
        )
    print(f"[INFO] {split}: {n_full:,} eligible in full prefix; "
          f"{n_outside:,} outside window excluded; {len(eval_targets):,} retained "
          f"({n_tail:,} tail excluded; window start {eval_start_date.date()})")

    eval_targets = apply_stride(eval_targets, stride)
    preds = load_predictions(output_root, dataset, iteration, split, adjust)

    merged = eval_targets.merge(
        preds, on=[SYMBOL_COL, DATE_COL], how="left", validate="one_to_one"
    )
    missing = merged["pred_future"].isna()
    if missing.any():
        ex = merged.loc[missing, [SYMBOL_COL, DATE_COL]].head(10)
        raise ValueError(
            f"iter_{iteration} [{split}]: {int(missing.sum())}/{len(merged)} "
            f"eligible points have no matching logits row.\n"
            f"First offending keys:\n{ex.to_string(index=False)}"
        )

    # directional signal P(BULL) - P(BEAR): identical definition to the IC
    # pipeline, standardized across methods.
    signal_bmb = (
        merged["p_fut_BULL"].to_numpy(dtype=float)
        - merged["p_fut_BEAR"].to_numpy(dtype=float)
    )

    points = pd.DataFrame(
        {
            "dataset": dataset,
            "iteration": iteration,
            "split": split,
            "group_by": "future3",
            "horizon": horizon,
            "stride": stride,
            "eval_start_date": str(eval_start_date.date()),
            SYMBOL_COL: merged[SYMBOL_COL].to_numpy(),
            DATE_COL: merged[DATE_COL].to_numpy(),
            "y_dC": merged["y_dC"].to_numpy(dtype=float),
            "y_slope": merged["y_slope"].to_numpy(dtype=float),
            "signal_bull_minus_bear": signal_bmb,
            "pred_future": merged["pred_future"].to_numpy(),
            "pred_joint": merged["pred_joint"].to_numpy(),
        }
    )
    for _state in STATE_NAMES:
        points[f"p_joint_{_state}"] = merged[f"p_joint_{_state}"].to_numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    points.to_parquet(out_path, index=False)
    print(f"[INFO] Saved {len(points):,} points -> {out_path}")
    return len(points)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--iterations", default="0,1,2,3,4,5")
    p.add_argument("--adjust", default="hfq")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--stride", type=int, default=1,
                   help="Event study / Viterbi require stride=1.")
    p.add_argument("--eval_start_date", required=True)
    p.add_argument("--eval_end_date", default=None)
    p.add_argument("--out_prefix", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    iterations = [int(x) for x in args.iterations.split(",")]
    eval_start = pd.Timestamp(args.eval_start_date)
    eval_end = pd.Timestamp(args.eval_end_date) if args.eval_end_date else None
    out_prefix = Path(args.out_prefix)

    if args.stride != 1:
        print(f"[WARN] stride={args.stride}: the event study and Viterbi need "
              f"consecutive days (stride=1). Proceeding as requested.")

    total = 0
    for it in iterations:
        out_path = out_prefix.parent / f"{out_prefix.name}_points_iter{it}.parquet"
        total += dump_iteration(
            output_root=Path(args.output_root),
            dataset=args.dataset,
            iteration=it,
            split=args.split,
            adjust=args.adjust,
            horizon=args.horizon,
            stride=args.stride,
            eval_start_date=eval_start,
            eval_end_date=eval_end,
            out_path=out_path,
        )
    print(f"\n[INFO] Done. {len(iterations)} iterations, {total:,} points total.")


if __name__ == "__main__":
    main()
