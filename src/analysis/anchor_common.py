"""
anchor_common.py
================
Prediction and realized-target readers used by run_dump_points.py for the
paper's event study and causal online Viterbi decoder. The historical module
name is retained; this release has no external-anchor IC/lookup workflow.
Keep the symbol/date alignment, marginal predictions and target definitions
shared here rather than duplicating them in report code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 9-state definition (kept in sync with src/states/current_future_9states.py)
# ---------------------------------------------------------------------------

STATE_NAMES = [
    "BEAR_TO_BEAR",
    "BEAR_TO_NEUTRAL",
    "BEAR_TO_BULL",
    "NEUTRAL_TO_BEAR",
    "NEUTRAL_TO_NEUTRAL",
    "NEUTRAL_TO_BULL",
    "BULL_TO_BEAR",
    "BULL_TO_NEUTRAL",
    "BULL_TO_BULL",
]
TREND_NAMES = ["BEAR", "NEUTRAL", "BULL"]

DATE_COL = "date"
SYMBOL_COL = "symbol"
CLOSE_COL = "close"
FUTURE_COL = "future_trend"
JOINT_COL = "joint_state"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def labels_dir(output_root: Path, dataset: str, iteration: int, split: str) -> Path:
    return output_root / dataset / f"iter_{iteration}" / "labels" / split


def logits_path(output_root: Path, dataset: str, iteration: int, split: str, adjust: str) -> Path:
    return output_root / dataset / f"iter_{iteration}" / "logits" / split / f"ALL_tokens_{adjust}.parquet"


# ---------------------------------------------------------------------------
# Loading, predictions, realized targets, stride (shared)
# ---------------------------------------------------------------------------

def load_label_split(
    output_root: Path, dataset: str, iteration: int, split: str, adjust: str
) -> pd.DataFrame:
    """Load all per-symbol label CSVs of one (iteration, split) into one frame."""
    d = labels_dir(output_root, dataset, iteration, split)
    if not d.exists():
        raise FileNotFoundError(f"Label directory not found: {d}")

    files = sorted(d.glob(f"*_{adjust}.csv"))
    if not files:
        raise FileNotFoundError(f"No '*_{adjust}.csv' files under {d}")

    frames: list[pd.DataFrame] = []
    for path in files:
        symbol = path.name.split("_", 1)[0]
        df = pd.read_csv(path)

        required = [DATE_COL, CLOSE_COL, FUTURE_COL, JOINT_COL]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing required columns: {missing}")

        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        if df[DATE_COL].isna().any():
            raise ValueError(f"{path}: unparseable dates present.")
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        if df[DATE_COL].duplicated().any():
            raise ValueError(f"{path}: duplicated dates present.")

        close = pd.to_numeric(df[CLOSE_COL], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(close)):
            raise ValueError(f"{path}: non-finite close values present.")
        if np.any(close <= 0):
            raise ValueError(f"{path}: non-positive close values present.")

        bad_future = set(df[FUTURE_COL].unique()) - set(TREND_NAMES)
        if bad_future:
            raise ValueError(f"{path}: unknown future_trend values: {bad_future}")
        bad_joint = set(df[JOINT_COL].unique()) - set(STATE_NAMES)
        if bad_joint:
            raise ValueError(f"{path}: unknown joint_state values: {bad_joint}")

        out = pd.DataFrame(
            {
                SYMBOL_COL: symbol,
                DATE_COL: df[DATE_COL].to_numpy(),
                CLOSE_COL: close,
                FUTURE_COL: df[FUTURE_COL].to_numpy(),
                JOINT_COL: df[JOINT_COL].to_numpy(),
            }
        )
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def load_predictions(
    output_root: Path, dataset: str, iteration: int, split: str, adjust: str
) -> pd.DataFrame:
    """
    Load iter_k logits for one split and derive:
      - p_joint_{STATE} : softmax over the 9 joint states (T = 1.0, matching
        the temperature used in the DDSL energy term)
      - p_fut_{TREND}   : marginal future-axis probabilities, obtained by
        summing p_joint over the three joint states sharing each future
        value (NOT by taking the future axis of the joint argmax)
      - pred_future     : hard future prediction = argmax of the MARGINAL
        future probabilities (decision-correct hard rule)
      - pred_joint      : argmax over the 9 joint states
      - pred_future_jointargmax : future axis of pred_joint (legacy hard
        rule, kept only to quantify how often the two rules disagree)
    """
    path = logits_path(output_root, dataset, iteration, split, adjust)
    if not path.exists():
        raise FileNotFoundError(f"Logits parquet not found: {path}")

    df = pd.read_parquet(path)

    key_missing = [c for c in (SYMBOL_COL, DATE_COL) if c not in df.columns]
    if key_missing:
        raise ValueError(f"{path}: missing key columns: {key_missing}")

    logit_cols = [f"logit_{s}" for s in STATE_NAMES]
    missing = [c for c in logit_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing logit columns: {missing}")

    df[SYMBOL_COL] = df[SYMBOL_COL].astype(str).str.strip()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    if df[DATE_COL].isna().any():
        raise ValueError(f"{path}: unparseable dates present.")
    if df.duplicated([SYMBOL_COL, DATE_COL]).any():
        raise ValueError(f"{path}: duplicated (symbol, date) keys present.")

    logits = df[logit_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(logits)):
        raise ValueError(f"{path}: non-finite logits present.")

    # Numerically stable softmax at T = 1.0
    z = logits - logits.max(axis=1, keepdims=True)
    expz = np.exp(z)
    p_joint = expz / expz.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(p_joint)):
        raise ValueError(f"{path}: non-finite softmax probabilities.")
    if np.any(np.abs(p_joint.sum(axis=1) - 1.0) > 1e-9):
        raise ValueError(f"{path}: softmax rows do not sum to 1.")

    # Marginalize the future axis over the current axis
    future_axis = np.array([s.split("_TO_")[1] for s in STATE_NAMES], dtype=object)
    p_fut = np.zeros((len(df), len(TREND_NAMES)), dtype=float)
    for j, trend in enumerate(TREND_NAMES):
        member_cols = np.flatnonzero(future_axis == trend)
        if len(member_cols) != 3:
            raise ValueError(
                f"Future axis {trend!r} maps to {len(member_cols)} joint states, expected 3."
            )
        p_fut[:, j] = p_joint[:, member_cols].sum(axis=1)
    if np.any(np.abs(p_fut.sum(axis=1) - 1.0) > 1e-9):
        raise ValueError(f"{path}: marginal future probabilities do not sum to 1.")

    pred_joint = np.array(STATE_NAMES, dtype=object)[np.argmax(p_joint, axis=1)]
    pred_future_jointargmax = np.array(
        [s.split("_TO_")[1] for s in pred_joint], dtype=object
    )
    pred_future = np.array(TREND_NAMES, dtype=object)[np.argmax(p_fut, axis=1)]

    out = pd.DataFrame(
        {
            SYMBOL_COL: df[SYMBOL_COL].to_numpy(),
            DATE_COL: df[DATE_COL].to_numpy(),
            "pred_joint": pred_joint,
            "pred_future": pred_future,
            "pred_future_jointargmax": pred_future_jointargmax,
        }
    )
    for j, trend in enumerate(TREND_NAMES):
        out[f"p_fut_{trend}"] = p_fut[:, j]
    for i, state in enumerate(STATE_NAMES):
        out[f"p_joint_{state}"] = p_joint[:, i]
    return out


# ---------------------------------------------------------------------------
# Realized forward-window targets (label-independent)
# ---------------------------------------------------------------------------

def realized_targets(split_df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, int]:
    """
    Compute y_dC and y_slope for every (symbol, t) with a full forward window.

    Returns (targets_df, n_tail_excluded). targets_df keeps the label columns
    (future_trend, joint_state) of the SAME frame so train-side grouping needs
    no further join.
    """
    if horizon < 2:
        raise ValueError(f"horizon must be >= 2 to fit an OLS slope, got {horizon}")

    idx = np.arange(horizon, dtype=float)
    idx_c = idx - idx.mean()
    idx_var = float(np.sum(idx_c**2))
    if idx_var <= 0:
        raise ValueError("Degenerate index variance in slope computation.")

    frames: list[pd.DataFrame] = []
    n_tail_excluded = 0

    for symbol, g in split_df.groupby(SYMBOL_COL, sort=True):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        n = len(g)
        n_eligible = n - horizon
        if n_eligible <= 0:
            raise ValueError(
                f"{symbol}: only {n} rows, fewer than horizon+1={horizon + 1}; "
                f"cannot form any forward window. Check the split or blacklist."
            )
        n_tail_excluded += n - n_eligible

        close = g[CLOSE_COL].to_numpy(dtype=float)
        log_close = np.log(close)

        # y_dC(t) = log(C[t+h]) - log(C[t]),  t = 0 .. n_eligible-1
        y_dc = log_close[horizon:] - log_close[:-horizon]

        # Forward windows of RAW close: window for t is close[t+1 : t+1+h].
        # Raw close, not EWMA smooth: the smoothed series at t+1..t+span
        # still carries prices at or before t (lookahead into the target).
        win = np.lib.stride_tricks.sliding_window_view(close[1:], horizon)
        win = win[:n_eligible]
        if win.shape != (n_eligible, horizon):
            raise ValueError(
                f"{symbol}: window shape mismatch {win.shape} != {(n_eligible, horizon)}"
            )

        win_mean = win.mean(axis=1)
        if np.any(np.abs(win_mean) <= 0):
            raise ValueError(
                f"{symbol}: zero mean close price in a forward window; "
                f"cannot normalize slope."
            )
        raw_slope = (win - win_mean[:, None]) @ idx_c / idx_var
        y_slope = raw_slope / np.abs(win_mean)
        if not np.all(np.isfinite(y_dc)) or not np.all(np.isfinite(y_slope)):
            raise ValueError(f"{symbol}: non-finite realized targets computed.")

        frames.append(
            pd.DataFrame(
                {
                    SYMBOL_COL: symbol,
                    DATE_COL: g[DATE_COL].to_numpy()[:n_eligible],
                    "y_dC": y_dc,
                    "y_slope": y_slope,
                    FUTURE_COL: g[FUTURE_COL].to_numpy()[:n_eligible],
                    JOINT_COL: g[JOINT_COL].to_numpy()[:n_eligible],
                }
            )
        )

    return pd.concat(frames, ignore_index=True), n_tail_excluded


def apply_stride(targets: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Keep every stride-th eligible point per symbol (phase 0)."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if stride == 1:
        return targets
    out = (
        targets.sort_values([SYMBOL_COL, DATE_COL])
        .groupby(SYMBOL_COL, sort=False, group_keys=False)
        .apply(lambda g: g.iloc[::stride])
        .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError("Stride subsampling produced an empty frame.")
    return out
