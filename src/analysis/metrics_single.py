from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .io import require_columns


# ---------------------------------------------------------------------------
# Feature allowlist for the row-level Fisher Ratio
#   Uses token-data columns produced by the tokenizer compute_displacement() function.
#   dC  = log(close_t / close_{t-1})   daily close log return
#   gap = log(open_t  / close_{t-1})   overnight gap log return
#   dC5 = log(close_t / close_{t-5})   five-day log return
# ---------------------------------------------------------------------------
ROW_FISHER_COLS: list[str] = ["dC", "gap", "dC5"]

# ---------------------------------------------------------------------------
# Feature allowlist for the segment-level Fisher Ratio
#   Uses output columns from compute_segment_level_features().
#   seg_slope: normalized OLS slope of smooth prices; positive for Bull, negative for Bear, near zero for Neutral.
#   dC_sum: sum of daily close log returns, approximating the total segment return.
#   dC_std: standard deviation of daily close log returns (volatility); excluded because results were unstable.
#   Exclude gap_* / dC5_*: gap reflects opening-mechanism noise and weakly tracks segment trends;
#   overlapping dC5 rolling windows bias sum/std statistics and are redundant with dC_sum.
# ---------------------------------------------------------------------------
SEG_FISHER_COLS: list[str] = ["seg_slope", "dC_sum"] #, "dC_std"]


# ---------------------------------------------------------------------------
# Row-level feature validation
# ---------------------------------------------------------------------------

def check_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the columns from ROW_FISHER_COLS that are present in df.
    Raise if none of the required columns are available.
    """
    present = [c for c in ROW_FISHER_COLS if c in df.columns]
    not_found = [c for c in ROW_FISHER_COLS if c not in df.columns]

    if not present:
        raise ValueError(
            f"check_feature_cols: none of ROW_FISHER_COLS={ROW_FISHER_COLS} found in data. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"[check_feature_cols] using: {present}")
    if not_found:
        print(f"[check_feature_cols] not in data (skipped): {not_found}")

    return present


# ---------------------------------------------------------------------------
# Fisher Ratio
# ---------------------------------------------------------------------------

def fisher_ratio(
    df: pd.DataFrame,
    *,
    label_col: str,
    feature_cols: list[str],
) -> dict[str, Any]:
    """
    Compute the Fisher Ratio for either row-level or segment-level observations.

    Fisher Ratio = between-class variance / within-class variance, with equal label weights.
    feature_cols must be nonempty, and label_col must exist.
    """
    require_columns(df, [label_col] + feature_cols, "labels")

    labels = df[label_col].to_numpy()
    unique_labels = pd.unique(labels[~pd.isnull(labels)])

    if len(unique_labels) <= 1:
        raise ValueError(
            f"fisher_ratio requires at least 2 label classes, "
            f"got: {unique_labels.tolist()}"
        )

    per_feature: dict[str, float] = {}
    fisher_values: list[float] = []

    for feat in feature_cols:
        x = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & ~pd.isnull(labels)

        if valid.sum() == 0:
            raise ValueError(
                f"fisher_ratio: feature '{feat}' has no finite values after removing NaN labels."
            )

        xv = x[valid]
        lv = labels[valid]

        class_means: list[float] = []
        class_vars: list[float] = []

        class_counts = pd.Series(lv).value_counts()
        n_singleton = int((class_counts < 2).sum())
        if n_singleton > 0:
            raise ValueError(
                f"Fisher Ratio: {n_singleton} class(es) with < 2 samples in this "
                f"window: {class_counts[class_counts < 2].to_dict()}. "
                f"Per-class variance is degenerate; refusing to compute a "
                f"misleading FR. Widen the window or report this state as absent."
            )

        for lab in pd.unique(lv):
            xs = xv[lv == lab]
            class_means.append(float(np.mean(xs)))
            class_vars.append(float(np.var(xs, ddof=0)))

        centroid_mean = float(np.mean(class_means))
        between = float(np.mean([(mu - centroid_mean) ** 2 for mu in class_means]))
        within = float(np.mean(class_vars))

        value = float(between / (within + 1e-12))
        per_feature[feat] = value
        fisher_values.append(value)

    return {
        "feature_cols": feature_cols,
        "macro_fisher": float(np.mean(fisher_values)),
        "per_feature": per_feature,
    }


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def extract_segments(
    df: pd.DataFrame,
    *,
    symbol_col: str,
    date_col: str,
    state_col: str,
) -> pd.DataFrame:
    """
    Merge consecutive rows with the same state into segments and return their summary DataFrame.
    """
    require_columns(df, [symbol_col, date_col, state_col], "labels")

    df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

    segments: list[dict[str, Any]] = []

    for symbol, g in df.groupby(symbol_col, sort=False):
        g = g.reset_index(drop=True)
        states = g[state_col].to_numpy()

        start = 0
        for i in range(1, len(g) + 1):
            if i == len(g) or states[i] != states[start]:
                seg: dict[str, Any] = {
                    "symbol": str(symbol),
                    "state": states[start],
                    "start_idx": int(start),
                    "end_idx": int(i - 1),
                    "length": int(i - start),
                    "start_date": g.iloc[start][date_col],
                    "end_date": g.iloc[i - 1][date_col],
                }
                segments.append(seg)
                start = i

    return pd.DataFrame(segments)


def compute_segment_stats(segments: pd.DataFrame) -> dict[str, Any]:
    """
    Compute segment-length statistics from the output of extract_segments.
    segments must be nonempty; an empty input indicates a data issue.
    """
    if segments.empty:
        raise ValueError("compute_segment_stats: segments DataFrame is empty.")

    out: dict[str, Any] = {
        "num_segments": int(len(segments)),
        "mean_length": float(segments["length"].mean()),
        "median_length": float(segments["length"].median()),
        "min_length": int(segments["length"].min()),
        "max_length": int(segments["length"].max()),
        "length_by_state": {},
    }

    by_state = (
        segments.groupby("state")["length"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    for _, row in by_state.iterrows():
        out["length_by_state"][str(row["state"])] = {
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "median": float(row["median"]),
            "min": int(row["min"]),
            "max": int(row["max"]),
        }

    return out


# ---------------------------------------------------------------------------
# Segment-level features
# ---------------------------------------------------------------------------

def compute_segment_level_features(
    df: pd.DataFrame,
    *,
    symbol_col: str,
    date_col: str,
    state_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Compute per-segment feature statistics for the segment-level Fisher Ratio.

    For each log-return column in feature_cols, compute:
      - {feat}_sum : cumulative segment log return (approximately the total segment return)
      - {feat}_std : standard deviation of daily returns within the segment

    Also compute the following whenever the smooth column exists:
      - seg_slope  : OLS slope of smooth prices, normalized by the segment mean price
                     for comparability across stocks: positive for Bull, negative for Bear, near zero for Neutral
    """
    require_columns(df, [symbol_col, date_col, state_col] + feature_cols, "labels")

    df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

    has_smooth = "smooth" in df.columns
    rows: list[dict[str, Any]] = []

    for symbol, g in df.groupby(symbol_col, sort=False):
        g = g.reset_index(drop=True)
        states = g[state_col].to_numpy()

        start = 0
        for i in range(1, len(g) + 1):
            if i == len(g) or states[i] != states[start]:
                seg = g.iloc[start:i]

                row: dict[str, Any] = {
                    "symbol": str(symbol),
                    "state": states[start],
                    "length": int(i - start),
                    "start_date": seg.iloc[0][date_col],
                    "end_date": seg.iloc[-1][date_col],
                }

                # seg_slope: OLS on smooth price, normalized by mean price
                if has_smooth:
                    smooth_vals = pd.to_numeric(
                        seg["smooth"], errors="coerce"
                    ).to_numpy(dtype=float)
                    valid_mask = np.isfinite(smooth_vals)
                    n_valid = int(valid_mask.sum())

                    if n_valid >= 2:
                        y = smooth_vals[valid_mask]
                        x = np.where(valid_mask)[0].astype(float)
                        x_mean, y_mean = x.mean(), y.mean()
                        denom = float(np.sum((x - x_mean) ** 2))
                        if denom > 1e-8:
                            raw_slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
                            row["seg_slope"] = raw_slope / (abs(y_mean) + 1e-8)
                        else:
                            row["seg_slope"] = 0.0
                    else:
                        row["seg_slope"] = math.nan

                # Log-return feature statistics
                for feat in feature_cols:
                    values = pd.to_numeric(seg[feat], errors="coerce").to_numpy(dtype=float)
                    finite = values[np.isfinite(values)]

                    row[f"{feat}_sum"] = float(finite.sum()) if len(finite) > 0 else math.nan
                    row[f"{feat}_std"] = (
                        float(np.std(finite, ddof=0)) if len(finite) >= 2 else 0.0
                    )

                rows.append(row)
                start = i

    return pd.DataFrame(rows)


def seg_fisher_feature_cols(seg_feat_df: pd.DataFrame) -> list[str]:
    """
    Return the columns used in the segment-level Fisher Ratio.

    Use the fixed SEG_FISHER_COLS allowlist; raise if any column is absent from seg_feat_df.
    """
    missing = [c for c in SEG_FISHER_COLS if c not in seg_feat_df.columns]
    if missing:
        raise ValueError(
            f"seg_fisher_feature_cols: whitelist columns missing from seg_feat_df: {missing}. "
            f"Available: {list(seg_feat_df.columns)}"
        )

    print(f"[seg_fisher_feature_cols] using: {SEG_FISHER_COLS}")
    return SEG_FISHER_COLS


# ---------------------------------------------------------------------------
# Boundary consistency
# ---------------------------------------------------------------------------

def _boundary_consistency_for_symbol(
    g: pd.DataFrame,
    *,
    state_col: str,
    radius: int,
) -> list[dict[str, Any]]:
    states = g[state_col].to_numpy()
    rows: list[dict[str, Any]] = []

    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            continue

        prev_state = states[i - 1]
        cur_state = states[i]

        left = states[max(0, i - radius):i]
        right = states[i:min(len(states), i + radius)]

        left_consistency = float(np.mean(left == prev_state)) if len(left) > 0 else math.nan
        right_consistency = float(np.mean(right == cur_state)) if len(right) > 0 else math.nan

        if math.isnan(left_consistency) and math.isnan(right_consistency):
            bc = math.nan
        elif math.isnan(left_consistency):
            bc = right_consistency
        elif math.isnan(right_consistency):
            bc = left_consistency
        else:
            bc = 0.5 * (left_consistency + right_consistency)

        rows.append({
            "boundary_index": int(i),
            "prev_state": prev_state,
            "new_state": cur_state,
            "left_consistency": left_consistency,
            "right_consistency": right_consistency,
            "boundary_consistency": bc,
        })

    return rows


def compute_boundary_consistency(
    df: pd.DataFrame,
    *,
    symbol_col: str,
    date_col: str,
    state_col: str,
    radius: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compute the fraction of matching-state rows within radius rows on each side of each transition boundary.
    Return (detail_df, summary_dict).
    An empty detail_df means there are no state transitions; this is valid and does not raise.
    """
    require_columns(df, [symbol_col, date_col, state_col], "labels")

    df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []

    for symbol, g in df.groupby(symbol_col, sort=False):
        g = g.reset_index(drop=True)
        local_rows = _boundary_consistency_for_symbol(g, state_col=state_col, radius=radius)

        for r in local_rows:
            idx = r["boundary_index"]
            r["symbol"] = str(symbol)
            r["boundary_date"] = g.iloc[idx][date_col]
            rows.append(r)

    bc_df = pd.DataFrame(rows)

    if bc_df.empty:
        return bc_df, {
            "num_boundaries": 0,
            "boundary_consistency_mean": None,
            "boundary_consistency_std": None,
            "left_consistency_mean": None,
            "right_consistency_mean": None,
            "by_transition": {},
        }

    summary: dict[str, Any] = {
        "num_boundaries": int(len(bc_df)),
        "boundary_consistency_mean": float(bc_df["boundary_consistency"].mean()),
        "boundary_consistency_std": float(bc_df["boundary_consistency"].std(ddof=0)),
        "left_consistency_mean": float(bc_df["left_consistency"].mean()),
        "right_consistency_mean": float(bc_df["right_consistency"].mean()),
        "by_transition": {},
    }

    by_trans = (
        bc_df.groupby(["prev_state", "new_state"])["boundary_consistency"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    for _, row in by_trans.iterrows():
        key = f"{row['prev_state']}->{row['new_state']}"
        summary["by_transition"][key] = {
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]) if not pd.isna(row["std"]) else None,
        }

    return bc_df, summary
