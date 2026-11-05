from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .io import distribution_dict, require_columns


# ---------------------------------------------------------------------------
# Align label DataFrames from two iterations.
# ---------------------------------------------------------------------------

def align_base_new(
    base_df: pd.DataFrame,
    new_df: pd.DataFrame,
    *,
    symbol_col: str,
    date_col: str,
    cols: list[str],
) -> pd.DataFrame:
    """
    Inner-join two label DataFrames on (symbol, date) to align labels for matching stock dates.
    Output columns are named base_{col} and new_{col}.
    Raise if the join is empty because there are no shared (symbol, date) pairs.
    """
    key_cols = [symbol_col, date_col]
    require_columns(base_df, key_cols + cols, "base labels")
    require_columns(new_df, key_cols + cols, "new labels")

    base = base_df[key_cols + cols].copy()
    new = new_df[key_cols + cols].copy()

    base[symbol_col] = base[symbol_col].astype(str)
    new[symbol_col] = new[symbol_col].astype(str)

    base[date_col] = pd.to_datetime(base[date_col])
    new[date_col] = pd.to_datetime(new[date_col])

    base = base.rename(columns={c: f"base_{c}" for c in cols})
    new = new.rename(columns={c: f"new_{c}" for c in cols})

    merged = base.merge(new, on=key_cols, how="inner", validate="one_to_one")
    merged = merged.sort_values([symbol_col, date_col]).reset_index(drop=True)

    if merged.empty:
        raise ValueError(
            f"align_base_new: inner join on {key_cols} returned 0 rows. "
            "base and new DataFrames share no common (symbol, date) pairs."
        )

    return merged


# ---------------------------------------------------------------------------
# Distribution distances
# ---------------------------------------------------------------------------

def _align_distributions(
    p: dict[str, float],
    q: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keys = sorted(set(p.keys()) | set(q.keys()))
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    pv = pv / max(pv.sum(), 1e-12)
    qv = qv / max(qv.sum(), 1e-12)
    return pv, qv, keys


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return float(0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m))


def _total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def distribution_similarity(
    base: pd.Series,
    new: pd.Series,
) -> dict[str, Any]:
    """Compute distribution differences between two label sequences using JSD and TV distance."""
    p = distribution_dict(base)
    q = distribution_dict(new)
    pv, qv, keys = _align_distributions(p, q)
    return {
        "labels": keys,
        "base_distribution": p,
        "new_distribution": q,
        "jsd": _js_divergence(pv, qv),
        "tv": _total_variation(pv, qv),
    }


# ---------------------------------------------------------------------------
# Cohen Kappa
# ---------------------------------------------------------------------------

def cohen_kappa(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute Cohen's Kappa.
    The caller must ensure that inputs are nonempty.
    Raise if pe == 1, where identical predictions produce a zero denominator.
    """
    a = y_true.astype(str).to_numpy()
    b = y_pred.astype(str).to_numpy()

    if len(a) == 0:
        raise ValueError("cohen_kappa: input series are empty.")

    labels = sorted(set(a) | set(b))
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=float)

    for x, y in zip(a, b):
        cm[idx[x], idx[y]] += 1.0

    n = cm.sum()
    po = np.trace(cm) / n
    row_sums = cm.sum(axis=1) / n
    col_sums = cm.sum(axis=0) / n
    pe = float(np.dot(row_sums, col_sums))

    if abs(1.0 - pe) < 1e-12:
        raise ValueError(
            f"cohen_kappa: pe={pe:.6f} makes denominator (1 - pe) ≈ 0. "
            "This means the expected agreement is 1.0, kappa is undefined."
        )

    return float((po - pe) / (1.0 - pe))


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def compute_agreement(
    aligned: pd.DataFrame,
    *,
    label_cols: list[str],
) -> dict[str, Any]:
    """
    Compute agreement between base and new values for each label column.
    aligned is the output of align_base_new and must contain base_{col} and new_{col}.
    """
    out: dict[str, Any] = {}

    for col in label_cols:
        base_col = f"base_{col}"
        new_col = f"new_{col}"
        require_columns(aligned, [base_col, new_col], f"aligned ({col})")

        mask = aligned[base_col].notna() & aligned[new_col].notna()
        n = int(mask.sum())

        if n == 0:
            raise ValueError(
                f"compute_agreement: column '{col}' has 0 valid (non-NaN) rows "
                "after aligning base and new."
            )

        base_vals = aligned.loc[mask, base_col].astype(str).to_numpy()
        new_vals = aligned.loc[mask, new_col].astype(str).to_numpy()

        num_same = int((base_vals == new_vals).sum())
        num_changed = n - num_same

        out[col] = {
            "n": n,
            "agreement": float(num_same / n),
            "kappa": cohen_kappa(
                aligned.loc[mask, base_col],
                aligned.loc[mask, new_col],
            ),
            "num_changed": num_changed,
            "change_rate": float(num_changed / n),
        }

    return out


# ---------------------------------------------------------------------------
# Transition series
# ---------------------------------------------------------------------------

def transition_series(
    df: pd.DataFrame,
    *,
    symbol_col: str,
    date_col: str,
    state_col: str,
) -> pd.Series:
    """
    Extract daily state transitions for each stock in "state_a->state_b" format.
    """
    require_columns(df, [symbol_col, date_col, state_col], "labels")

    df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

    transitions: list[str] = []

    for _, g in df.groupby(symbol_col, sort=False):
        states = g[state_col].astype(str).to_numpy()
        if len(states) < 2:
            continue
        transitions.extend([f"{a}->{b}" for a, b in zip(states[:-1], states[1:])])

    return pd.Series(transitions, dtype="object")


# ---------------------------------------------------------------------------
# Boundary positions
# ---------------------------------------------------------------------------

def boundary_positions(
    df: pd.DataFrame,
    *,
    symbol_col: str,
    date_col: str,
    state_col: str,
) -> pd.DataFrame:
    """
    Find all state-transition boundaries and return (symbol, date, prev_state, new_state) for each.
    """
    require_columns(df, [symbol_col, date_col, state_col], "labels")

    df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []

    for symbol, g in df.groupby(symbol_col, sort=False):
        g = g.reset_index(drop=True)
        states = g[state_col].to_numpy()

        for i in range(1, len(g)):
            if states[i] != states[i - 1]:
                rows.append({
                    "symbol": str(symbol),
                    "boundary_index": int(i),
                    "boundary_date": g.iloc[i][date_col],
                    "prev_state": states[i - 1],
                    "new_state": states[i],
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Boundary overlap
# ---------------------------------------------------------------------------

def compute_boundary_overlap(
    base_df: pd.DataFrame,
    new_df: pd.DataFrame,
    *,
    symbol_col: str,
    date_col: str,
    state_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compute Jaccard overlap between boundary sets from two iterations.
    Use (symbol, boundary_date) pairs for set operations.
    """
    base_b = boundary_positions(
        base_df,
        symbol_col=symbol_col,
        date_col=date_col,
        state_col=state_col,
    )
    new_b = boundary_positions(
        new_df,
        symbol_col=symbol_col,
        date_col=date_col,
        state_col=state_col,
    )

    key_cols = ["symbol", "boundary_date"]

    base_keys = base_b[key_cols].drop_duplicates()
    new_keys = new_b[key_cols].drop_duplicates()

    merged = base_keys.merge(
        new_keys,
        on=key_cols,
        how="outer",
        indicator=True,
    )

    intersection = int((merged["_merge"] == "both").sum())
    union = int(len(merged))

    if union == 0:
        raise ValueError(
            "compute_boundary_overlap: both base and new have 0 boundaries. "
            "Cannot compute Jaccard overlap."
        )

    jaccard = float(intersection / union)

    summary = {
        "base_num_boundaries": int(len(base_keys)),
        "new_num_boundaries": int(len(new_keys)),
        "intersection": intersection,
        "union": union,
        "jaccard": jaccard,
    }

    return merged, summary


# ---------------------------------------------------------------------------
# Confusion table
# ---------------------------------------------------------------------------

def confusion_table(
    aligned: pd.DataFrame,
    *,
    col: str,
) -> pd.DataFrame:
    """
    Compute the confusion matrix between base and new labels.
    aligned is the output of align_base_new.
    """
    base_col = f"base_{col}"
    new_col = f"new_{col}"
    require_columns(aligned, [base_col, new_col], f"confusion_table ({col})")

    tab = pd.crosstab(
        aligned[base_col].astype(str),
        aligned[new_col].astype(str),
        rownames=[base_col],
        colnames=[new_col],
        dropna=False,
    )

    return tab.reset_index()