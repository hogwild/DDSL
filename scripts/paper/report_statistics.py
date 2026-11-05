"""Statistics shared by the final-paper reports; no filesystem or CLI behavior."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Preserve the order used by the saved label-distribution reports.
EXPECTED_STATES = (
    "BEAR_TO_BEAR", "BEAR_TO_BULL", "BEAR_TO_NEUTRAL",
    "BULL_TO_BEAR", "BULL_TO_BULL", "BULL_TO_NEUTRAL",
    "NEUTRAL_TO_BEAR", "NEUTRAL_TO_BULL", "NEUTRAL_TO_NEUTRAL",
)


def shannon_entropy_bits(p):
    """Zero-probability states contribute zero and remain visible in coverage."""
    nz = p[p > 0.0]
    return float(-np.sum(nz * np.log2(nz)))


def mean_std(frame, keys, metrics):
    """Equal weight per seed, sample SD (ddof=1); never pool events."""
    if frame.duplicated([*keys, "seed"]).any():
        raise ValueError(f"Duplicate seed observations for {keys}")
    result = frame.groupby(keys, dropna=False, sort=False)[metrics].agg(["mean", "std"])
    result.columns = [f"{a}_{b}" for a, b in result.columns]
    result["n_seeds"] = frame.groupby(keys, dropna=False, sort=False)["seed"].nunique()
    return result.reset_index()


def rho(a, b):
    """Spearman rank correlation; a constant trajectory has undefined rho."""
    if pd.Series(a).nunique() < 2 or pd.Series(b).nunique() < 2:
        return None
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))
