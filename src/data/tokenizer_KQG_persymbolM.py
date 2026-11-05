"""
tokenizer.py
============
Core K/Q/G/M tokenization utilities.

This module is intentionally label-agnostic: it only converts OHLCV features into
K/Q/G/M/W token columns. Label merging and parquet writing should be handled by
`build_dataset.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import load


@dataclass(frozen=True)
class KQGMTokenizerArtifacts:
    """Loaded tokenizer artifacts and hyperparameters."""

    geom_kmeans: object
    q_mode: str
    Qn: int
    Gn: int
    Mn: int
    edges_q: Optional[np.ndarray]
    edges_g: Optional[np.ndarray]
    edges_m_by_symbol: Dict[str, np.ndarray]
    m_col: str = "turnover"
    vol_window: int = 60
    q_min_periods: int = 30


def load_tokenizer_artifacts(tokenizer_dir: str | Path) -> KQGMTokenizerArtifacts:
    """Load KMeans geometry model and bin edges from a tokenizer directory."""
    tokenizer_dir = Path(tokenizer_dir)
    km = load(tokenizer_dir / "geom_kmeans.joblib")
    bins = load(tokenizer_dir / "bins.joblib")

    return KQGMTokenizerArtifacts(
        geom_kmeans=km,
        q_mode=bins.get("q_mode", "dc"),
        Qn=int(bins["bins_q"]),
        Gn=int(bins["bins_g"]),
        Mn=int(bins["bins_m"]),
        edges_q=bins["edges_q"],
        edges_g=bins["edges_g"],
        edges_m_by_symbol=bins["edges_m_by_symbol"],
        m_col=bins.get("m_col", "turnover"),
        vol_window=int(bins.get("vol_window", 60)),
        q_min_periods=int(bins.get("q_min_periods", 30)),
    )


def compute_geometry(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Compute candlestick geometry features b, clv, upper wick, lower wick."""
    o = pd.to_numeric(df["open"], errors="coerce").astype(float).to_numpy()
    h = pd.to_numeric(df["high"], errors="coerce").astype(float).to_numpy()
    l = pd.to_numeric(df["low"], errors="coerce").astype(float).to_numpy()
    c = pd.to_numeric(df["close"], errors="coerce").astype(float).to_numpy()

    R = (h - l) + eps
    b = (c - o) / R
    clv = (c - l) / R
    uw = (h - np.maximum(o, c)) / R
    lw = (np.minimum(o, c) - l) / R

    return pd.DataFrame({"b": b, "clv": clv, "uw": uw, "lw": lw})


def compute_displacement(
    df: pd.DataFrame,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Log-return features for tokenization:
      dC  = log(close_t / close_{t-1})
      gap = log(open_t  / close_{t-1})
    First row per symbol is NaN (no previous close).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    c = pd.to_numeric(df["close"], errors="coerce").astype(float)
    o = pd.to_numeric(df["open"], errors="coerce").astype(float)
    g = df["symbol"]

    c_prev = c.groupby(g).shift(1)
 
    dC1 = np.log((c + eps) / (c_prev + eps))
    gap = np.log((o + eps) / (c_prev + eps))

    return pd.DataFrame({"dC": dC1, "gap": gap})


def compute_volnorm_dc(
    df: pd.DataFrame,
    vol_window: int,
    min_periods: int,
    eps: float = 1e-12,
) -> pd.Series:
    """Compute volatility-normalized one-day close return."""
    dC = compute_displacement(df, eps=eps)["dC"]
    sigma = dC.rolling(vol_window, min_periods=min_periods).std().shift(1)
    return dC / (sigma + eps)


def bin_to_int_per_symbol(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map continuous values to integer bin ids using precomputed bin edges."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape[0], -1, dtype=np.int32)

    if edges is None or not np.isfinite(edges).all():
        return out

    ok = np.isfinite(x)
    out[ok] = np.searchsorted(edges, x[ok], side="right").astype(np.int32)
    return out


def composite_token_id(
    K: np.ndarray,
    Q: np.ndarray,
    G: np.ndarray,
    M: np.ndarray,
    Qn: int,
    Gn: int,
    Mn: int,
) -> np.ndarray:
    """Build composite token id: W = (((K * Qn) + Q) * Gn + G) * Mn + M."""
    return (((K * Qn) + Q) * Gn + G) * Mn + M


class KQGMTokenizer:
    """Label-agnostic K/Q/G/M tokenizer for one symbol dataframe."""

    def __init__(self, artifacts: KQGMTokenizerArtifacts):
        self.artifacts = artifacts

    @classmethod
    def from_dir(cls, tokenizer_dir: str | Path) -> "KQGMTokenizer":
        return cls(load_tokenizer_artifacts(tokenizer_dir))

    @property
    def m_col(self) -> str:
        return self.artifacts.m_col

    def transform_symbol(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Tokenize one symbol dataframe.

        Parameters
        ----------
        df:
            DataFrame containing at least date, symbol, open, high, low, close,
            volume, and the configured M column, usually turnover.
        symbol:
            Symbol used to select per-symbol M-bin edges.

        Returns
        -------
        pd.DataFrame
            Columns: date, symbol, K, Q, G, M, W. Rows with incomplete token
            features are dropped before tokenization.
        """
        if self.m_col not in df.columns:
            raise KeyError(f"{symbol}: missing required column '{self.m_col}' for M token.")

        edges_m = self.artifacts.edges_m_by_symbol.get(symbol)
        if self.artifacts.edges_q is None or self.artifacts.edges_g is None or edges_m is None:
            raise ValueError(f"{symbol}: missing codebook edges.")

        g = compute_geometry(df)
        d = compute_displacement(df)

        if self.artifacts.q_mode == "dc":
            q_cont = d["dC"].rename("q_cont")
            g_cont = d["gap"].rename("g_cont")
        elif self.artifacts.q_mode == "volnorm":
            q_cont = compute_volnorm_dc(
                df,
                vol_window=self.artifacts.vol_window,
                min_periods=self.artifacts.q_min_periods,
            ).rename("q_cont")
            g_cont = d["gap"].rename("g_cont")
        else:
            raise ValueError(f"Unknown q_mode='{self.artifacts.q_mode}'")

        m_cont = pd.to_numeric(df[self.m_col], errors="coerce").astype(float).rename(self.m_col)

        # Concatenate only q_cont and g_cont, already extracted from d, not the entire d frame.
        feat = pd.concat([df[["date"]], g, q_cont, g_cont, m_cont], axis=1)
        feat = feat.dropna().reset_index(drop=True)
        if feat.empty:
            return pd.DataFrame(columns=["date", "symbol", "K", "Q", "G", "M", "W"])

        Xg = feat[["b", "clv", "uw", "lw"]].to_numpy(dtype=float)
        K = self.artifacts.geom_kmeans.predict(Xg).astype(np.int32)
        Q = bin_to_int_per_symbol(feat["q_cont"].to_numpy(dtype=float), self.artifacts.edges_q)
        G = bin_to_int_per_symbol(feat["g_cont"].to_numpy(dtype=float), self.artifacts.edges_g)
        M = bin_to_int_per_symbol(feat[self.m_col].to_numpy(dtype=float), edges_m)
        W = composite_token_id(K, Q, G, M, Qn=self.artifacts.Qn, Gn=self.artifacts.Gn, Mn=self.artifacts.Mn)

        return pd.DataFrame(
            {
                "date": feat["date"].dt.date.astype(str),
                "symbol": symbol,
                "K": K,
                "Q": Q,
                "G": G,
                "M": M,
                "W": W,
            }
        )
