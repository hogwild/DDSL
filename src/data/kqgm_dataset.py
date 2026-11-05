from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from states import state_name_to_id, state_id_to_pair_ids


@dataclass(frozen=True)
class SampleIndex:
    symbol: str
    end_pos: int


class KQGMJointDataset(Dataset):
    """Sliding-window dataset for DDSL decoder-transformer training.

    Expected parquet columns:
        symbol, date, K, Q, G, M, joint_state

    Each sample uses rows [t - seq_len + 1, ..., t] as input and predicts
    ``joint_state`` at row t.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        label_col: str = "joint_state",
        symbol_col: str = "symbol",
        date_col: str = "date",
    ) -> None:
        required = {symbol_col, date_col, "K", "Q", "G", "M", label_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")

        self.seq_len = int(seq_len)
        self.label_col = label_col
        self.symbol_col = symbol_col
        self.date_col = date_col

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].isna().any():
            raise ValueError("Dataset contains invalid dates.")
        df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

        # Store per-symbol data as numpy arrays instead of DataFrames.
        # __getitem__ uses plain array slicing, which is ~10x faster than
        # pandas.iloc on the same data.
        self.symbol_arrays: dict[str, dict[str, np.ndarray]] = {}
        self.samples: list[SampleIndex] = []

        for sym, g in df.groupby(symbol_col, sort=False):
            g = g.reset_index(drop=True)
            if len(g) < self.seq_len:
                continue
            sym_str = str(sym)

            # Pre-convert label column to integer ids.
            joint_ids = np.array(
                [state_name_to_id(str(v)) for v in g[label_col]],
                dtype=np.int64,
            )

            self.symbol_arrays[sym_str] = {
                "K":        g["K"].to_numpy(dtype=np.int64),
                "Q":        g["Q"].to_numpy(dtype=np.int64),
                "G":        g["G"].to_numpy(dtype=np.int64),
                "M":        g["M"].to_numpy(dtype=np.int64),
                "joint_id": joint_ids,
                "dates":    g[date_col].dt.strftime("%Y-%m-%d").to_numpy(),
            }

            for end_pos in range(self.seq_len - 1, len(g)):
                self.samples.append(SampleIndex(symbol=sym_str, end_pos=end_pos))

        if not self.samples:
            raise ValueError("No valid samples were built. Check seq_len and input data length.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        arrs = self.symbol_arrays[sample.symbol]
        start = sample.end_pos - self.seq_len + 1
        end = sample.end_pos + 1  # exclusive

        y_joint = int(arrs["joint_id"][sample.end_pos])
        y_current, y_future = state_id_to_pair_ids(y_joint)

        return {
            "K": torch.as_tensor(arrs["K"][start:end]),
            "Q": torch.as_tensor(arrs["Q"][start:end]),
            "G": torch.as_tensor(arrs["G"][start:end]),
            "M": torch.as_tensor(arrs["M"][start:end]),
            "y_joint":   torch.tensor(y_joint,   dtype=torch.long),
            "y_current": torch.tensor(y_current, dtype=torch.long),
            "y_future":  torch.tensor(y_future,  dtype=torch.long),
            "symbol": sample.symbol,
            "date":   arrs["dates"][sample.end_pos],
        }

    def get_label_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_joint   = np.zeros(len(self.samples), dtype=np.int64)
        y_current = np.zeros(len(self.samples), dtype=np.int64)
        y_future  = np.zeros(len(self.samples), dtype=np.int64)
        for i, sample in enumerate(self.samples):
            jid = int(self.symbol_arrays[sample.symbol]["joint_id"][sample.end_pos])
            c, f = state_id_to_pair_ids(jid)
            y_joint[i]   = jid
            y_current[i] = c
            y_future[i]  = f
        return y_current, y_future, y_joint


def collate_kqgm_joint(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "K": torch.stack([b["K"] for b in batch], dim=0),
        "Q": torch.stack([b["Q"] for b in batch], dim=0),
        "G": torch.stack([b["G"] for b in batch], dim=0),
        "M": torch.stack([b["M"] for b in batch], dim=0),
        "y_joint":   torch.stack([b["y_joint"]   for b in batch], dim=0),
        "y_current": torch.stack([b["y_current"] for b in batch], dim=0),
        "y_future":  torch.stack([b["y_future"]  for b in batch], dim=0),
        "symbol": [b["symbol"] for b in batch],
        "date":   [b["date"]   for b in batch],
    }