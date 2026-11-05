from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_STATE_COL = "joint_state"
DEFAULT_CURRENT_COL = "current_trend"
DEFAULT_FUTURE_COL = "future_trend"
DEFAULT_DATE_COL = "date"
DEFAULT_SYMBOL_COL = "symbol"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_iteration_label_dir(
    *,
    output_root: str | Path,
    dataset: str,
    iteration: int,
    split: str,
) -> Path:
    return Path(output_root) / dataset / f"iter_{iteration}" / "labels" / split


def resolve_single_analysis_dir(
    *,
    output_root: str | Path,
    dataset: str,
    iteration: int,
) -> Path:
    return (
        Path(output_root)
        / dataset
        / f"iter_{iteration}"
        / "label_analysis"
        / "single_iteration"
    )


def resolve_pairwise_analysis_dir(
    *,
    output_root: str | Path,
    dataset: str,
    new_iteration: int,
) -> Path:
    return (
        Path(output_root)
        / dataset
        / f"iter_{new_iteration}"
        / "label_analysis"
        / "compare_to_prev"
    )


def list_label_files(
    label_dir: str | Path,
    *,
    adjust: str,
    pattern: str | None = None,
) -> list[Path]:
    label_dir = Path(label_dir)

    if pattern:
        files = sorted(label_dir.glob(pattern))
    else:
        files = sorted(label_dir.glob(f"*_{adjust}.csv"))
        if not files:
            files = sorted(label_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No label CSV files found under {label_dir}")

    return files


def load_label_dir(
    label_dir: str | Path,
    *,
    adjust: str,
    pattern: str | None = None,
    date_col: str = DEFAULT_DATE_COL,
    symbol_col: str = DEFAULT_SYMBOL_COL,
) -> pd.DataFrame:
    files = list_label_files(label_dir, adjust=adjust, pattern=pattern)

    frames: list[pd.DataFrame] = []

    for path in files:
        df = pd.read_csv(path)

        if symbol_col not in df.columns:
            symbol = path.name.split("_", 1)[0]
            df[symbol_col] = symbol

        df[symbol_col] = df[symbol_col].astype(str)

        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    sort_cols = [symbol_col]
    if date_col in out.columns:
        sort_cols.append(date_col)

    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def load_symbol_groups(
    *,
    codebook_dir: str | None = None,
    output_root: str | None = None,
    dataset: str | None = None,
) -> dict[str, set[str]] | None:
    if codebook_dir is not None:
        info_path = Path(codebook_dir) / "symbol_codebook_info.json"
    elif output_root is not None and dataset is not None:
        info_path = Path(output_root) / dataset / "codebook" / "symbol_codebook_info.json"
    else:
        return None

    if not info_path.exists():
        print(f"[WARN] symbol_codebook_info.json not found: {info_path}")
        return None

    with open(info_path, "r", encoding="utf-8") as f:
        symbol_info = json.load(f)
    
    return {
        "has_model_score": set(
            str(s).zfill(6) for s in symbol_info.get("symbols_with_codebook", [])
        ),
        "no_model_score": set(
            str(s).zfill(6) for s in symbol_info.get("symbols_without_codebook", [])
        ),
    }


def require_columns(df: pd.DataFrame, cols: list[str], name: str = "df") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def to_json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_json_safe(v) for v in x]
    if isinstance(x, tuple):
        return [to_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, pd.Timestamp):
        return str(x)
    if x is pd.NA:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(obj), f, indent=2, ensure_ascii=False)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def value_counts_dict(s: pd.Series) -> dict[str, int]:
    vc = s.value_counts(dropna=False).sort_index()
    return {str(k): int(v) for k, v in vc.items()}


def distribution_dict(s: pd.Series) -> dict[str, float]:
    vc = s.value_counts(dropna=False).sort_index()
    total = float(vc.sum())
    if total <= 0:
        return {}
    return {str(k): float(v / total) for k, v in vc.items()}


def flatten_nested_dict(
    d: dict[str, Any],
    *,
    prefix: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """
    Flatten nested dictionaries for CSV summary output.

    Lists and dicts that are too complex are converted to JSON strings.
    """
    out: dict[str, Any] = {}

    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)

        if isinstance(v, dict):
            # If it is a simple scalar dict, flatten it.
            if all(not isinstance(x, (dict, list, tuple)) for x in v.values()):
                for kk, vv in v.items():
                    out[f"{key}{sep}{kk}"] = vv
            else:
                # For deeply nested objects, recursively flatten one level.
                nested = flatten_nested_dict(v, prefix=key, sep=sep)
                out.update(nested)
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(to_json_safe(v), ensure_ascii=False)
        else:
            out[key] = v

    return out