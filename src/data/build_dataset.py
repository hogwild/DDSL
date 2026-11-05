"""
build_dataset.py
================
Build training-ready parquet datasets by tokenizing labeled CSV files and
attaching label/diagnostic columns.

This module owns all dataset-level concerns: file discovery, CSV loading, label
forwarding, date filtering, tail trimming, per-symbol parquet writing, and
combined parquet writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from common.symbols import (
    normalize_symbol_series,
    normalize_symbol_value,
    validate_symbol_series,
)

from .tokenizer_KQG_persymbolM import KQGMTokenizer
from .check_price_quality import load_blacklist


LABEL_COLS_TO_FORWARD: List[str] = [
    # current_future 9-state scheme
    "joint_state",
    "current_trend",
    "future_trend",
    # short_long 9-state scheme
    "decoded_short_trend",
    "decoded_long_trend",
    # shared diagnostics
    "smooth",
]


REQUIRED_OHLCV_COLS: List[str] = ["date", "open", "high", "low", "close", "volume"]


def list_csv_files(raw_dir: Path, adjust: str) -> List[Path]:
    """List adjusted CSV files under a directory."""
    return sorted(Path(raw_dir).glob(f"*_{adjust}.csv"))


def load_one_csv(path: Path, require_state: bool = True, market: str="cn") -> pd.DataFrame:
    """Load one labeled OHLCV CSV and perform basic validation/typing."""
    df = pd.read_csv(path, dtype={"symbol": str})
    if "symbol" not in df.columns:
        symbol = path.name.split("_", 1)[0]
        df["symbol"] = normalize_symbol_value(symbol, market=market)
    else:
        df["symbol"] = normalize_symbol_series(df["symbol"], market=market)

    if require_state and "joint_state" not in df.columns:
        raise ValueError(f"{path.name}: missing state column 'joint_state'")

    missing = [c for c in REQUIRED_OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def attach_labels(
    tokens: pd.DataFrame,
    source_df: pd.DataFrame,
    label_cols: List[str] = LABEL_COLS_TO_FORWARD,
) -> pd.DataFrame:
    """
    Attach label/diagnostic columns to token rows by matching date.

    Tokenization drops initial rows with NaN displacement features, so direct
    positional assignment can become fragile. Date-based merging keeps the
    alignment explicit.
    """
    if tokens.empty:
        return tokens

    keep_cols = ["date"] + [c for c in label_cols if c in source_df.columns]
    if len(keep_cols) == 1:
        return tokens

    labels = source_df[keep_cols].copy()
    labels["date"] = pd.to_datetime(labels["date"]).dt.date.astype(str)

    out = tokens.merge(labels, on="date", how="left", validate="one_to_one")
    return out


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Filter rows by inclusive date range."""
    if df.empty or (start_date is None and end_date is None):
        return df

    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None

    dts = pd.to_datetime(df["date"], errors="coerce")
    mask = dts.notna()
    if start_ts is not None:
        mask &= dts >= start_ts
    if end_ts is not None:
        mask &= dts <= end_ts

    return df.loc[mask].reset_index(drop=True)


def trim_tail(df: pd.DataFrame, trim_tail_days: int) -> pd.DataFrame:
    """Drop the last N rows for one symbol, usually to remove unreliable future labels."""
    trim_tail_days = int(trim_tail_days)
    if trim_tail_days <= 0:
        return df
    if len(df) <= trim_tail_days:
        return df.iloc[0:0].copy()
    return df.iloc[:-trim_tail_days].reset_index(drop=True)


def build_one_symbol_dataset(
    csv_path: Path,
    tokenizer: KQGMTokenizer,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    trim_tail_days: int = 0,
    require_state: bool = True,
    label_cols: List[str] = LABEL_COLS_TO_FORWARD,
    market: str="cn",
) -> pd.DataFrame:
    """Tokenize one labeled CSV and attach labels."""
    symbol = normalize_symbol_value(
        csv_path.name.split("_", 1)[0],
        market=market,
    )
    df = load_one_csv(
        csv_path,
        require_state=require_state,
        market=market,
    )

    tokens = tokenizer.transform_symbol(df, symbol=symbol)
    tokens = attach_labels(tokens, df, label_cols=label_cols)
    tokens = filter_by_date_range(tokens, start_date=start_date, end_date=end_date)
    tokens = trim_tail(tokens, trim_tail_days=trim_tail_days)
    if not tokens.empty:
        tokens["symbol"] = normalize_symbol_series(tokens["symbol"], market=market)
        validate_symbol_series(
            tokens["symbol"],
            market=market,
            name=f"{csv_path.name}:tokens.symbol",
        )
    return tokens


def run_split(
    split_name: str,
    split_cfg: dict,
    shared: dict,
    max_files_override: Optional[int] = None,
    blacklist: Optional[set[str]] = None,   # Added
) -> None:
    """Build token parquet files for one split: train / val / test."""
    print(f"\n{'=' * 56}")
    print(f"  SPLIT: {split_name}")
    print(f"{'=' * 56}")

    raw_dir = Path(split_cfg["raw_dir"])
    out_dir = Path(split_cfg["out_dir"])
    tokenizer_dir = Path(shared["tokenizer"]["dir"])
    adjust = shared["dataset"].get("adjust", "hfq")
    market = shared["dataset"].get("market", "cn")

    opts = shared.get("options", {})
    trim_tail_days = int(opts.get("trim_tail_days", 0))
    combine = bool(opts.get("combine", True))
    max_files = int(opts.get("max_files", 0))
    require_state = bool(opts.get("require_state", True))
    if max_files_override is not None:
        max_files = max_files_override

    start_date = split_cfg.get("start_date")
    end_date = split_cfg.get("end_date")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  raw_dir       : {raw_dir}")
    print(f"  tokenizer_dir : {tokenizer_dir}")
    print(f"  out_dir       : {out_dir}")
    print(f"  date range    : {start_date} → {end_date}")
    print(f"  market        : {market}")
    print(f"  trim_tail_days: {trim_tail_days}  combine: {combine}")

    tokenizer = KQGMTokenizer.from_dir(tokenizer_dir)

    files = list_csv_files(raw_dir, adjust)
    # Apply the blacklist filter.
    if blacklist:
        before = len(files)
        files = [f for f in files if f.name.split("_", 1)[0] not in blacklist]
        print(f"[INFO] Blacklist filtered {before - len(files)} files, "
              f"{len(files)} remaining.")
    if max_files > 0:
        files = files[:max_files]
        print(f"  [DEBUG] limited to {max_files} files")
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir} matching '*_{adjust}.csv'")

    print(f"  {len(files)} files to process.\n")

    combined_rows: List[pd.DataFrame] = []

    for i, csv_path in enumerate(files, 1):
        symbol = csv_path.name.split("_", 1)[0]
        try:
            out = build_one_symbol_dataset(
                csv_path=csv_path,
                tokenizer=tokenizer,
                start_date=start_date,
                end_date=end_date,
                trim_tail_days=trim_tail_days,
                require_state=require_state,
                market=market,
            )
        except ValueError as exc:
            if "missing codebook edges" in str(exc):
                print(f"  [{i}/{len(files)}] SKIP {symbol}: {exc}")
                continue
            raise

        if out.empty:
            raise RuntimeError(f"[{i}/{len(files)}] {symbol}: produced 0 rows after processing")

        if combine:
            combined_rows.append(out)
        else:
            out_path = out_dir / f"{symbol}_tokens_{adjust}.parquet"
            out["symbol"] = normalize_symbol_series(out["symbol"], market=market)
            validate_symbol_series(
                out["symbol"],
                market=market,
                name=f"{out_path.name}:symbol",
            )
            out.to_parquet(out_path, index=False)
            if i % 50 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {symbol}: {len(out)} rows -> {out_path.name}")

    if combine and combined_rows:
        all_df = pd.concat(combined_rows, ignore_index=True)

        all_df["symbol"] = normalize_symbol_series(all_df["symbol"], market=market)
        validate_symbol_series(
            all_df["symbol"],
            market=market,
            name=f"ALL_tokens_{adjust}.parquet:symbol",
        )
        all_df = (
            all_df
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )
        
        out_path = out_dir / f"ALL_tokens_{adjust}.parquet"
        all_df.to_parquet(out_path, index=False)
        print(f"\n  [COMBINE] {len(all_df):,} rows -> {out_path.resolve()}")
    elif combine and not combined_rows:
        raise RuntimeError("combine=True but no rows collected — check data and config")
    # elif combine and not combined_rows:
    #     print("  [WARN] combine=true but no rows collected — output parquet not written.")