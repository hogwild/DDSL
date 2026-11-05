"""
check_price_quality.py
======================
Scan raw CSVs for two types of unsuitable stocks and write a combined blacklist:

  Type A - price quality issues
    Nonpositive back-adjusted prices (<=0) make log returns undefined.

  Type B - insufficient data in the training window
    Fewer than min_rows valid observations before fit_end_date prevent
    codebook inclusion and meaningful participation in the pipeline.

Run before codebook fitting to exclude unsuitable data from the pipeline.

Usage
-----
Import as a module (called by run_check_price_quality.py):
    from data.check_price_quality import check_price_quality, save_blacklist

Run directly for debugging:
    python src/data/check_price_quality.py \\
        --raw_dir data/raw/CSI300 \\
        --adjust hfq \\
        --fit_end_date 2018-12-31 \\
        --min_rows 50 \\
        --out_blacklist configs/symbols/CSI300_blacklist.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================================================================
# Core validation logic
# =============================================================================

PRICE_COLS = ["open", "high", "low", "close"]

# Exclusion reason types for blacklist entries
REASON_NEGATIVE_PRICE = "negative_price"   # Type A
REASON_INSUFFICIENT_ROWS = "insufficient_rows"  # Type B


def _load_raw_csv(fpath: Path) -> pd.DataFrame:
    df = pd.read_csv(fpath, dtype={"symbol": str})
    if "date" not in df.columns:
        raise ValueError(f"{fpath.name}: missing 'date' column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _check_negative_price(
    sym: str,
    df: pd.DataFrame,
) -> list[dict]:
    """
    Type A: check price columns for nonpositive values.
    Return one record for each affected column.
    """
    records = []
    for col in PRICE_COLS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        neg_mask = series <= 0
        neg_count = int(neg_mask.sum())
        if neg_count == 0:
            continue
        records.append({
            "symbol": sym,
            "reason": REASON_NEGATIVE_PRICE,
            "col": col,
            "total_rows": len(df),
            "flagged_rows": neg_count,
            "flagged_ratio": round(neg_count / len(df), 4),
            "date_min": str(df.loc[neg_mask, "date"].dt.strftime("%Y-%m-%d").min()),
            "date_max": str(df.loc[neg_mask, "date"].dt.strftime("%Y-%m-%d").max()),
            "detail": (
                f"{neg_count} non-positive '{col}' rows "
                f"({neg_count / len(df):.1%}, "
                f"{df.loc[neg_mask, 'date'].dt.strftime('%Y-%m-%d').min()} ~ "
                f"{df.loc[neg_mask, 'date'].dt.strftime('%Y-%m-%d').max()})"
            ),
        })
    return records


def _check_insufficient_rows(
    sym: str,
    df: pd.DataFrame,
    fit_end_date: pd.Timestamp,
    min_rows: int,
) -> list[dict]:
    """
    Type B: check whether valid observations before fit_end_date fall below min_rows.
    """
    df_fit = df[df["date"] <= fit_end_date]
    row_count = len(df_fit)
    if row_count >= min_rows:
        return []
    return [{
        "symbol": sym,
        "reason": REASON_INSUFFICIENT_ROWS,
        "col": None,
        "total_rows": len(df),
        "flagged_rows": row_count,
        "flagged_ratio": round(row_count / max(len(df), 1), 4),
        "date_min": str(df_fit["date"].dt.strftime("%Y-%m-%d").min()) if row_count > 0 else "N/A",
        "date_max": str(df_fit["date"].dt.strftime("%Y-%m-%d").max()) if row_count > 0 else "N/A",
        "detail": (
            f"only {row_count} rows before fit_end_date "
            f"{fit_end_date.strftime('%Y-%m-%d')} "
            f"(min_rows={min_rows})"
        ),
    }]


def check_price_quality(
    raw_dir: Path,
    adjust: str,
    fit_end_date: str | pd.Timestamp,
    min_rows: int = 60,  # Must be >= training.seq_len.
    symbols: Optional[set[str]] = None,
) -> tuple[list[str], list[dict]]:
    """
    Run both checks on all *_{adjust}.csv files under raw_dir.

    Parameters
    ----------
    raw_dir       : raw data directory
    adjust        : price adjustment convention, such as 'hfq'
    fit_end_date  : codebook fitting cutoff date (YYYY-MM-DD or Timestamp)
    min_rows      : minimum valid observations required in the fitting window
    symbols       : optional symbol allowlist; None scans all files

    Returns
    -------
    blacklist : sorted, deduplicated list of symbols to exclude
    report    : detailed records, one for each detected issue
    """
    fit_end_ts = pd.Timestamp(fit_end_date)
    files = sorted(raw_dir.glob(f"*_{adjust}.csv"))

    if symbols is not None:
        files = [f for f in files if f.name.split("_", 1)[0] in symbols]

    blacklist_set: set[str] = set()
    report: list[dict] = []

    for fpath in files:
        sym = fpath.name.split("_", 1)[0]
        df = _load_raw_csv(fpath)
        if df is None:
            continue

        # Type A: check for nonpositive prices.
        neg_records = _check_negative_price(sym, df)
        if neg_records:
            blacklist_set.add(sym)
            report.extend(neg_records)

        # Type B: insufficient training-window data; identify symbols not already excluded by A.
        # Also record findings for symbols excluded by A to keep the report complete.
        insuf_records = _check_insufficient_rows(sym, df, fit_end_ts, min_rows)
        if insuf_records:
            blacklist_set.add(sym)
            report.extend(insuf_records)

    return sorted(blacklist_set), report


# =============================================================================
# File output
# =============================================================================

def save_blacklist(
    blacklist: list[str],
    report: list[dict],
    out_path: Path,
    fit_end_date: str,
    min_rows: int,
) -> None:
    """
    Write a text blacklist grouped by type, with comments explaining exclusions.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    neg_syms   = sorted({r["symbol"] for r in report if r["reason"] == REASON_NEGATIVE_PRICE})
    insuf_syms = sorted({r["symbol"] for r in report if r["reason"] == REASON_INSUFFICIENT_ROWS})
    # Symbols belonging only to type B, without type A price issues
    insuf_only = sorted(set(insuf_syms) - set(neg_syms))

    lines: list[str] = [
        "# Auto-generated: do not edit manually. Rerun run_check_price_quality.py to update.",
        f"# fit_end_date = {fit_end_date}   min_rows = {min_rows}",
        "#",
    ]

    # --- Type A ---
    lines.append("# ── Type A: nonpositive adjusted prices invalidate log returns ──────────────────")
    if neg_syms:
        for sym in neg_syms:
            sym_records = [r for r in report
                           if r["symbol"] == sym and r["reason"] == REASON_NEGATIVE_PRICE]
            for r in sym_records:
                lines.append(f"#   {sym}  {r['detail']}")
        lines.append("#")
        for sym in neg_syms:
            lines.append(sym)
    else:
        lines.append("#   (none)")
    lines.append("#")

    # --- Type B ---
    lines.append("# ── Type B: too few rows in the fit window for codebook fitting ───────────────────")
    if insuf_only:
        for sym in insuf_only:
            sym_records = [r for r in report
                           if r["symbol"] == sym and r["reason"] == REASON_INSUFFICIENT_ROWS]
            for r in sym_records:
                lines.append(f"#   {sym}  {r['detail']}")
        lines.append("#")
        for sym in insuf_only:
            lines.append(sym)
    else:
        lines.append("#   (none)")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_report_json(report: list[dict], out_path: Path) -> None:
    import json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def load_blacklist(txt_path: Path) -> set[str]:
    """
    Read a text blacklist and return its symbol set.
    Ignore blank lines and comments beginning with '#'.
    """
    if not txt_path.exists():
        return set()
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


# =============================================================================
# Debug CLI; the pipeline uses run_check_price_quality.py.
# =============================================================================

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Scan raw CSV files and generate a price-quality blacklist.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--raw_dir",       required=True,              help="Raw-data directory")
    ap.add_argument("--adjust",        default="hfq",              help="Price adjustment")
    ap.add_argument("--fit_end_date",  required=True,              help="Fit window end date (YYYY-MM-DD)")
    ap.add_argument("--min_rows",      type=int, default=60,       help="Minimum number of valid rows in the fit window")
    ap.add_argument("--out_blacklist", required=True,              help="Blacklist output path (.txt)")
    ap.add_argument("--out_report",    default=None,               help="Detailed report output path (.json)")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    raw_dir = Path(args.raw_dir)
    print(f"[INFO] Scanning {raw_dir} (*_{args.adjust}.csv) ...")
    print(f"[INFO] fit_end_date={args.fit_end_date}  min_rows={args.min_rows}")

    blacklist, report = check_price_quality(
        raw_dir=raw_dir,
        adjust=args.adjust,
        fit_end_date=args.fit_end_date,
        min_rows=args.min_rows,
    )

    out_blacklist = Path(args.out_blacklist)
    save_blacklist(blacklist, report, out_blacklist,
                   fit_end_date=args.fit_end_date, min_rows=args.min_rows)
    print(f"[OK]  blacklist ({len(blacklist)} symbols) → {out_blacklist}")

    if args.out_report:
        out_report = Path(args.out_report)
        save_report_json(report, out_report)
        print(f"[OK]  report ({len(report)} entries) → {out_report}")

    # Print the summary.
    neg_syms   = sorted({r["symbol"] for r in report if r["reason"] == REASON_NEGATIVE_PRICE})
    insuf_syms = sorted({r["symbol"] for r in report if r["reason"] == REASON_INSUFFICIENT_ROWS})
    insuf_only = sorted(set(insuf_syms) - set(neg_syms))

    print(f"\n=== Quality check summary ===")
    print(f"Type A (nonpositive prices): {len(neg_syms)} symbols  {neg_syms}")
    print(f"Type B (insufficient rows) : {len(insuf_only)} symbols  {insuf_only}")
    print(f"Total excluded            : {len(blacklist)} symbols")


if __name__ == "__main__":
    main()