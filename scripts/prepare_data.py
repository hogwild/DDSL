#!/usr/bin/env python3
"""Convert local Chinese CSV exports to DDSL input CSVs.

Uses only the Python standard library. Run from any working directory:
  python3 scripts/prepare_data.py --dry-run
  python3 scripts/prepare_data.py
  python3 scripts/prepare_data.py --drop-suspension-placeholders

With --drop-suspension-placeholders, the default complete-kqgm-v3 policy drops
rows missing open/high/low/close/turnover after missing-marker normalization.
This covers the existing M=turnover representation's required raw inputs; it
does not remove structural first-row or sequence warmup rows. Date filtering
runs afterwards. Select --suspension-policy triple-na-v2 or strict-v1 to
reproduce an earlier policy. Use a new output root when changing policies.

Original files are read-only. Existing identical outputs are verified and reused;
different outputs are never overwritten. Each invocation writes a JSON report
under OUTPUT_ROOT/.conversion_reports, except --dry-run (no writes).
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("SSE50", "CSI300", "CSI500", "CSIA500")
# Unicode escapes preserve the original Chinese export headers; values name the English output columns.
COLUMN_MAP = {
    "\u4ee3\u7801": "symbol", "\u7b80\u79f0": "name", "\u65e5\u671f": "date",
    "\u524d\u6536\u76d8\u4ef7(\u5143)": "prev_close", "\u5f00\u76d8\u4ef7(\u5143)": "open",
    "\u6700\u9ad8\u4ef7(\u5143)": "high", "\u6700\u4f4e\u4ef7(\u5143)": "low", "\u6536\u76d8\u4ef7(\u5143)": "close",
    "\u6210\u4ea4\u91cf(\u80a1)": "volume", "\u6210\u4ea4\u91d1\u989d(\u5143)": "amount",
    "\u6da8\u8dcc(\u5143)": "chg", "\u6da8\u8dcc\u5e45(%)": "pct_chg", "\u5747\u4ef7(\u5143)": "avg_price",
    "\u6362\u624b\u7387(%)": "turnover",
    "A\u80a1\u6d41\u901a\u80a1\u672c(\u80a1)": "free_float_shares",
    "\u603b\u80a1\u672c(\u80a1)": "total_shares_outstanding",
}
REQUIRED = {"date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover"}
NUMERIC = set(COLUMN_MAP.values()) - {"date", "symbol", "name"}
PRICES = {"open", "high", "low", "close"}
MISSING = {"", "--", "N/A", "NA"}
FILE_PATTERN = re.compile(r"(?P<symbol>\d{6})\.(?P<exchange>SH|SZ|BJ)\.csv", re.I)
SUSPENSION_POLICY = (
    "suspension_placeholder_v1: volume/amount/turnover all missing; "
    "OHLC all equal the previous source row's close (exact decimal equality); "
    "prev_close, chg and pct_chg, when present as columns, must respectively "
    "equal that close, zero and zero. First rows and unmatched missing rows are retained."
)
DEFAULT_SUSPENSION_POLICY = "complete-kqgm-v3"
SUSPENSION_POLICIES = {
    "strict-v1": SUSPENSION_POLICY,
    "triple-na-v2": (
        "suspension_placeholder_v2: source volume/amount/turnover all exactly N/A "
        "after stripping whitespace; no price-equality or previous-row requirement. "
        "Zero, partial N/A and other missing markers do not match."
    ),
    "complete-kqgm-v3": (
        "complete_kqgm_v3: drop rows missing any open/high/low/close/turnover "
        "after empty/--/NA/N/A normalization, for M=turnover; no imputation. "
        "Other columns and nonpositive prices do not trigger this filter. "
        "Structural tokenizer first-row and sequence warmup handling is unchanged."
    ),
}
TRADING_COLUMNS = {"volume", "amount", "turnover"}
TOKEN_INPUT_COLUMNS = ("open", "high", "low", "close", "turnover")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_suspension_placeholder(values: dict[str, str], previous_close: str) -> bool:
    """Legacy strict-v1 rule on normalized values; retained for reproduction."""
    if not previous_close or any(values[c] for c in ("volume", "amount", "turnover")):
        return False
    close = Decimal(previous_close)
    if any(not values[c] or Decimal(values[c]) != close for c in PRICES):
        return False
    for column, expected in (("prev_close", close), ("chg", Decimal(0)), ("pct_chg", Decimal(0))):
        if column in values and (not values[column] or Decimal(values[column]) != expected):
            return False
    return True


def parse_date_bound(value: str) -> str:
    """Accept only a valid calendar date written as YYYY-MM-DD."""
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError(f"Expected YYYY-MM-DD, got {value!r}")
    return date.fromisoformat(value).isoformat()


def convert_file(
    source: Path, *, drop_suspension_placeholders: bool = False,
    start_date: str | None = None, end_date: str | None = None,
    suspension_policy: str = DEFAULT_SUSPENSION_POLICY,
) -> tuple[bytes, dict]:
    """Validate one file and return UTF-8 CSV bytes plus an audit record.

    Numeric text is copied exactly (no float serialization or unit conversion).
    Only explicit missing markers become empty fields. Row order is preserved;
    duplicate/invalid dates and unexpected text fail. The optional row filter
    records every removed date and requires chronological input. Inclusive date
    bounds apply afterwards, preserving the previous source row context.
    Validation and source quality statistics always cover the entire source.
    """
    if suspension_policy not in SUSPENSION_POLICIES:
        raise ValueError(f"Unknown suspension policy: {suspension_policy!r}")
    start_date = parse_date_bound(start_date) if start_date is not None else None
    end_date = parse_date_bound(end_date) if end_date is not None else None
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    match = FILE_PATTERN.fullmatch(source.name)
    if match is None:
        raise ValueError(f"Unexpected source filename: {source.name}")
    symbol = match["symbol"]
    full_symbol = f"{symbol}.{match['exchange'].upper()}"
    raw = source.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Cannot decode as UTF-8 or GB18030")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = next(reader, None)
    if not header:
        raise ValueError("Empty CSV or missing header")
    header = [column.strip() for column in header]
    unknown = set(header) - set(COLUMN_MAP) - {""}
    if unknown:
        raise ValueError(f"Unrecognized columns: {sorted(unknown)}")
    columns = [COLUMN_MAP[column] for column in header if column]
    if len(set(columns)) != len(columns) or not REQUIRED.issubset(columns):
        raise ValueError(f"Duplicate columns or missing required columns: {columns}")

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    dates: set[str] = set()
    missing: Counter = Counter()
    output_missing: Counter = Counter()
    nonpositive: Counter = Counter()
    previous_date = ""
    previous_close = ""
    kept_dates: list[str] = []
    dropped: list[dict] = []
    date_excluded_rows = 0
    chronological = True
    rows = 0
    for line_number, row in enumerate(reader, start=2):
        if len(row) != len(header):
            raise ValueError(f"Row {line_number}: expected {len(header)} fields, got {len(row)}")
        values = {}
        source_na_fields: set[str] = set()
        for column, value in zip(header, row):
            if not column:
                if value.strip():
                    raise ValueError(f"Row {line_number}: data in unnamed column")
                continue
            key = COLUMN_MAP[column]
            if key in NUMERIC:
                marker = value.strip()
                if marker == "N/A":
                    source_na_fields.add(key)
                if marker in MISSING:
                    missing[key] += 1
                    value = ""
                else:
                    try:
                        number = float(value)
                    except ValueError as exc:
                        raise ValueError(f"Row {line_number}: invalid {key}={value!r}") from exc
                    if not math.isfinite(number):
                        raise ValueError(f"Row {line_number}: non-finite {key}={value!r}")
                    if key in PRICES and number <= 0:
                        nonpositive[key] += 1
            values[key] = value
        if values["symbol"].strip().upper() != full_symbol:
            raise ValueError(f"Row {line_number}: symbol disagrees with filename")
        values["symbol"] = symbol
        values["date"] = date.fromisoformat(values["date"].strip()).isoformat()
        day = values["date"]
        if day in dates:
            raise ValueError(f"Row {line_number}: duplicate date {day}; no deduplication performed")
        chronological = chronological and day >= previous_date
        if drop_suspension_placeholders and not chronological:
            raise ValueError(f"Row {line_number}: row filtering requires chronological dates")
        previous_date = day
        dates.add(day)
        remove = False
        missing_inputs = []
        if drop_suspension_placeholders:
            if suspension_policy == "complete-kqgm-v3":
                missing_inputs = [column for column in TOKEN_INPUT_COLUMNS if not values[column]]
                remove = bool(missing_inputs)
            elif suspension_policy == "triple-na-v2":
                # Test the source markers, before normalization merges N/A with
                # empty/NA/-- fields. Prices do not participate in this policy.
                remove = TRADING_COLUMNS.issubset(source_na_fields)
            else:
                remove = is_suspension_placeholder(values, previous_close)
        previous_close = values["close"]
        if remove:
            entry = {"date": day, "source_csv_row": line_number}
            if suspension_policy == "triple-na-v2":
                entry["reason"] = "volume_amount_turnover_all_N/A"
            elif suspension_policy == "complete-kqgm-v3":
                entry["reason"] = "missing_required_token_inputs"
                entry["missing_columns"] = missing_inputs
            dropped.append(entry)
            continue
        if ((start_date is not None and day < start_date)
                or (end_date is not None and day > end_date)):
            date_excluded_rows += 1
            continue
        writer.writerow([values[column] for column in columns])
        kept_dates.append(day)
        output_missing.update(key for key in NUMERIC & values.keys() if not values[key])
        rows += 1
    if not rows:
        if dates and (start_date is not None or end_date is not None):
            raise ValueError(
                f"No data rows remain in requested date range [{start_date or 'unbounded'}, "
                f"{end_date or 'unbounded'}] after filtering; input_rows={len(dates)}, "
                f"dropped_rows={len(dropped)}, "
                f"date_excluded_rows={date_excluded_rows}"
            )
        raise ValueError("CSV contains no data rows")
    payload = output.getvalue().encode("utf-8")
    return payload, {
        "source": str(source), "source_sha256": sha256(raw), "source_bytes": len(raw),
        "encoding": encoding, "symbol": symbol, "exchange": match["exchange"].upper(),
        "rows": rows, "input_rows": len(dates),
        "dropped_rows": len(dropped), "dropped_suspension_rows": dropped,
        "start_date": start_date, "end_date": end_date,
        "date_excluded_rows": date_excluded_rows,
        "suspension_policy_name": suspension_policy if drop_suspension_placeholders else "preserve_all",
        "suspension_policy": SUSPENSION_POLICIES[suspension_policy] if drop_suspension_placeholders else "preserve_all",
        "date_min": min(kept_dates), "date_max": max(kept_dates),
        "chronological": chronological, "missing_values": dict(missing),
        "output_missing_values": dict(output_missing),
        "nonpositive_prices": dict(nonpositive), "output_sha256": sha256(payload),
        "output_bytes": len(payload),
    }


def write_or_verify(destination: Path, payload: bytes) -> str:
    """Publish a complete file atomically without replacing an existing file."""
    if destination.is_symlink():
        raise ValueError(f"Refusing output symlink: {destination}")
    if destination.exists():
        if destination.read_bytes() == payload:
            return "verified_existing"
        raise FileExistsError(f"Different output already exists; refusing overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".convert-", delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(payload)
            f.flush()
            os.link(temporary, destination)  # Atomic and fails if destination now exists.
        finally:
            temporary.unlink()
    return "written"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data/raw_unprocessed")
    parser.add_argument("--output-dir", "--output-root", dest="output_root", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, help="Default: all available CN datasets")
    parser.add_argument("--adjust", nargs="+", choices=("hfq", "qfq"), default=["hfq", "qfq"])
    parser.add_argument("--limit", type=int, default=0, help="First N files per directory for smoke checks; 0 means all")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing anything")
    parser.add_argument("--start-date", type=parse_date_bound, metavar="YYYY-MM-DD",
                        help="Inclusive output start date; default: no lower bound")
    parser.add_argument("--end-date", type=parse_date_bound, metavar="YYYY-MM-DD",
                        help="Inclusive output end date; default: no upper bound")
    parser.add_argument("--drop-suspension-placeholders", action="store_true",
                        help="Drop rows using the selected policy (default: complete-kqgm-v3, missing OHLC/turnover); use a new output root")
    parser.add_argument("--suspension-policy", choices=tuple(SUSPENSION_POLICIES), default=None,
                        help="Requires --drop-suspension-placeholders; default: complete-kqgm-v3; strict-v1 and triple-na-v2 reproduce earlier rules")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.start_date is not None and args.end_date is not None and args.start_date > args.end_date:
        parser.error("--start-date must be <= --end-date")
    if args.suspension_policy is not None and not args.drop_suspension_placeholders:
        parser.error("--suspension-policy requires --drop-suspension-placeholders")
    args.suspension_policy = args.suspension_policy or DEFAULT_SUSPENSION_POLICY
    return args


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve(strict=True)
    if args.output_root.is_symlink():
        raise ValueError("Output root must be a real directory, not the original raw-data symlink")
    output_root = args.output_root.resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Input and output roots must be disjoint")
    datasets = list(dict.fromkeys(args.datasets or [
        dataset for dataset in DATASETS
        if any((source_root / f"{dataset}_{adjust}").is_dir() for adjust in args.adjust)
    ]))
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "converter_sha256": sha256(Path(__file__).read_bytes()),
        "source_root": str(source_root), "output_root": str(output_root),
        "datasets": datasets, "adjust": args.adjust, "limit_per_directory": args.limit,
        "start_date": args.start_date, "end_date": args.end_date,
        "date_filter_policy": "inclusive_date_range_v1: after suspension filtering; validate all source rows",
        "dry_run": args.dry_run, "units": {
            "prices": "CNY", "volume": "shares", "amount": "CNY",
            "turnover": "percent (not divided by 100)", "pct_chg": "percent",
            "free_float_shares": "shares", "total_shares_outstanding": "shares",
        },
        "policy": "Preserve retained numerical text, units, row order and universe; empty/--/NA/N/A numeric fields become empty; no filling, deduplication or adjustment recalculation.",
        "suspension_policy_name": args.suspension_policy if args.drop_suspension_placeholders else "preserve_all",
        "suspension_policy": SUSPENSION_POLICIES[args.suspension_policy] if args.drop_suspension_placeholders else "preserve_all",
        "groups": {}, "files": [], "errors": [], "warnings": [],
    }
    if not datasets:
        raise ValueError(f"No supported dataset directories found under {source_root}")
    print(f"[INFO] Suspension policy: {report['suspension_policy_name']}", flush=True)
    for dataset in datasets:
        destination_dir = output_root / dataset
        if destination_dir.is_symlink() or destination_dir.resolve().parent != output_root:
            raise ValueError(f"Unsafe output dataset directory: {destination_dir}")
        for adjust in dict.fromkeys(args.adjust):
            directory = source_root / f"{dataset}_{adjust}"
            if not directory.is_dir() or directory.is_symlink():
                report["errors"].append({"source": str(directory), "error": "Missing directory or unsupported symlink"})
                continue
            files = sorted(path for path in directory.iterdir() if path.suffix.lower() == ".csv")
            if not files:
                report["errors"].append({"source": str(directory), "error": "No CSV files"})
                continue
            symbols = [FILE_PATTERN.fullmatch(path.name)["symbol"] for path in files if FILE_PATTERN.fullmatch(path.name)]
            if len(set(symbols)) != len(symbols):
                report["errors"].append({"source": str(directory), "error": "Multiple input files map to the same output symbol"})
                continue
            group = {"available_files": len(files), "symbols": sorted(symbols), "processed_files": 0,
                     "rows": 0, "dropped_rows": 0, "date_excluded_rows": 0}
            report["groups"][directory.name] = group
            print(f"[INFO] {directory.name}: {len(files)} CSV files", flush=True)
            for source in files[:args.limit or None]:
                try:
                    if source.is_symlink():
                        raise ValueError("Source file symlinks are not supported")
                    payload, record = convert_file(
                        source, drop_suspension_placeholders=args.drop_suspension_placeholders,
                        start_date=args.start_date, end_date=args.end_date,
                        suspension_policy=args.suspension_policy,
                    )
                    destination = destination_dir / f"{record['symbol']}_{adjust}.csv"
                    record.update(dataset=dataset, adjust=adjust, output=str(destination))
                    # Even a dry-run detects conflicting existing artifacts.
                    if args.dry_run:
                        if destination.is_symlink() or (destination.exists() and destination.read_bytes() != payload):
                            raise FileExistsError(f"Conflicting existing output: {destination}")
                        record["status"] = "validated"
                    else:
                        record["status"] = write_or_verify(destination, payload)
                    report["files"].append(record)
                    group["processed_files"] += 1
                    group["rows"] += record["rows"]
                    group["dropped_rows"] += record["dropped_rows"]
                    group["date_excluded_rows"] += record["date_excluded_rows"]
                except (ValueError, OSError, csv.Error) as exc:
                    report["errors"].append({"source": str(source), "error": str(exc)})
                    print(f"[ERROR] {source.name}: {exc}", flush=True)
            print(f"[INFO] {directory.name}: processed={group['processed_files']}, rows={group['rows']}, "
                  f"dropped={group['dropped_rows']}, date_excluded={group['date_excluded_rows']}", flush=True)
        hfq = report["groups"].get(f"{dataset}_hfq")
        qfq = report["groups"].get(f"{dataset}_qfq")
        if hfq and qfq and hfq["symbols"] != qfq["symbols"]:
            report["warnings"].append({
                "dataset": dataset, "message": "hfq and qfq symbol universes differ; all available files preserved",
                "only_hfq": sorted(set(hfq["symbols"]) - set(qfq["symbols"])),
                "only_qfq": sorted(set(qfq["symbols"]) - set(hfq["symbols"])),
            })
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "failed" if report["errors"] else "complete"
    for warning in report["warnings"]:
        print(f"[WARN] {warning['dataset']}: {warning['message']}")
    if not args.dry_run:
        report_path = output_root / ".conversion_reports" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex}.json"
        if report_path.parent.is_symlink():
            raise ValueError("Report directory must not be a symlink")
        write_or_verify(report_path, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print(f"[INFO] Report: {report_path}")
    print(f"[INFO] {report['status']}: {len(report['files'])} files, {len(report['errors'])} errors")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
