"""Identify existing standardized CSVs without claiming a conversion history."""
from __future__ import annotations

import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path

from pipeline.batch_common import sha256


def inventory(raw_dir: Path, adjust: str) -> list[dict]:
    files = sorted(raw_dir.glob(f"*_{adjust}.csv"))
    if not files:
        raise ValueError(f"No *_{adjust}.csv inputs under {raw_dir}")
    symbols = set()
    records = []
    for path in files:
        symbol = path.name.split("_", 1)[0]
        if symbol in symbols:
            raise ValueError(f"Multiple {adjust} CSVs for {symbol} in {raw_dir}")
        symbols.add(symbol)
        dates = []
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                dates.append(date.fromisoformat(row["date"].strip()).isoformat())
        if not dates or len(dates) != len(set(dates)):
            raise ValueError(f"Empty input or duplicate dates: {path}")
        records.append({"name": path.name, "symbol": symbol, "sha256": sha256(path),
                        "rows": len(dates), "date_min": min(dates), "date_max": max(dates)})
    return records


def validate_standardized_inputs(raw_dir: Path, records: list[dict]) -> None:
    """Check the CSV interface without modifying values or filtering rows."""
    required = {"date", "open", "high", "low", "close", "turnover"}
    for record in records:
        path = raw_dir / record["name"]
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing standardized columns in {path}: {sorted(required - set(reader.fieldnames or []))}")
            previous = None
            for row in reader:
                current = date.fromisoformat(row["date"].strip())
                if previous is not None and current <= previous:
                    raise ValueError(f"Dates must be strictly increasing: {path}")
                if "symbol" in row and row["symbol"].strip() != record["symbol"]:
                    raise ValueError(f"Symbol column does not match filename: {path}")
                previous = current


def create_registration(raw_dir: Path, dataset: str, adjust: str, source_description: str) -> dict:
    """Record current bytes and an author declaration; never infer past filtering."""
    if not dataset.strip() or not source_description.strip():
        raise ValueError("Dataset and source description must be nonempty")
    if adjust not in {"hfq", "qfq"}:
        raise ValueError("Adjustment must be hfq or qfq")
    raw_dir = raw_dir.resolve()
    records = inventory(raw_dir, adjust)
    validate_standardized_inputs(raw_dir, records)
    if records != inventory(raw_dir, adjust):
        raise ValueError("Input CSVs changed during registration")
    return {
        "schema_version": 1, "kind": "historical_standardized_data",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"name": dataset, "adjust": adjust}, "raw_dir": str(raw_dir),
        "source_description": source_description.strip(),
        "copy_statement": "author_declared_copy_without_reprocessing",
        "historical_preprocessing": "not_recorded", "registration_transforms": [],
        "limitations": "Hashes identify the current copy; historical processing and manuscript provenance are not independently certified.",
        "files": records,
    }


def registration_provenance(records: list[dict], dataset: dict, path: Path) -> tuple[dict, Path]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("schema_version") != 1 or report.get("kind") != "historical_standardized_data"
            or report.get("dataset") != {"name": dataset["name"], "adjust": dataset["adjust"]}
            or report.get("copy_statement") != "author_declared_copy_without_reprocessing"
            or report.get("historical_preprocessing") != "not_recorded"
            or report.get("registration_transforms") != []
            or not isinstance(report.get("source_description"), str)
            or not report["source_description"].strip()):
        raise ValueError("Invalid historical data registration or dataset/adjustment mismatch")
    # Exact inventory equality detects additions, removals, renames and changed bytes.
    # The recorded absolute location is informational, allowing unchanged copies to move.
    if report.get("files") != records:
        raise ValueError("Data registration does not match every current input CSV; register a new data version")
    return {"policy": "historical-unrecorded", "provenance_kind": report["kind"],
            "provenance_file": "data_registration.json", "report_sha256": sha256(path),
            "historical_preprocessing": report["historical_preprocessing"],
            "source_description": report["source_description"],
            "copy_statement": report["copy_statement"]}, path
