#!/usr/bin/env python3
"""Register an unchanged copy of historical standardized data; never convert CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline.batch_common import freeze
from pipeline.data_registration import create_registration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adjust", choices=("hfq", "qfq"), required=True)
    parser.add_argument("--source-description", required=True, help="Author-provided origin of the historical copy")
    parser.add_argument("--copied-without-reprocessing", action="store_true", required=True,
                        help="Confirm that this copy has not been reprocessed")
    parser.add_argument("--output", type=Path, required=True, help="New registration JSON outside the raw-data directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing a registration")
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.is_relative_to(args.raw_dir.resolve()):
            raise ValueError("Registration output must be outside the raw-data directory")
        if output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Registration output already exists: {args.output}")
        report = create_registration(args.raw_dir, args.dataset, args.adjust, args.source_description)
        print(f"[INFO] Verified {len(report['files'])} CSVs; historical preprocessing is not recorded.")
        if args.dry_run:
            print("[DRY RUN] No files written or transformed.")
        else:
            freeze(output, json.dumps(report, indent=2) + "\n")
            print(f"[OK] Data registration: {output}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"[ERROR] {exc}\n")


if __name__ == "__main__":
    main()
