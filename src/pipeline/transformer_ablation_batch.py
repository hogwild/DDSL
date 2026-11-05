"""Prepare and locally run frozen Transformer beta-ablation batches.

Orchestration only: existing quality checks, pipeline stages and metric code are
reused. Preparation never trains; --dry-run never writes files or runs tasks.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from pipeline.batch_common import (
    code_inventory, execute_local_cell, freeze, load_manifest, prepare_and_run, run_local_batch, sha256,
)
from pipeline.pipeline_config import compile_pipeline_config, load_pipeline_config

from pipeline.data_registration import inventory, registration_provenance, validate_standardized_inputs

DEFAULT_BETAS = ["01", "02", "03", "04", "05", "06", "07", "08", "inf"]
DEFAULT_SEEDS = [41, 42, 43]


def conversion_provenance(records: list[dict], dataset: dict, reports: list[Path]) -> tuple[dict, Path]:
    """Use a complete conversion report whose output hashes match every input.

    Old absolute output locations are deliberately ignored: copying verified
    standardized inputs between machines does not change their identity.
    """
    for path in sorted(reports, reverse=True):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete" or report.get("dry_run"):
            continue
        policy = report.get("suspension_policy_name")
        if policy not in {"preserve_all", "strict-v1", "triple-na-v2", "complete-kqgm-v3"}:
            continue
        # Intermediate reports used a separate filter. They must not be
        # mistaken for the suspension-only policy named in their metadata.
        if report.get("token_input_policy_name", "preserve_all") != "preserve_all":
            continue
        outputs = {Path(f["output"]).name: f for f in report.get("files", [])
                   if f.get("dataset") == dataset["name"] and f.get("adjust") == dataset["adjust"]}
        if all(outputs.get(f["name"], {}).get("output_sha256") == f["sha256"] for f in records):
            if policy == "complete-kqgm-v3" and any(
                outputs[f["name"]].get("suspension_policy_name") != policy for f in records
            ):
                raise ValueError(f"Inconsistent conversion policy in {path}")
            return {"policy": policy, "report_sha256": sha256(path),
                    "converter_sha256": report.get("converter_sha256"),
                    "source_root": report.get("source_root"),
                    "conversion_start_date": report.get("start_date"),
                    "conversion_end_date": report.get("end_date"),
                    "sources": [{k: outputs[f["name"]].get(k) for k in
                                 ("source", "source_sha256", "input_rows", "rows", "dropped_rows")}
                                for f in records]}, path
    raise ValueError("No complete conversion report matches every input CSV and its policy. "
                     "Pass --conversion-report <matching .conversion_reports/*.json>; "
                     "for an unchanged historical copy, use scripts/register_data.py and --data-registration. "
                     "Preprocessing policy is never inferred from filenames.")


def cell_command(manifest: dict, cell: dict) -> list[str]:
    return [sys.executable, "-u", str(REPO / "src/pipeline/run_transformer_pipeline.py"),
            "--config", str(Path(manifest["batch_dir"]) / "pipeline.yaml"),
            "--output_root", cell["output_root"], "--seed", str(cell["seed"]),
            "--labeler", cell["labeler"], "--beta_model", str(cell["beta_model"])]


def make_plan(args: argparse.Namespace) -> tuple[dict, dict, Path | None, str]:
    compiled = load_pipeline_config(str(args.config))
    cfg = compiled.resolved
    if cfg["pipeline"]["start_iter"] != 0:
        raise ValueError("A new ablation batch must start at iteration 0")
    if cfg["pipeline"]["skip_test"] or cfg["pipeline"]["skip_export_logits"]:
        raise ValueError("Ablation aggregation requires test results and logits; skip flags must be false")
    if not args.betas or len(set(args.betas)) != len(args.betas):
        raise ValueError("Beta tags must be nonempty and unique")
    if any(not re.fullmatch(r"0[0-9]|inf", b) for b in args.betas):
        raise ValueError("Beta tags must be 00 ... 09 (03 means 0.3) or inf")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("Seeds must be unique nonnegative integers")
    ds = cfg["dataset"]
    raw_dir = Path(cfg["data"]["raw_dir"])
    files = inventory(raw_dir, ds["adjust"])
    registration = getattr(args, "data_registration", None)
    if registration:
        if args.conversion_report:
            raise ValueError("Choose either --data-registration or --conversion-report")
        provenance, report_path = registration_provenance(files, ds, registration.resolve())
    elif args.conversion_report:
        provenance, report_path = conversion_provenance(files, ds, [args.conversion_report.resolve()])
    else:
        validate_standardized_inputs(raw_dir, files)
        provenance = {"policy": "current-snapshot", "provenance_kind": "standardized_data_snapshot",
                      "provenance_file": "data_inventory.json", "historical_preprocessing": "not_recorded"}
        report_path = None
    now = datetime.now().astimezone()
    name = f"{ds['name']}_{ds['adjust']}_{provenance['policy']}_{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    batch = args.output_base.resolve() / name
    if raw_dir == batch or raw_dir in batch.parents or batch in raw_dir.parents:
        raise ValueError("Batch output and raw input directories must be disjoint")
    if batch.exists():
        raise FileExistsError(batch)
    cfg["data"]["blacklist_path"] = str(batch / "blacklist.txt")
    # The batch prepares this file once; cells only read it. Other run controls
    # (fit/shared rebuild, workers, iterations) retain the YAML's settings.
    cfg["pipeline"]["force_regenerate_blacklist"] = False
    cfg["pipeline"]["output_root"] = str(batch)
    symbols = cfg["codebook"].get("symbol_txt_path")
    if symbols:
        cfg["codebook"]["symbol_txt_path"] = str(Path(symbols).resolve())
        if not Path(cfg["codebook"]["symbol_txt_path"]).is_file():
            raise FileNotFoundError(cfg["codebook"]["symbol_txt_path"])
    cells = []
    for beta in args.betas:
        for seed in args.seeds:
            cell = {"index": len(cells), "beta_tag": beta, "seed": seed,
                    "labeler": "pseudo" if beta == "inf" else "auto",
                    "beta_model": 0.0 if beta == "inf" else int(beta) / 10,
                    "output_root": str(batch / f"outputs_beta_model_{beta}" / f"seed{seed}")}
            compile_pipeline_config(cfg, overrides={k: cell[k] for k in
                                    ("seed", "labeler", "beta_model", "output_root")}).prepare(dry_run=True)
            cells.append(cell)
    manifest = {"schema_version": 1, "created_at": now.isoformat(), "batch_dir": str(batch),
                "project_root": str(REPO), "dataset": ds,
                "iterations": list(range(cfg["pipeline"]["end_iter"] + 1)),
                "betas": args.betas, "seeds": args.seeds, "cells": cells,
                "data": {"raw_dir": str(raw_dir), "files": files,
                         "date_min": min(f["date_min"] for f in files),
                         "date_max": max(f["date_max"] for f in files), **provenance},
                "code_sha256": code_inventory()}
    return manifest, cfg, report_path, compiled.source_text


def verify_inputs(manifest: dict) -> None:
    raw = Path(manifest["data"]["raw_dir"])
    actual = {p.name: sha256(p) for p in raw.glob(f"*_{manifest['dataset']['adjust']}.csv")}
    expected = {f["name"]: f["sha256"] for f in manifest["data"]["files"]}
    if actual != expected:
        raise ValueError("Input CSVs changed since batch preparation; use a new batch")


def verify_code(manifest: dict) -> None:
    if manifest.get("kind", "transformer") != "transformer":
        raise ValueError("Expected a Transformer batch")
    if manifest["project_root"] != str(REPO) or code_inventory() != manifest["code_sha256"]:
        raise ValueError("Project code changed since preparation; use the recorded version or a new batch")


def prepare_batch(manifest: dict, cfg: dict, report_path: Path | None, source: str) -> Path:
    # Reuse exactly the existing quality rules, with an isolated output path.
    from data.check_price_quality import check_price_quality, save_blacklist, save_report_json

    fit = cfg["codebook"]["fit"]
    blacklist, quality = check_price_quality(Path(cfg["data"]["raw_dir"]), cfg["dataset"]["adjust"],
                                             fit["end_date"], min_rows=fit["min_rows"])
    verify_inputs(manifest)
    if report_path is not None:
        if sha256(report_path) != manifest["data"]["report_sha256"]:
            raise ValueError("Data provenance record changed during batch preparation")
        provenance_bytes = report_path.read_bytes()
    else:
        provenance_bytes = (json.dumps({
            "schema_version": 1, "kind": "standardized_data_snapshot",
            "created_at": manifest["created_at"], "dataset": manifest["dataset"],
            "raw_dir": manifest["data"]["raw_dir"], "files": manifest["data"]["files"],
            "historical_preprocessing": "not_recorded", "snapshot_transforms": [],
            "limitations": "Current file identity only; no historical source or preprocessing certification.",
        }, indent=2) + "\n").encode("utf-8")
    batch = Path(manifest["batch_dir"])
    batch.mkdir(parents=True, exist_ok=False)
    (batch / "logs").mkdir()
    freeze(batch / "source.yaml", source)
    provenance_path = batch / manifest["data"].get("provenance_file", "conversion_report.json")
    freeze(provenance_path, provenance_bytes)
    manifest["data"]["report_sha256"] = sha256(provenance_path)
    symbols = cfg["codebook"].get("symbol_txt_path")
    if symbols:
        freeze(batch / "codebook_symbols.txt", Path(symbols).read_bytes())
        cfg["codebook"]["symbol_txt_path"] = str(batch / "codebook_symbols.txt")
    freeze(batch / "pipeline.yaml", yaml.safe_dump(cfg, sort_keys=False))
    blacklist_path = batch / "blacklist.txt"
    save_blacklist(blacklist, quality, blacklist_path, fit_end_date=fit["end_date"], min_rows=fit["min_rows"])
    save_report_json(quality, batch / "data_quality_report.json")
    blacklist_path.chmod(0o444)
    (batch / "data_quality_report.json").chmod(0o444)
    manifest["symbols_after_blacklist"] = [f["symbol"] for f in manifest["data"]["files"]
                                            if f["symbol"] not in set(blacklist)]
    manifest["blacklisted_symbols"] = blacklist
    for option, field in [("rev-parse", "git_commit"), ("status", "git_status")]:
        cmd = ["git", option, "HEAD"] if option == "rev-parse" else ["git", "status", "--short"]
        result = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=True)
        manifest[field] = result.stdout.strip()
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=REPO, capture_output=True, check=True)
    freeze(batch / "code.diff", diff.stdout)
    # Save source bytes as well: dirty/untracked local implementations can then
    # be reconstructed even when a commit hash alone would be insufficient.
    for name, digest in manifest["code_sha256"].items():
        if sha256(REPO / name) != digest:
            raise ValueError(f"Code changed during preparation: {name}")
        freeze(batch / "code_snapshot" / name, (REPO / name).read_bytes())
    manifest["artifacts_sha256"] = {str(p.relative_to(batch)): sha256(p)
                                     for p in sorted(batch.rglob("*")) if p.is_file()}
    path = batch / "manifest.yaml"
    freeze(path, yaml.safe_dump(manifest, sort_keys=False))
    freeze(path.with_suffix(".sha256"), sha256(path) + "\n")
    return path


def prepare_local(args: argparse.Namespace) -> Path | None:
    manifest, cfg, report_path, source = make_plan(args)
    batch = Path(manifest["batch_dir"])
    print(f"[INFO] Batch: {batch}", flush=True)
    print(f"[INFO] Verified {len(manifest['data']['files'])} CSVs; policy={manifest['data']['policy']}", flush=True)
    print("[INFO] Blacklist: generate once during preparation; cells read batch/blacklist.txt", flush=True)
    for cell in manifest["cells"]:
        print(f"[CELL {cell['index']}] {shlex.join(cell_command(manifest, cell))}", flush=True)
    if args.dry_run:
        print("[DRY RUN] No files written, no blacklist generated, no tasks run.")
        return
    path = prepare_batch(manifest, cfg, report_path, source)
    print(f"[INFO] Prepared: {path}")
    if getattr(args, "action", None) != "start":
        print("[INFO] Preparation only; no training started.")
        print(f"[INFO] Run: {shlex.join([sys.executable, str(Path(__file__).resolve()), 'run', '--batch-dir', str(batch)])}")
    return path


def run_batch(args: argparse.Namespace) -> None:
    path = args.batch_dir.resolve() / "manifest.yaml"
    manifest = load_manifest(path)
    verify_code(manifest)
    verify_inputs(manifest)
    run_local_batch(manifest, sha256(path), cell_command, run_cell, dry_run=args.dry_run)


def run_cell(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest.resolve(), args.manifest_sha256)
    if not 0 <= args.index < len(manifest["cells"]):
        raise ValueError(f"Cell index out of range: {args.index}")
    verify_code(manifest)
    verify_inputs(manifest)
    cell = manifest["cells"][args.index]
    cmd = cell_command(manifest, cell)
    execute_local_cell(manifest, cell, args.manifest_sha256, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", required=True, type=Path)
    provenance = p.add_mutually_exclusive_group()
    provenance.add_argument("--conversion-report", type=Path, help="Optional conversion report; must match every input CSV and its policy")
    provenance.add_argument("--data-registration", type=Path, help="Optional historical-data registration; must match the exact input inventory")
    p.epilog = "Without a provenance option, prepare records the current CSV inventory automatically; historical preprocessing remains unknown."
    p.add_argument("--output-base", type=Path, default=Path("outputs/ablation_runs"))
    p.add_argument("--betas", nargs="+", default=DEFAULT_BETAS, help="Tags: 01 ... 08, inf")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("start", parents=[p], epilog=p.epilog, help="Prepare a new batch and immediately train/evaluate it")
    sub.add_parser("prepare", parents=[p], epilog=p.epilog, help="Validate and freeze a new batch without training")
    p = sub.add_parser("run", help="Run pending cells sequentially; never overwrite partial results")
    p.add_argument("--batch-dir", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("cell")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    try:
        {"start": lambda args: prepare_and_run(args, prepare_local, run_batch),
         "prepare": prepare_local, "run": run_batch, "cell": run_cell}[args.action](args)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        sys.exit(exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1)


if __name__ == "__main__":
    main()
