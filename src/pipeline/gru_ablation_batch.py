"""GRU probe batches linked to frozen Transformer ablation datasets.

The GRU never generates labels, blacklists, codebooks or model-feedback logits.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from pipeline.batch_common import freeze, sha256, load_manifest, code_inventory
from pipeline.batch_common import execute_local_cell, prepare_and_run, run_local_batch
from pipeline.gru_pipeline_config import input_inventory, verify_inventory, check_outputs, label_signature
from pipeline.pipeline_config import load_pipeline_config, compile_pipeline_config, compile_gru_training


def probe_code_inventory() -> dict:
    return code_inventory()


def command(manifest: dict, cell: dict) -> list[str]:
    return [sys.executable, "-u", str(REPO / "src/pipeline/run_gru_pipeline.py"),
            "--config", str(Path(manifest["batch_dir"]) / "pipeline.yaml"),
            "--input_root", cell["input_root"], "--output_root", cell["output_root"],
            "--seed", str(cell["seed"])]


def plan(args: argparse.Namespace) -> tuple[dict, dict, str]:
    compiled = load_pipeline_config(str(args.config), {"start_iter": args.start_iter, "end_iter": args.end_iter})
    cfg = compiled.resolved
    compile_gru_training(cfg)
    if cfg["pipeline"]["skip_test"]:
        raise ValueError("GRU ablation requires test results; skip_test must be false")
    source_path = args.source_batch.resolve() / "manifest.yaml"
    source_hash = sha256(source_path)
    source = load_manifest(source_path, source_hash)
    if source.get("kind", "transformer") != "transformer":
        raise ValueError("--source-batch must select a Transformer batch")
    source_cfg = load_pipeline_config(str(source_path.parent / "pipeline.yaml")).resolved
    if label_signature(cfg) != label_signature(source_cfg):
        raise ValueError("GRU shared dataset/token/label settings disagree with the Transformer batch")
    iterations = list(range(cfg["pipeline"]["start_iter"], cfg["pipeline"]["end_iter"] + 1))
    if not set(iterations).issubset(source["iterations"]):
        raise ValueError("GRU iteration range is not covered by the source batch")
    betas, seeds = args.betas or source["betas"], args.seeds or source["seeds"]
    if (len(set(betas)) != len(betas) or len(set(seeds)) != len(seeds)
            or not set(betas).issubset(source["betas"]) or not set(seeds).issubset(source["seeds"])):
        raise ValueError("Requested beta/seed matrix must be a unique subset of the source batch")
    ds = cfg["dataset"]
    stamp = datetime.now().astimezone()
    name = f"{ds['name']}_{ds['adjust']}_{source['data']['policy']}_gru_{stamp:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    batch = args.output_base.resolve() / name
    if batch.exists():
        raise FileExistsError(batch)
    if args.source_batch.resolve() in batch.parents or Path(cfg["data"]["raw_dir"]) in batch.parents:
        raise ValueError("GRU batch must be outside source Transformer outputs and raw data")
    cfg["pipeline"]["output_root"] = str(batch)
    cfg["pipeline"]["force_regenerate_blacklist"] = False
    cfg["data"]["blacklist_path"] = str(batch / "blacklist.txt")
    # Unused by GRU, but retain the actual source universe reference in its config.
    cfg["codebook"]["symbol_txt_path"] = source_cfg["codebook"].get("symbol_txt_path")
    cells = []
    lookup = {(c["beta_tag"], c["seed"]): c for c in source["cells"]}
    for beta in betas:
        for seed in seeds:
            original = lookup[(beta, seed)]
            marker = source_path.parent / "completed" / f"{original['index']}.json"
            completed = json.loads(marker.read_text()) if marker.is_file() else {}
            if completed.get("manifest_sha256") != source_hash or completed.get("cell") != original:
                raise ValueError(f"Transformer cell beta={beta}, seed={seed} has not completed")
            output = batch / f"outputs_beta_model_{beta}" / f"seed{seed}"
            cell_cfg = compile_pipeline_config(cfg, overrides={"seed": seed, "output_root": str(output)}).resolved
            check_outputs(cell_cfg)
            records = input_inventory(Path(original["output_root"]), cell_cfg)
            cells.append({"index": len(cells), "source_index": original["index"], "beta_tag": beta,
                          "seed": seed, "input_root": original["output_root"], "output_root": str(output),
                          "input_files": records})
    manifest = {"schema_version": 1, "kind": "gru_probe", "created_at": stamp.isoformat(),
                "batch_dir": str(batch), "project_root": str(REPO), "dataset": ds,
                "source_manifest": str(source_path), "source_manifest_sha256": source_hash,
                "data": copy.deepcopy(source["data"]), "iterations": iterations,
                "betas": betas, "seeds": seeds, "cells": cells,
                "code_sha256": probe_code_inventory()}
    return manifest, cfg, compiled.source_text


def verify_source(manifest: dict, cells: list[dict] | None = None) -> None:
    load_manifest(Path(manifest["source_manifest"]), manifest["source_manifest_sha256"])
    for cell in cells if cells is not None else manifest["cells"]:
        verify_inventory(cell["input_files"])


def prepare(manifest: dict, cfg: dict, source_text: str) -> Path:
    verify_source(manifest)
    if probe_code_inventory() != manifest["code_sha256"]:
        raise ValueError("Code changed during GRU batch preparation")
    provenance_file = manifest["data"].get("provenance_file", "conversion_report.json")
    if provenance_file not in {"conversion_report.json", "data_registration.json", "data_inventory.json"}:
        raise ValueError("Unsupported data provenance artifact")
    batch = Path(manifest["batch_dir"])
    batch.mkdir(parents=True, exist_ok=False)
    (batch / "logs").mkdir()
    freeze(batch / "source.yaml", source_text)
    source_root = Path(manifest["source_manifest"]).parent
    freeze(batch / "transformer_manifest.yaml", Path(manifest["source_manifest"]).read_bytes())
    for name in ("blacklist.txt", "data_quality_report.json", provenance_file):
        freeze(batch / name, (source_root / name).read_bytes())
    symbol_path = cfg["codebook"].get("symbol_txt_path")
    if symbol_path:
        freeze(batch / "codebook_symbols.txt", Path(symbol_path).read_bytes())
        cfg["codebook"]["symbol_txt_path"] = str(batch / "codebook_symbols.txt")
    freeze(batch / "pipeline.yaml", yaml.safe_dump(cfg, sort_keys=False))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
    manifest["git_commit"] = commit.stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=REPO, capture_output=True, text=True, check=True)
    manifest["git_status"] = status.stdout.strip()
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
    manifest, cfg, source = plan(args)
    batch = Path(manifest["batch_dir"])
    print(f"[INFO] GRU batch: {batch}", flush=True)
    print(f"[INFO] Transformer source: {manifest['source_manifest']}", flush=True)
    for cell in manifest["cells"]:
        print(f"[CELL {cell['index']}] {shlex.join(command(manifest, cell))}", flush=True)
    if args.dry_run:
        print("[DRY RUN] Verified existing DDSL datasets; no files written or tasks run.")
        return
    path = prepare(manifest, cfg, source)
    print(f"[INFO] Prepared: {path}")
    if getattr(args, "action", None) != "start":
        print("[INFO] Preparation only; no training started.")
        print(f"[INFO] Run: {shlex.join([sys.executable, str(Path(__file__).resolve()), 'run', '--batch-dir', str(batch)])}")
    return path


def checked_manifest(path: Path, digest: str | None = None) -> dict:
    manifest = load_manifest(path, digest)
    if manifest.get("kind") != "gru_probe":
        raise ValueError("Expected a GRU probe batch")
    if manifest["project_root"] != str(REPO) or manifest["code_sha256"] != probe_code_inventory():
        raise ValueError("GRU project code changed since preparation; use the recorded version or a new batch")
    return manifest


def run_batch(args: argparse.Namespace) -> None:
    path = args.batch_dir.resolve() / "manifest.yaml"
    manifest = checked_manifest(path)
    verify_source(manifest)
    run_local_batch(manifest, sha256(path), command, run_cell, dry_run=args.dry_run)


def run_cell(args: argparse.Namespace) -> None:
    manifest = checked_manifest(args.manifest.resolve(), args.manifest_sha256)
    if not 0 <= args.index < len(manifest["cells"]):
        raise ValueError("GRU cell index out of range")
    cell = manifest["cells"][args.index]
    verify_source(manifest, [cell])
    cmd = command(manifest, cell)
    execute_local_cell(manifest, cell, args.manifest_sha256, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--source-batch", required=True, type=Path)
    p.add_argument("--output-base", default=Path("outputs/ablation_runs"), type=Path)
    p.add_argument("--betas", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--start-iter", type=int)
    p.add_argument("--end-iter", type=int)
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("start", parents=[p], help="Prepare a linked GRU batch and immediately train/evaluate it")
    sub.add_parser("prepare", parents=[p], help="Validate and freeze a linked GRU batch without training")
    p = sub.add_parser("run", help="Run pending GRU cells sequentially without overwriting partial results")
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
