"""Shared batch provenance, completion checks and sequential local execution.

Model-specific configuration, command construction and input checks belong to
transformer_ablation_batch.py and gru_ablation_batch.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import yaml

REPO = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(content.encode("utf-8") if isinstance(content, str) else content)
    path.chmod(0o444)


def code_inventory() -> dict[str, str]:
    # Both the local batch entry points and all experiment stages live in src.
    paths = list((REPO / "src").rglob("*.py"))
    return {str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)}


def load_manifest(path: Path, expected_hash: str | None = None) -> dict:
    expected_hash = expected_hash or path.with_suffix(".sha256").read_text().strip()
    if sha256(path) != expected_hash:
        raise ValueError("Batch manifest checksum mismatch")
    manifest = yaml.safe_load(path.read_text())
    if manifest.get("schema_version") != 1 or Path(manifest["batch_dir"]) != path.resolve().parent:
        raise ValueError("Invalid batch manifest or batch has been moved")
    for name, digest in manifest["artifacts_sha256"].items():
        if sha256(path.parent / name) != digest:
            raise ValueError(f"Frozen batch artifact changed: {name}")
    return manifest


def completed_cell(manifest: dict, cell: dict, digest: str) -> bool:
    marker = Path(manifest["batch_dir"]) / "completed" / f"{cell['index']}.json"
    if not marker.exists():
        return False
    completed = json.loads(marker.read_text())
    if completed.get("manifest_sha256") != digest or completed.get("cell") != cell:
        raise ValueError(f"Completion marker does not match this batch/cell: {marker}")
    return True


def ensure_cell_available(manifest: dict, cell: dict) -> None:
    batch = Path(manifest["batch_dir"])
    for path in (Path(cell["output_root"]), batch / "logs" / f"cell_{cell['index']}.log",
                 batch / "logs" / f"cell_{cell['index']}.json"):
        if path.exists():
            raise FileExistsError(f"Incomplete cell has existing output or execution record: {path}; "
                                  "preserve it and prepare a new batch; automatic retries are disabled")


def execute_local_cell(manifest: dict, cell: dict, digest: str, cmd: list[str]) -> None:
    """Record one local execution, retaining partial logs and never retrying it."""
    if completed_cell(manifest, cell, digest):
        raise FileExistsError(f"Cell {cell['index']} already completed; refusing to rerun it")
    ensure_cell_available(manifest, cell)
    batch = Path(manifest["batch_dir"])
    log_path = batch / "logs" / f"cell_{cell['index']}.log"
    receipt = log_path.with_suffix(".json")
    record = {"manifest_sha256": digest, "cell": cell, "command": cmd,
              "started_at": datetime.now().astimezone().isoformat(),
              "log": str(log_path), "returncode": None, "status": "running"}
    print(f"[INFO] Cell {cell['index']}: {shlex.join(cmd)}", flush=True)
    print(f"[INFO] Log: {log_path}", flush=True)
    # Exclusive creation also prevents two local invocations running one cell.
    with log_path.open("x", encoding="utf-8") as output:
        try:
            result = subprocess.run(cmd, cwd=REPO, stdout=output, stderr=subprocess.STDOUT, check=True)
            record.update(returncode=result.returncode, status="complete")
        except BaseException as exc:
            record.update(returncode=getattr(exc, "returncode", None), status="failed",
                          error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["finished_at"] = datetime.now().astimezone().isoformat()
            freeze(receipt, json.dumps(record, indent=2))
    freeze(batch / "completed" / f"{cell['index']}.json", json.dumps({
        "manifest_sha256": digest, "cell": cell,
        "completed_at": datetime.now().astimezone().isoformat()}, indent=2))


def run_local_batch(manifest: dict, digest: str, command_builder, cell_runner, *, dry_run: bool) -> None:
    """Preflight all pending cells, then execute them serially without retries."""
    pending = []
    for cell in manifest["cells"]:
        if completed_cell(manifest, cell, digest):
            print(f"[SKIP] Cell {cell['index']} already completed this batch.", flush=True)
        else:
            ensure_cell_available(manifest, cell)
            pending.append(cell)
    for cell in pending:
        print(f"[CELL {cell['index']}] {shlex.join(command_builder(manifest, cell))}", flush=True)
    if dry_run:
        print(f"[DRY RUN] {len(pending)} pending cells; no files written or tasks run.")
        return
    for cell in pending:
        cell_runner(argparse.Namespace(manifest=Path(manifest["batch_dir"]) / "manifest.yaml",
                                       manifest_sha256=digest, index=cell["index"]))
    print("[INFO] All cells completed. Use scripts/run_paper_reports.sh to report completed Transformer/GRU batches.", flush=True)


def prepare_and_run(args: argparse.Namespace, prepare, run) -> None:
    """Start exactly the batch returned by preparation; never parse console output."""
    path = prepare(args)
    if args.dry_run:
        return
    if path is None:
        raise RuntimeError("Preparation did not return a batch manifest")
    print(f"[INFO] Starting training/evaluation for batch: {path.parent}", flush=True)
    run(argparse.Namespace(batch_dir=path.parent, dry_run=False))
