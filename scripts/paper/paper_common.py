"""Read-only batch readers and provenance for paper reports.

Kept outside src/: adding report tools must not invalidate queued training batches.
Training-code equality is deliberately NOT required to read completed artifacts.
Frozen manifest/config integrity and the exact consumed result hashes are recorded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from report_statistics import EXPECTED_STATES, shannon_entropy_bits, mean_std

np.seterr(all="raise")

ITERATIONS = list(range(6))
SCOPE = "existing_test_prefix"
METRICS = ["transformer_f1", "gru_f1", "segment_fisher"]


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value, name, low=None, high=None):
    value = float(value)
    if not np.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def integer(value, name):
    number = finite(value, name)
    if number != int(number):
        raise ValueError(f"Expected integer {name}: {value}")
    return int(number)


def require_keys(obj, allowed, required, name):
    if not isinstance(obj, dict) or set(obj) - set(allowed) or set(required) - set(obj):
        raise ValueError(f"Invalid keys in {name}; allowed={allowed}, required={required}")


def one_row(frame, where, source):
    missing = set(where) - set(frame)
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")
    selected = frame
    for key, value in where.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise ValueError(f"{source}: expected one row {where}, found {len(selected)}")
    return selected.iloc[0]


class Report:
    def __init__(self, config):
        self.inputs = {}
        self.batches = {}
        self.config_path = Path(config).resolve()
        self.cfg = self.yaml(self.config_path)
        require_keys(self.cfg, ["schema_version", "seeds", "main_beta", "ablation_betas", "label_scope", "datasets"],
                     ["schema_version", "seeds", "main_beta", "ablation_betas", "label_scope", "datasets"], "report config")
        if self.cfg["schema_version"] != 1 or self.cfg["label_scope"] != SCOPE:
            raise ValueError(f"Require schema_version=1, label_scope={SCOPE}")
        self.seeds = self.cfg["seeds"]
        if (not isinstance(self.seeds, list) or len(self.seeds) < 2 or
                any(type(s) is not int or s < 0 for s in self.seeds) or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError("Use at least two unique integer seeds for sample std")
        self.beta = self.cfg["main_beta"]
        if self.beta != "03":
            raise ValueError('This paper uses main_beta: "03"')
        betas = self.cfg["ablation_betas"]
        if (not isinstance(betas, list) or not betas or len(set(betas)) != len(betas) or
                any(b not in [f"0{i}" for i in range(1, 9)] + ["inf"] for b in betas)):
            raise ValueError('ablation_betas must be unique quoted tags "01".."08", "inf"')
        if not isinstance(self.cfg["datasets"], dict) or not self.cfg["datasets"]:
            raise ValueError("datasets must be a nonempty mapping")
        for ds, spec in self.cfg["datasets"].items():
            require_keys(spec, ["transformer_batch", "gru_batch", "adjust", "ablation"],
                         ["transformer_batch", "gru_batch", "adjust", "ablation"], ds)
            if type(spec["ablation"]) is not bool or spec["adjust"] not in {"qfq", "hfq"}:
                raise ValueError(f"{ds}: specify adjust and boolean ablation explicitly")

    def path(self, value):
        path = Path(value).expanduser()
        return (self.config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()

    def track(self, path):
        path = Path(path).resolve()
        digest = sha256(path)
        if str(path) in self.inputs and self.inputs[str(path)] != digest:
            raise ValueError(f"Input changed while reading: {path}")
        self.inputs[str(path)] = digest
        return path

    def json(self, path):
        return json.loads(self.track(path).read_text())

    def yaml(self, path):
        return yaml.load(self.track(path).read_text(), Loader=UniqueLoader)

    def csv(self, path):
        return pd.read_csv(self.track(path))

    def batch(self, path, kind):
        path = Path(path).resolve()
        if path in self.batches:
            item = self.batches[path]
            if item["manifest"].get("kind", "transformer") != kind:
                raise ValueError(f"Wrong batch kind: {path}")
            return item
        mp = path / "manifest.yaml"
        m = self.yaml(mp)
        expected = self.track(path / "manifest.sha256").read_text().strip()
        if self.inputs[str(mp)] != expected:
            raise ValueError(f"Manifest checksum mismatch: {mp}")
        if m.get("schema_version") != 1 or m.get("kind", "transformer") != kind or Path(m["batch_dir"]).resolve() != path:
            raise ValueError(f"Wrong kind/schema or relocated batch: {path}")
        if not m.get("artifacts_sha256") or "pipeline.yaml" not in m["artifacts_sha256"]:
            raise ValueError(f"Missing frozen artifacts: {path}")
        for name, digest in m["artifacts_sha256"].items():
            artifact = (path / name).resolve()
            if not artifact.is_relative_to(path) or artifact == path:
                raise ValueError(f"Invalid artifact path: {name}")
            self.track(artifact)
            if self.inputs[str(artifact)] != digest:
                raise ValueError(f"Frozen artifact checksum mismatch: {artifact}")
        cells = {(c["beta_tag"], c["seed"]): c for c in m["cells"]}
        if len(cells) != len(m["cells"]) or len({c["index"] for c in m["cells"]}) != len(cells):
            raise ValueError(f"Duplicate cells: {path}")
        cfg = self.yaml(path / "pipeline.yaml")
        if cfg["dataset"] != m["dataset"]:
            raise ValueError(f"Config/manifest dataset mismatch: {path}")
        item = {"manifest": m, "hash": expected, "path": path, "cfg": cfg, "cells": cells}
        self.batches[path] = item
        return item

    def pair(self, ds, need_gru=True):
        spec = self.cfg["datasets"][ds]
        tf = self.batch(self.path(spec["transformer_batch"]), "transformer")
        if tf["manifest"]["dataset"]["name"] != ds or tf["manifest"]["dataset"]["adjust"] != spec["adjust"]:
            raise ValueError(f"{ds}: selected dataset/adjust differs from manifest")
        if not set(ITERATIONS).issubset(tf["manifest"]["iterations"]):
            raise ValueError(f"{ds}: need Transformer iterations 0..5")
        if not need_gru:
            return tf, None
        gru = self.batch(self.path(spec["gru_batch"]), "gru_probe")
        gm = gru["manifest"]
        if (gm["dataset"] != tf["manifest"]["dataset"] or gm["source_manifest_sha256"] != tf["hash"] or
                Path(gm["source_manifest"]).resolve() != tf["path"] / "manifest.yaml" or
                not set(ITERATIONS).issubset(gm["iterations"])):
            raise ValueError(f"{ds}: GRU does not link to this Transformer batch/iterations")
        return tf, gru

    def cell(self, batch, beta, seed):
        if (beta, seed) not in batch["cells"]:
            raise ValueError(f"Missing cell beta={beta}, seed={seed} in {batch['path']}")
        cell = batch["cells"][(beta, seed)]
        marker = self.json(batch["path"] / "completed" / f"{cell['index']}.json")
        if marker.get("manifest_sha256") != batch["hash"] or marker.get("cell") != cell:
            raise ValueError(f"Incomplete or mismatched completion marker: {batch['path']}, {beta}, {seed}")
        root = Path(cell["output_root"]).resolve()
        if not root.is_relative_to(batch["path"]) or root == batch["path"]:
            raise ValueError(f"Cell output outside batch: {root}")
        if batch["manifest"].get("kind", "transformer") == "transformer":
            expected = 0.0 if beta == "inf" else int(beta) / 10
            if cell["beta_model"] != expected or cell["labeler"] != ("pseudo" if beta == "inf" else "auto"):
                raise ValueError(f"Cell labeler/beta mismatch: {cell}")
        return cell

    def identity(self, ds, tf, gru=None):
        return {"dataset": ds, "adjust": tf["manifest"]["dataset"]["adjust"],
                "data_policy": tf["manifest"]["data"]["policy"],
                "transformer_batch": str(tf["path"]), "transformer_manifest_sha256": tf["hash"],
                "gru_batch": str(gru["path"]) if gru else "",
                "gru_manifest_sha256": gru["hash"] if gru else ""}

    def label_coverage(self, ds, tf, seed, iteration):
        cell = self.cell(tf, self.beta, seed)
        root = Path(cell["output_root"]) / ds / f"iter_{iteration}/labels/test"
        adjust = tf["manifest"]["dataset"]["adjust"]
        paths = sorted(root.glob(f"*_{adjust}.csv"))
        if not paths or len({p.name.split("_", 1)[0] for p in paths}) != len(paths):
            raise ValueError(f"Missing labels or multiple files per symbol: {root}")
        date_col = tf["cfg"]["labeling"]["columns"].get("date_col", "date")
        count, starts, ends = 0, [], []
        for path in paths:
            dates = pd.to_datetime(pd.read_csv(self.track(path), usecols=[date_col])[date_col], errors="raise")
            if dates.empty or dates.isna().any() or dates.duplicated().any():
                raise ValueError(f"Empty or invalid/duplicate dates: {path}")
            count += len(dates)
            starts.append(dates.min())
            ends.append(dates.max())
        return {**self.identity(ds, tf), "seed": seed, "iteration": iteration, "label_scope": SCOPE,
                "label_date_min": min(starts).date().isoformat(), "label_date_max": max(ends).date().isoformat(),
                "label_rows": count, "label_symbols": len(paths), "label_dir": str(root)}

    def observation(self, ds, tf, gru, beta, seed, iteration, states=False):
        tc, gc = self.cell(tf, beta, seed), self.cell(gru, beta, seed)
        if Path(gc["input_root"]).resolve() != Path(tc["output_root"]).resolve():
            raise ValueError("GRU cell input_root differs from Transformer cell")
        tdir = Path(tc["output_root"]) / ds / f"iter_{iteration}"
        gdir = Path(gc["output_root"]) / ds / f"iter_{iteration}"
        tj = self.json(tdir / "eval" / f"seed_{seed}" / "test_metrics.json")
        gj = self.json(gdir / "eval_gru" / f"seed_{seed}" / "test_metrics.json")
        source = tdir / "label_analysis/single_iteration/single_iteration_test_summary.json"
        labels = self.json(source)
        if labels.get("dataset") != ds or labels.get("iteration") != iteration or labels.get("split") != "test":
            raise ValueError(f"Label analysis identity mismatch: {source}")
        metrics = labels["group_metrics"]["all"]
        row = {**self.identity(ds, tf, gru), "seed": seed, "iteration": iteration, "beta_tag": beta,
               "labeler_kind": "pseudo" if beta == "inf" else "ddsl", "label_scope": SCOPE,
               "transformer_f1": finite(tj["joint_macro_f1"], "Transformer F1", 0, 1),
               "gru_f1": finite(gj["joint_macro_f1"], "GRU F1", 0, 1),
               "segment_fisher": finite(metrics["segment_level_fisher_ratio"]["macro_fisher"], "Fisher", 0)}
        row["transformer_minus_gru"] = row["transformer_f1"] - row["gru_f1"]
        if states:
            dist = metrics["state_distribution"]
            if set(dist) != set(EXPECTED_STATES):
                raise ValueError(f"Expected exactly nine state shares: {source}")
            p = np.array([finite(dist[s], s, 0, 1) for s in EXPECTED_STATES])
            if not np.isclose(p.sum(), 1, rtol=0, atol=1e-6):
                raise ValueError(f"State shares do not sum to 1: {source}")
            row.update({f"state_dist.{s}": dist[s] for s in EXPECTED_STATES})
            row.update(entropy_norm=shannon_entropy_bits(p) / np.log2(9),
                       min_state_share=float(p.min()), n_zero_states=int((p == 0).sum()))
        return row

    def verify_inputs(self):
        for path, digest in self.inputs.items():
            if sha256(path) != digest:
                raise ValueError(f"Input changed during report: {path}")

    def check_output(self, output):
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError(f"Output already exists; choose a new directory: {output}")
        for batch in self.batches.values():
            for protected in [batch["path"], Path(batch["manifest"]["data"]["raw_dir"]).resolve()]:
                if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                    raise ValueError(f"Output must be separate from source batch/raw data: {output}")
        return output

    def metadata(self, command):
        return {"schema_version": 1, "command": command, "created_at": datetime.now(timezone.utc).isoformat(),
                "config": self.cfg, "config_path": str(self.config_path), "label_scope": SCOPE,
                "aggregation": "paper mean/std tables: equal weight per seed; sample std ddof=1; no event pooling",
                "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "pyyaml": yaml.__version__},
                "inputs_sha256": self.inputs,
                "report_code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "notes": ["Existing labels/test statistics cover the historical prefix, not only 2023–2025.",
                          "Numerical reproduction requires matching source data, universe and adjustment.",
                          "Declared datasets/seeds/betas only; no automatic latest-batch discovery."]}

    def write(self, output, tables, command, dry_run=False, extra=None):
        output = self.check_output(output)
        self.verify_inputs()
        destinations = {}
        for name, value in tables.items():
            path = (output / name).resolve()
            if (Path(name).is_absolute() or not path.is_relative_to(output) or path == output or
                    path in destinations.values()):
                raise ValueError(f"Invalid or duplicate report output: {name}")
            if not ((path.suffix == ".csv" and isinstance(value, pd.DataFrame)) or
                    (path.suffix == ".json" and isinstance(value, dict))):
                raise ValueError(f"Expected CSV DataFrame or JSON mapping: {name}")
            destinations[name] = path
            units = f"{len(value)} rows" if isinstance(value, pd.DataFrame) else "JSON statistics"
            print(f"[{'DRY RUN' if dry_run else 'REPORT'}] {name}: {units}")
        if dry_run:
            return
        output.mkdir(parents=True, exist_ok=False)
        for name, value in tables.items():
            path = destinations[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(value, pd.DataFrame):
                value.to_csv(path, index=False, float_format="%.12g")
            else:
                path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        metadata = self.metadata(command)
        metadata.update(extra or {})
        metadata["outputs_sha256"] = {name: sha256(output / name) for name in tables}
        (output / "report.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        self.verify_inputs()
        (output / "COMPLETE").write_text(sha256(output / "report.json") + "\n")
        print(f"[OK] {output}")


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true", help="Validate and compute tables in memory; write nothing")
    return p
