"""Configuration and input provenance for the GRU learnability probe.

No labeling, tokenizer fitting, model imports or changes to experimental data.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

import yaml

from pipeline.batch_common import freeze, sha256
from pipeline.pipeline_config import compile_gru_training, load_pipeline_config


def label_signature(cfg: dict) -> dict:
    labeling = copy.deepcopy(cfg["labeling"])
    # Beta differs between ablation cells; the source cell is recorded explicitly.
    labeling.get("ddsl", {}).pop("beta_model", None)
    return {"dataset": cfg["dataset"], "tokens": cfg["codebook"]["tokens"],
            "splits": cfg["splits"], "labeling": labeling}


def input_inventory(root: Path, cfg: dict) -> list[dict]:
    dataset = cfg["dataset"]["name"]
    adjust = cfg["dataset"]["adjust"]
    records = []
    source_config = root / dataset / "pipeline_config/resolved.yaml"
    if source_config.is_file():
        source = load_pipeline_config(str(source_config)).resolved
        if label_signature(source) != label_signature(cfg):
            raise ValueError(f"GRU shared dataset/token/label settings disagree with source: {source_config}")
        records.append({"path": str(source_config.resolve()), "sha256": sha256(source_config)})
    for iteration in range(cfg["pipeline"]["start_iter"], cfg["pipeline"]["end_iter"] + 1):
        for split in ("train", "val", "test"):
            path = root / dataset / f"iter_{iteration}/dataset/{split}/ALL_tokens_{adjust}.parquet"
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f"GRU requires an existing nonempty DDSL dataset: {path}")
            records.append({"path": str(path.resolve()), "sha256": sha256(path)})
    return records


def verify_inventory(records: list[dict]) -> None:
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"GRU source artifact changed: {path}")


def check_outputs(cfg: dict) -> None:
    root = Path(cfg["pipeline"]["output_root"]) / cfg["dataset"]["name"]
    for iteration in range(cfg["pipeline"]["start_iter"], cfg["pipeline"]["end_iter"] + 1):
        for kind in ("model_gru", "eval_gru"):
            path = root / f"iter_{iteration}" / kind / f"seed_{cfg['pipeline']['seed']}"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite GRU artifacts: {path}")


@dataclass
class GRURunConfig:
    source_text: str
    resolved: dict
    input_root: Path
    train: dict

    @property
    def config_dir(self) -> Path:
        run = self.resolved["pipeline"]
        return (Path(run["output_root"]) / self.resolved["dataset"]["name"] /
                "gru_pipeline_config" / f"seed_{run['seed']}" /
                f"iter_{run['start_iter']}_{run['end_iter']}")

    @property
    def train_path(self) -> Path:
        return self.config_dir / "train_gru.yaml"

    def prepare(self, dry_run: bool = False) -> list[dict]:
        check_outputs(self.resolved)
        if self.config_dir.exists():
            raise FileExistsError(f"GRU configuration already exists: {self.config_dir}")
        records = input_inventory(self.input_root, self.resolved)
        if dry_run:
            return records
        self.config_dir.mkdir(parents=True, exist_ok=False)
        freeze(self.config_dir / "source.yaml", self.source_text)
        freeze(self.config_dir / "resolved.yaml", yaml.safe_dump(self.resolved, sort_keys=False))
        freeze(self.train_path, yaml.safe_dump(self.train, sort_keys=False))
        freeze(self.config_dir / "inputs.json", json.dumps({"input_root": str(self.input_root),
               "files": records}, indent=2))
        return records


def load_gru_config(path: str, overrides: dict | None = None, input_root: str | None = None) -> GRURunConfig:
    compiled = load_pipeline_config(path, overrides)
    train = compile_gru_training(compiled.resolved)
    return GRURunConfig(compiled.source_text, compiled.resolved,
                        Path(input_root or compiled.run["output_root"]).resolve(), train)
