from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def build_out_dir(cfg: dict[str, Any]) -> Path:
    explicit = cfg.get("output", {}).get("out_dir")
    if explicit not in (None, "", "null"):
        out_dir = Path(explicit)
    else:
        dataset = cfg["dataset"]["name"]
        label_source = cfg["dataset"].get("label_source", "baseline")
        seed = cfg["training"]["seed"]
        token_tag = cfg["model"].get("token_tag", "kqgm")
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        out_dir = (
            Path(cfg.get("output", {}).get("root", "experiments/runs"))
            / dataset
            / label_source
            / "decoder_1head"
            / token_tag
            / f"seed{seed}_job{job_id}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir