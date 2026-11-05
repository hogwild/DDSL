"""Compile a full-pipeline YAML into the existing standalone stage schemas.

This module only handles configuration. It never imports training/labeling code
or changes their experimental definitions. Paths retain the existing convention:
relative input paths are relative to the working directory (repository root).
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """Fail on repeated keys instead of silently keeping the last value."""


def _unique_mapping(loader: UniqueKeyLoader, node: Any, deep: bool = False) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _mapping(value: Any, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown fields in {name}: {sorted(unknown, key=str)}")
    return value


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _date(value: Any, name: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)") from exc


RUN_DEFAULTS = {
    "start_iter": 0, "end_iter": 3, "seed": 42, "n_workers": 8,
    "labeler": "auto", "output_root": "outputs",
    "skip_test": False, "skip_export_logits": False,
    "force_fit_codebook": False, "force_regenerate_blacklist": False,
    "force_rebuild_shared_logits_dataset": False,
}


@dataclass
class CompiledPipelineConfig:
    source_path: str
    source_text: str
    resolved: dict[str, Any]
    stages: dict[str, dict[str, Any]]
    output_root_generated: bool = False

    @property
    def run(self) -> dict[str, Any]:
        return self.resolved["pipeline"]

    @property
    def run_dir(self) -> Path:
        return Path(self.run["output_root"]) / self.resolved["dataset"]["name"]

    @property
    def config_dir(self) -> Path:
        return self.run_dir / "pipeline_config"

    @property
    def paths(self) -> dict[str, str]:
        return {name: str(self.config_dir / f"{name}.yaml") for name in self.stages}

    def identity(self) -> dict[str, Any]:
        # Iteration range and execution controls may change when continuing a run.
        # Dataset, stage settings, training seed and labeler must remain identical.
        return {"stages": self.stages, "seed": self.run["seed"], "labeler": self.run["labeler"]}

    def prepare(self, *, dry_run: bool = False) -> None:
        """Validate resume identity and freeze stage inputs before any subprocess."""
        if self.run["start_iter"] > 0 and self.output_root_generated:
            raise ValueError(
                "DDSL resume requires the original output_root; set --output_root or "
                "pipeline.output_root explicitly when start_iter > 0"
            )
        if self.run["start_iter"] == 0:
            if self.run_dir.exists():
                raise FileExistsError(f"Refusing to overwrite an existing run: {self.run_dir}")
        else:
            manifest = self.config_dir / "identity.json"
            if not manifest.is_file():
                raise ValueError(f"Resume requires a run created with --config: {manifest}")
            if json.loads(manifest.read_text(encoding="utf-8")) != self.identity():
                raise ValueError("Unified configuration differs from the original run; use a new output_root and iteration 0")
            for name, path in self.paths.items():
                if not Path(path).is_file() or yaml.safe_load(Path(path).read_text(encoding="utf-8")) != self.stages[name]:
                    raise ValueError(f"Frozen stage configuration is missing or modified: {path}")
            for iteration in range(self.run["start_iter"], self.run["end_iter"] + 1):
                if (self.run_dir / f"iter_{iteration}").exists():
                    raise FileExistsError(f"Refusing to overwrite iter_{iteration} in {self.run_dir}")
            for required in [self.run_dir / "codebook", self.run_dir / "shared" / "logits_dataset"]:
                if not required.is_dir():
                    raise FileNotFoundError(f"Resume dependency missing: {required}")
        if dry_run or self.run["start_iter"] > 0:
            return
        # mkdir without exist_ok also rejects a concurrently created run.
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.config_dir.mkdir()
        (self.config_dir / "source.yaml").write_text(self.source_text, encoding="utf-8")
        (self.config_dir / "resolved.yaml").write_text(yaml.safe_dump(self.resolved, sort_keys=False), encoding="utf-8")
        for name, path in self.paths.items():
            Path(path).write_text(yaml.safe_dump(self.stages[name], sort_keys=False), encoding="utf-8")
        (self.config_dir / "identity.json").write_text(json.dumps(self.identity(), indent=2), encoding="utf-8")


def load_pipeline_config(path: str, overrides: dict[str, Any] | None = None) -> CompiledPipelineConfig:
    source = Path(path).read_text(encoding="utf-8")
    try:
        cfg = yaml.load(source, Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid pipeline YAML: {exc}") from exc
    return compile_pipeline_config(cfg, source_path=path, source_text=source, overrides=overrides)


def compile_pipeline_config(
    config: dict[str, Any], *, source_path: str = "<memory>", source_text: str = "",
    overrides: dict[str, Any] | None = None,
) -> CompiledPipelineConfig:
    cfg = copy.deepcopy(config)
    _mapping(cfg, "pipeline config", {"schema_version", "dataset", "data", "pipeline", "splits", "codebook", "labeling", "build_dataset", "train", "gru"})
    if type(cfg.get("schema_version")) is not int or cfg["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    ds = _mapping(cfg.get("dataset"), "dataset", {"name", "market", "adjust"})
    markets = {"SSE50": "cn", "CSI300": "cn", "CSI500": "cn", "NASDAQ100": "us"}
    if ds.get("name") not in markets or ds.get("market") != markets[ds["name"]]:
        raise ValueError("dataset.name/market must describe one of SSE50, CSI300, CSI500 (cn), NASDAQ100 (us)")
    if ds.get("adjust") not in {"hfq", "qfq"}:
        raise ValueError("dataset.adjust must be hfq or qfq")
    data = _mapping(cfg.get("data"), "data", {"raw_dir", "blacklist_path"})
    for key in ["raw_dir", "blacklist_path"]:
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"data.{key} must be a nonempty path")
        data[key] = str(Path(data[key]).resolve())
    # The quality-check entrypoint uses this same path. Batch runs may point to
    # a frozen batch-local blacklist; standalone configs retain their old paths.

    pipeline_config = _mapping(cfg.get("pipeline", {}), "pipeline", set(RUN_DEFAULTS))
    run = {**RUN_DEFAULTS, **pipeline_config}
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    _mapping(overrides, "CLI overrides", set(RUN_DEFAULTS) | {"beta_model", "train_precision"})
    run.update({k: v for k, v in overrides.items() if k in RUN_DEFAULTS})
    for key in ["start_iter", "end_iter", "seed"]:
        _integer(run[key], f"pipeline.{key}", 0)
    _integer(run["n_workers"], "pipeline.n_workers")
    if run["end_iter"] < run["start_iter"]:
        raise ValueError("pipeline.end_iter must be >= start_iter")
    if run["labeler"] not in {"auto", "dp", "pseudo"}:
        raise ValueError("pipeline.labeler must be auto, dp, or pseudo")
    for key, default in RUN_DEFAULTS.items():
        if type(default) is bool and type(run[key]) is not bool:
            raise ValueError(f"pipeline.{key} must be a boolean")
    if run["skip_export_logits"] and run["end_iter"] > run["start_iter"]:
        raise ValueError("A multi-iteration pipeline needs exported logits; skip_export_logits must be false")
    output_root_generated = "output_root" not in pipeline_config and "output_root" not in overrides
    if not isinstance(run["output_root"], str) or not run["output_root"]:
        raise ValueError("pipeline.output_root must be a nonempty path")

    splits = _mapping(cfg.get("splits"), "splits", {"train", "val", "test"})
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("The full pipeline requires train, val and test splits")
    previous_end = None
    for name in ["train", "val", "test"]:
        split = _mapping(splits[name], f"splits.{name}", {"start_date", "end_date"})
        split["start_date"] = _date(split.get("start_date"), f"splits.{name}.start_date", nullable=name == "train")
        split["end_date"] = _date(split.get("end_date"), f"splits.{name}.end_date")
        if split["start_date"] and split["start_date"] > split["end_date"]:
            raise ValueError(f"splits.{name}: start_date is after end_date")
        if previous_end and split["start_date"] <= previous_end:
            raise ValueError(f"splits.{name} overlaps the previous split")
        previous_end = split["end_date"]

    cb = _mapping(cfg.get("codebook"), "codebook", {"fit", "tokens", "debug", "symbol_txt_path"})
    fit = _mapping(cb.setdefault("fit", {}), "codebook.fit", {"start_date", "end_date", "min_rows"})
    fit.setdefault("start_date", splits["train"]["start_date"])
    fit.setdefault("end_date", splits["train"]["end_date"])
    fit.setdefault("min_rows", 60)
    fit["start_date"] = _date(fit["start_date"], "codebook.fit.start_date", nullable=True)
    fit["end_date"] = _date(fit["end_date"], "codebook.fit.end_date")
    _integer(fit["min_rows"], "codebook.fit.min_rows")
    if fit["end_date"] > splits["train"]["end_date"] or (fit["start_date"] and fit["start_date"] > fit["end_date"]):
        raise ValueError("codebook.fit must be a valid interval ending no later than the training cutoff")
    tokens = _mapping(cb.get("tokens"), "codebook.tokens", {"K", "Q", "G", "M"})
    dimensions = {}
    for component, allowed, size_key in [
        ("K", {"k_geom"}, "k_geom"), ("Q", {"bins", "mode", "vol_window", "min_periods"}, "bins"),
        ("G", {"bins"}, "bins"), ("M", {"bins", "col"}, "bins"),
    ]:
        token = _mapping(tokens.get(component), f"codebook.tokens.{component}", allowed)
        dimensions[f"num_{component}"] = _integer(token.get(size_key), f"codebook.tokens.{component}.{size_key}")
    if tokens["Q"].get("mode", "dc") not in {"dc", "volnorm"}:
        raise ValueError("codebook.tokens.Q.mode must be dc or volnorm")
    q_window = _integer(tokens["Q"].get("vol_window", 60), "codebook.tokens.Q.vol_window")
    q_min = _integer(tokens["Q"].get("min_periods", 30), "codebook.tokens.Q.min_periods")
    if q_min > q_window:
        raise ValueError("codebook.tokens.Q.min_periods must not exceed vol_window")
    debug = _mapping(cb.get("debug", {}), "codebook.debug", {"max_files"})
    _integer(debug.get("max_files", 0), "codebook.debug.max_files", 0)

    label = _mapping(cfg.get("labeling"), "labeling", {"history_start_date", "columns", "smoothing", "windows", "scoring", "transition", "ddsl"})
    label.setdefault("history_start_date", splits["train"]["start_date"])
    history_start = _date(label["history_start_date"], "labeling.history_start_date", nullable=True)
    label["history_start_date"] = history_start
    if history_start is not None and (splits["train"]["start_date"] is None or history_start > splits["train"]["start_date"]):
        raise ValueError("labeling.history_start_date must cover the training interval start")
    for section, keys in {
        "columns": {"date_col", "price_col", "symbol_col"}, "smoothing": {"smooth_span"},
        "windows": {"current_left", "current_right", "future"},
        "scoring": {"tau", "lambda_current", "lambda_future", "w_slope", "w_flip", "w_pos", "w_flip_neutral", "w_pos_neutral"},
        "transition": {"lambda_neutral_switch", "lambda_opposite_switch", "mu_neutral_switch", "mu_opposite_switch"},
        "ddsl": {"beta_model", "model_score_kind", "model_temperature", "require_model_scores"},
    }.items():
        _mapping(label.get(section, {}), f"labeling.{section}", keys)
    ddsl = label.setdefault("ddsl", {})
    if "beta_model" in overrides:
        ddsl["beta_model"] = overrides["beta_model"]
    beta = ddsl.get("beta_model")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0:
        raise ValueError("labeling.ddsl.beta_model must be finite and >= 0 (use labeler: pseudo for the inf arm)")
    if ddsl.get("model_score_kind", "auto") not in {"auto", "prob", "logprob", "logit"}:
        raise ValueError("labeling.ddsl.model_score_kind must be auto, prob, logprob or logit")
    temperature = ddsl.get("model_temperature", 1.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("labeling.ddsl.model_temperature must be finite and positive")
    # Dataset construction expects these fixed columns. A custom price column
    # remains a labeling-only choice; date/symbol aliases cannot cross stages.
    for key, expected in [("date_col", "date"), ("symbol_col", "symbol")]:
        if label.get("columns", {}).get(key, expected) != expected:
            raise ValueError(f"labeling.columns.{key} must be {expected!r} for the current dataset builder")

    build = _mapping(cfg.get("build_dataset"), "build_dataset", {"options"})
    options = _mapping(build.setdefault("options", {}), "build_dataset.options", {"trim_tail_days", "combine", "max_files", "require_state"})
    if options.get("combine", True) is not True:
        raise ValueError("build_dataset.options.combine must be true: training requires ALL_tokens parquet files")
    for key in ["trim_tail_days", "max_files"]:
        _integer(options.get(key, 0), f"build_dataset.options.{key}", 0)
    if "require_state" in options and type(options["require_state"]) is not bool:
        raise ValueError("build_dataset.options.require_state must be a boolean")

    train = _mapping(cfg.get("train"), "train", {"model", "training", "loss"})
    model = _mapping(train.get("model"), "train.model", set(dimensions) | {"d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "max_seq_len", "pooling"})
    training = _mapping(train.get("training"), "train.training", {"seq_len", "batch_size", "num_workers", "epochs", "lr", "weight_decay", "grad_clip", "scheduler", "loss_type", "use_class_weights", "precision"})
    if "train_precision" in overrides:
        training["precision"] = overrides["train_precision"]
    if training.get("precision", "fp32") not in ("fp32", "bf16", "fp16"):
        raise ValueError("train.training.precision must be fp32, bf16 or fp16")
    # Keep an omitted FP32 default omitted so pre-AMP run identities remain valid.
    # run_train_transformer accesses cfg['loss'] even for cross_entropy; preserve its default.
    _mapping(train.setdefault("loss", {}), "train.loss", {"focal_gamma"})
    for key in ["seq_len", "batch_size", "epochs"]:
        _integer(training.get(key), f"train.training.{key}")
    _integer(training.get("num_workers", 0), "train.training.num_workers", 0)
    lr = training.get("lr")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("train.training.lr must be finite and positive")
    if model.get("pooling") not in {"last", "mean", "cls"}:
        raise ValueError("train.model.pooling must be last, mean or cls")
    for key, size in dimensions.items():
        if key in model and model[key] != size:
            raise ValueError(f"train.model.{key} conflicts with codebook.tokens ({size})")
        model[key] = size
    model.setdefault("max_seq_len", training["seq_len"])
    if _integer(model["max_seq_len"], "train.model.max_seq_len") < training["seq_len"]:
        raise ValueError("train.model.max_seq_len must be >= train.training.seq_len")
    if _integer(model.get("d_model", 64), "train.model.d_model") % _integer(model.get("nhead", 4), "train.model.nhead"):
        raise ValueError("train.model.d_model must be divisible by nhead")

    if output_root_generated:
        # Name from validated, effective parameters after applying CLI overrides.
        # Choose once without creating directories, including during dry-run.
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        name = f"{ds['name']}_seed{run['seed']}_beta{beta}_{stamp}"
        base = Path(RUN_DEFAULTS["output_root"]) / name
        candidate = base
        suffix = 2
        while candidate.exists() or candidate.is_symlink():
            candidate = base.with_name(f"{base.name}_{suffix}")
            suffix += 1
        run["output_root"] = str(candidate)
    run["output_root"] = str(Path(run["output_root"]).resolve())
    cfg["pipeline"] = run
    root = Path(run["output_root"]) / ds["name"]

    # Standalone schemas are built in memory; no base YAML is consulted here.
    shared_data = {"blacklist_path": data["blacklist_path"]}
    stages = {
        "codebook": {
            "dataset": ds, "data": {**data, "symbol_txt_path": cb.get("symbol_txt_path")},
            "fit": fit, "tokens": tokens, "debug": cb.get("debug", {}),
            "output": {"out_dir": str(root / "codebook")},
        },
        "label": {
            "dataset": ds, "data": shared_data, "labeler": {"type": run["labeler"]},
            **{k: v for k, v in label.items() if k != "history_start_date"},
            "io": {"in_dir": data["raw_dir"], "pattern": f"*_{ds['adjust']}.csv"},
            "splits": {k: {"start_date": history_start, "end_date": v["end_date"]} for k, v in splits.items()},
        },
        "build": {"dataset": ds, "data": shared_data, "tokenizer": {"dir": str(root / "codebook")}, "options": options, "splits": splits},
        "train": {"dataset": ds, **train, "training": {**training, "seed": run["seed"]}},
    }
    if "gru" in cfg:
        compile_gru_training(cfg)  # Validate the optional probe without changing Transformer stages.
    return CompiledPipelineConfig(source_path, source_text, cfg, copy.deepcopy(stages), output_root_generated)


def compile_gru_training(cfg: dict) -> dict:
    """Compile the optional GRU probe section to its existing training schema."""
    if "gru" not in cfg:
        raise ValueError("GRU execution requires a gru section in the unified pipeline YAML")
    gru = copy.deepcopy(_mapping(cfg["gru"], "gru", {"model", "training", "loss"}))
    dimensions = {"num_K": cfg["codebook"]["tokens"]["K"]["k_geom"],
                  **{f"num_{k}": cfg["codebook"]["tokens"][k]["bins"] for k in "QGM"}}
    model = _mapping(gru.get("model"), "gru.model", set(dimensions) | {
        "d_model", "hidden_size", "num_layers", "dropout", "bidirectional", "pooling"})
    for key, value in dimensions.items():
        if key in model and model[key] != value:
            raise ValueError(f"gru.model.{key} conflicts with codebook.tokens")
        model[key] = value
    for key in ("d_model", "hidden_size", "num_layers"):
        _integer(model.get(key), f"gru.model.{key}")
    if model.get("pooling") not in {"last", "mean"}:
        raise ValueError("gru.model.pooling must be last or mean")
    if type(model.get("bidirectional", False)) is not bool:
        raise ValueError("gru.model.bidirectional must be a boolean")
    dropout = model.get("dropout", 0.3)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("gru.model.dropout must be in [0, 1)")
    training = _mapping(gru.get("training"), "gru.training", {
        "seq_len", "batch_size", "num_workers", "epochs", "lr", "weight_decay", "grad_clip",
        "scheduler", "loss_type", "use_class_weights"})
    for key in ("seq_len", "batch_size", "epochs"):
        _integer(training.get(key), f"gru.training.{key}")
    _integer(training.get("num_workers", 0), "gru.training.num_workers", 0)
    lr = training.get("lr")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("gru.training.lr must be finite and positive")
    _mapping(gru.setdefault("loss", {}), "gru.loss", {"focal_gamma"})
    training["seed"] = cfg["pipeline"]["seed"]
    return {"dataset": copy.deepcopy(cfg["dataset"]), **gru}
