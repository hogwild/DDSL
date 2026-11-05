from __future__ import annotations

from pathlib import Path
from typing import Any


def get_nested(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_dataset_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize dataset config.

    Preferred schema:
        dataset:
          name: SSE50
          market: cn
          adjust: hfq
    """
    ds = cfg.get("dataset", {})

    if isinstance(ds, dict):
        out = dict(ds)

        if not out.get("name"):
            name = (
                cfg.get("dataset_name")
                or get_nested(cfg, ["data", "dataset"])
                or get_nested(cfg, ["experiment", "dataset"])
                or get_nested(cfg, ["runtime", "dataset"])
            )
            if name:
                out["name"] = str(name)

        out.setdefault("market", cfg.get("market", "cn"))
        out.setdefault("adjust", cfg.get("adjust", "hfq"))
        return out

    # legacy: dataset: SSE50
    if isinstance(ds, str):
        return {
            "name": ds,
            "market": cfg.get("market", "cn"),
            "adjust": cfg.get("adjust", "hfq"),
        }

    return {
        "name": str(cfg.get("dataset_name", "UNKNOWN")),
        "market": cfg.get("market", "cn"),
        "adjust": cfg.get("adjust", "hfq"),
    }


def infer_dataset_name(
    cfg: dict[str, Any],
    config_path: str | None = None,
    cli_dataset: str | None = None,
) -> str:
    if cli_dataset:
        return cli_dataset

    name = get_dataset_cfg(cfg).get("name")
    if name and name != "UNKNOWN":
        return str(name)

    if config_path:
        return Path(config_path).stem

    raise ValueError("Cannot infer dataset name from config.")


def infer_adjust(
    cfg: dict[str, Any],
    cli_adjust: str | None = None,
    default: str = "hfq",
) -> str:
    if cli_adjust:
        return cli_adjust

    return str(get_dataset_cfg(cfg).get("adjust", default))


def infer_market(
    cfg: dict[str, Any],
    default: str = "cn",
) -> str:
    return str(get_dataset_cfg(cfg).get("market", default))


def ensure_dataset_schema(
    cfg: dict[str, Any],
    *,
    dataset_name: str,
    market: str,
    adjust: str,
) -> None:
    """
    Ensure cfg['dataset'] is a dict and contains name/market/adjust.
    Mutates cfg in place.
    """
    ds = cfg.get("dataset", {})

    if not isinstance(ds, dict):
        ds = {}

    ds.setdefault("name", dataset_name)
    ds.setdefault("market", market)
    ds.setdefault("adjust", adjust)

    cfg["dataset"] = ds


def apply_runtime_dataset_info(
    cfg: dict[str, Any],
    *,
    dataset_name: str,
    market: str,
    adjust: str,
) -> None:
    """
    Write normalized dataset info into both dataset and runtime.
    Mutates cfg in place.
    """
    ensure_dataset_schema(
        cfg,
        dataset_name=dataset_name,
        market=market,
        adjust=adjust,
    )

    cfg.setdefault("runtime", {})
    cfg["runtime"].update(
        {
            "dataset": dataset_name,
            "market": market,
            "adjust": adjust,
        }
    )