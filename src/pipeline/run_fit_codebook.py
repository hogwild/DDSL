from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any


# Allow running from repository root without installing the package.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from data.fit_codebook_KQG_persymbolM import (
    fit_codebook_from_config,
    load_yaml_config,
)

from common.config_utils import (
    infer_dataset_name,
    infer_adjust,
    infer_market,
    apply_runtime_dataset_info,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline entrypoint for fitting K/Q/G/M codebooks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config",
        required=True,
        help="Base YAML config for codebook fitting.",
    )
    p.add_argument("--dataset", default=None)
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--raw_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--adjust", default=None)

    p.add_argument("--start_date", default=None)
    p.add_argument("--end_date", default=None)
    p.add_argument("--max_files", type=int, default=None)

    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def resolve_out_dir(
    *,
    cfg: dict[str, Any],
    dataset_name: str,
    output_root: str,
    out_dir_arg: str | None,
) -> Path:
    if out_dir_arg:
        return Path(out_dir_arg)

    return Path(output_root) / dataset_name / "codebook"


def codebook_ready(out_dir: Path) -> bool:
    """
    Current fit_codebook_KQG_persymbolM.py writes:
        geom_kmeans.joblib
        bins.joblib
        fit_config.json
        symbol_codebook_info.json
    """
    expected = [
        out_dir / "geom_kmeans.joblib",
        out_dir / "bins.joblib",
        out_dir / "fit_config.json",
        out_dir / "symbol_codebook_info.json",  # Added
    ]

    return all(p.exists() and p.stat().st_size > 0 for p in expected)


def apply_pipeline_overrides(
    cfg: dict[str, Any],
    *,
    config_path: str,
    dataset: str | None,
    output_root: str,
    raw_dir: str | None,
    out_dir: str | None,
    adjust: str | None,
    start_date: str | None,
    end_date: str | None,
    max_files: int | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    dataset_name = infer_dataset_name(
        cfg,
        config_path=config_path,
        cli_dataset=dataset,
    )
    adjust_name = infer_adjust(
        cfg,
        cli_adjust=adjust,
    )
    market_name = infer_market(cfg)

    resolved_out_dir = resolve_out_dir(
        cfg=cfg,
        dataset_name=dataset_name,
        output_root=output_root,
        out_dir_arg=out_dir,
    )

    apply_runtime_dataset_info(
        cfg,
        dataset_name=dataset_name,
        market=market_name,
        adjust=adjust_name,
    )

    cfg["runtime"].update(
        {
            "output_root": str(output_root),
            "codebook_dir": str(resolved_out_dir),
        }
    )

    cfg.setdefault("data", {})
    cfg.setdefault("fit", {})
    cfg.setdefault("output", {})
    cfg.setdefault("debug", {})

    # Sync normalized dataset info into data section for fit_codebook_KQG_persymbolM.py
    cfg["data"]["adjust"] = adjust_name
    cfg["data"]["market"] = market_name

    if raw_dir is not None:
        cfg["data"]["raw_dir"] = raw_dir

    if start_date is not None:
        cfg["fit"]["start_date"] = start_date

    if end_date is not None:
        cfg["fit"]["end_date"] = end_date

    if max_files is not None:
        cfg["debug"]["max_files"] = max_files

    cfg["output"]["out_dir"] = str(resolved_out_dir)

    return cfg


def print_config_summary(cfg: dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {})
    fit_cfg = cfg.get("fit", {})
    output_cfg = cfg.get("output", {})
    token_cfg = cfg.get("tokens", {})
    debug_cfg = cfg.get("debug", {})
    runtime_cfg = cfg.get("runtime", {})

    print("\nFit codebook pipeline config")
    print("-" * 80)
    print(f"dataset    : {runtime_cfg.get('dataset')}")
    print(f"market     : {runtime_cfg.get('market', data_cfg.get('market'))}")
    print(f"raw_dir    : {data_cfg.get('raw_dir')}")
    print(f"adjust     : {data_cfg.get('adjust')}")
    print(f"start_date : {fit_cfg.get('start_date')}")
    print(f"end_date   : {fit_cfg.get('end_date')}")
    print(f"out_dir    : {output_cfg.get('out_dir')}")
    print(f"max_files  : {debug_cfg.get('max_files')}")
    print(
        "tokens     : "
        f"K={token_cfg.get('K', {}).get('k_geom')}, "
        f"Q={token_cfg.get('Q', {}).get('bins')}, "
        f"G={token_cfg.get('G', {}).get('bins')}, "
        f"M={token_cfg.get('M', {}).get('bins')}"
    )
    print("-" * 80)


def main() -> None:
    args = parse_args()

    cfg = load_yaml_config(args.config)

    cfg = apply_pipeline_overrides(
        cfg,
        config_path=args.config,
        dataset=args.dataset,
        output_root=args.output_root,
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        adjust=args.adjust,
        start_date=args.start_date,
        end_date=args.end_date,
        max_files=args.max_files,
    )

    print_config_summary(cfg)

    out_dir = Path(cfg["output"]["out_dir"])

    if codebook_ready(out_dir) and not args.force:
        print(f"[INFO] Codebook already exists. Skip fitting: {out_dir}")
        print("[INFO] Use --force to refit.")
        return

    if args.dry_run:
        print("[DRY RUN] Would call fit_codebook_from_config(cfg).")
        return

    fit_codebook_from_config(cfg)

    if not codebook_ready(out_dir):
        raise RuntimeError(
            "Codebook fitting finished, but expected artifacts were not found:\n"
            f"  {out_dir / 'geom_kmeans.joblib'}\n"
            f"  {out_dir / 'bins.joblib'}\n"
            f"  {out_dir / 'fit_config.json'}"
        )

    print("\nCodebook fitting finished.")


if __name__ == "__main__":
    main()