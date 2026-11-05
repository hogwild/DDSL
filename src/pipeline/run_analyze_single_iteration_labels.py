from __future__ import annotations

import argparse
import sys 
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running from repo root without installing package.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from analysis.io import (
    DEFAULT_CURRENT_COL,
    DEFAULT_DATE_COL,
    DEFAULT_FUTURE_COL,
    DEFAULT_STATE_COL,
    DEFAULT_SYMBOL_COL,
    distribution_dict,
    ensure_dir,
    flatten_nested_dict,
    load_label_dir,
    load_symbol_groups,
    require_columns,
    resolve_iteration_label_dir,
    resolve_single_analysis_dir,
    save_csv,
    save_json,
    value_counts_dict,
)

from analysis.metrics_single import (
    compute_boundary_consistency,
    compute_segment_level_features,
    compute_segment_stats,
    extract_segments,
    fisher_ratio,
    check_feature_cols,
    seg_fisher_feature_cols,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze labels from a single DDSL iteration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pipeline mode
    p.add_argument("--dataset", default=None)
    p.add_argument("--iteration", type=int, default=None)
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--adjust", default="hfq")
    p.add_argument("--splits", nargs="*", default=["test"])  # ["train", "val", "test"]

    # Manual mode
    p.add_argument("--label_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--split", default=None)

    # Columns
    p.add_argument("--date_col", default=DEFAULT_DATE_COL)
    p.add_argument("--symbol_col", default=DEFAULT_SYMBOL_COL)
    p.add_argument("--state_col", default=DEFAULT_STATE_COL)
    p.add_argument("--current_col", default=DEFAULT_CURRENT_COL)
    p.add_argument("--future_col", default=DEFAULT_FUTURE_COL)

    p.add_argument("--window_start", default=None,
                help="Optional date window start (inclusive), e.g. 2023-01-01. "
                     "Slices the loaded labels by date BEFORE any metric is "
                     "computed. Requires --window_name.")
    p.add_argument("--window_end", default=None,
                help="Optional date window end (inclusive).")
    p.add_argument("--window_name", default=None,
                help="Short tag for the window, e.g. trainperiod / testperiod. "
                     "Appended to the split name in ALL output paths and "
                     "summary rows, so downstream scripts can address the "
                     "windowed results as split='test_{window_name}'.")

    # File matching
    p.add_argument(
        "--pattern",
        default=None,
        help="CSV pattern. If omitted, tries '*_{adjust}.csv' first, then '*.csv'.",
    )

    # Metrics
    p.add_argument("--bc_radius", type=int, default=2)
   
    # Output control
    p.add_argument(
        "--save_detail_tables",
        action="store_true",
        help="Save boundary, segment, and segment-feature detail tables.",
    )
    # Path to codebook symbols
    p.add_argument(
        "--codebook_dir",
        default=None,
        help="Path to codebook directory containing symbol_codebook_info.json. "
             "If provided, metrics are computed separately for symbols with and without codebook.",
    )

    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def compute_metrics_for_group(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:

    feature_cols = check_feature_cols(df)

    segments = extract_segments(
        df,
        symbol_col=args.symbol_col,
        date_col=args.date_col,
        state_col=args.state_col,
    )

    bc_df, bc_summary = compute_boundary_consistency(
        df,
        symbol_col=args.symbol_col,
        date_col=args.date_col,
        state_col=args.state_col,
        radius=args.bc_radius,
    )

    row_fisher = fisher_ratio(
        df,
        label_col=args.state_col,
        feature_cols=feature_cols,
    )

    seg_feat_df = compute_segment_level_features(
        df,
        symbol_col=args.symbol_col,
        date_col=args.date_col,
        state_col=args.state_col,
        feature_cols=feature_cols,
    )

    seg_fisher_features = seg_fisher_feature_cols(seg_feat_df)

    if not seg_feat_df.empty:
        seg_fisher = fisher_ratio(
            seg_feat_df,
            label_col="state",
            feature_cols=seg_fisher_features,
        )
    else:
        seg_fisher = {
            "feature_cols": [],
            "macro_fisher": None,
            "per_feature": {},
            "note": "No segment-level features available.",
        }

    return {
        "num_rows": int(len(df)),
        "num_symbols": int(df[args.symbol_col].nunique()),
        "feature_cols": feature_cols,
        "state_counts": value_counts_dict(df[args.state_col]),
        "state_distribution": distribution_dict(df[args.state_col]),
        "current_counts": (
            value_counts_dict(df[args.current_col])
            if args.current_col in df.columns else {}
        ),
        "current_distribution": (
            distribution_dict(df[args.current_col])
            if args.current_col in df.columns else {}
        ),
        "future_counts": (
            value_counts_dict(df[args.future_col])
            if args.future_col in df.columns else {}
        ),
        "future_distribution": (
            distribution_dict(df[args.future_col])
            if args.future_col in df.columns else {}
        ),
        "boundary_consistency": bc_summary,
        "segment_stats": compute_segment_stats(segments),
        "row_level_fisher_ratio": row_fisher,
        "segment_level_fisher_ratio": seg_fisher,
    }


def resolve_label_dir_for_split(args: argparse.Namespace, split: str) -> Path:
    if args.label_dir is not None:
        return Path(args.label_dir)

    if args.dataset is None or args.iteration is None:
        raise ValueError("Pipeline mode requires --dataset and --iteration, or use --label_dir.")

    return resolve_iteration_label_dir(
        output_root=args.output_root,
        dataset=args.dataset,
        iteration=args.iteration,
        split=split,
    )


def resolve_out_dir_for_split(args: argparse.Namespace, split: str) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir)

    if args.dataset is None or args.iteration is None:
        raise ValueError("Pipeline mode requires --dataset and --iteration, or use --out_dir.")

    return resolve_single_analysis_dir(
        output_root=args.output_root,
        dataset=args.dataset,
        iteration=args.iteration,
    )


def make_summary_row(
    summary: dict[str, Any],
    group_name: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        # These fields come from the top level of summary.
        "dataset": summary.get("dataset"),
        "iteration": summary.get("iteration"),
        "split": summary.get("split"),
        "group": group_name,
        # The following fields come from metrics.
        "num_rows": metrics.get("num_rows"),
        "num_symbols": metrics.get("num_symbols"),
        "num_segments": metrics.get("segment_stats", {}).get("num_segments"),
        "mean_segment_length": metrics.get("segment_stats", {}).get("mean_length"),
        "median_segment_length": metrics.get("segment_stats", {}).get("median_length"),
        "bc_num_boundaries": metrics.get("boundary_consistency", {}).get("num_boundaries"),
        "bc_mean": metrics.get("boundary_consistency", {}).get("boundary_consistency_mean"),
        "bc_std": metrics.get("boundary_consistency", {}).get("boundary_consistency_std"),
        "row_macro_fisher": metrics.get("row_level_fisher_ratio", {}).get("macro_fisher"),
        "segment_macro_fisher": metrics.get("segment_level_fisher_ratio", {}).get("macro_fisher"),
    }


def analyze_one_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    label_dir = resolve_label_dir_for_split(args, split)
    out_dir = ensure_dir(resolve_out_dir_for_split(args, split))

    print("\n" + "=" * 80)
    print(f"Single-iteration label analysis: split={split}")
    print(f"label_dir: {label_dir}")
    print(f"out_dir  : {out_dir}")
    print("=" * 80)

    if args.dry_run:
        return {}

    df = load_label_dir(
        label_dir,
        adjust=args.adjust,
        pattern=args.pattern,
        date_col=args.date_col,
        symbol_col=args.symbol_col,
    )
    
    require_columns(df, [args.symbol_col, args.state_col], "labels")
    # Normalize symbol formatting.
    df[args.symbol_col] = df[args.symbol_col].astype(str).str.zfill(6)

    if args.date_col in df.columns:
        df = df.sort_values([args.symbol_col, args.date_col]).reset_index(drop=True)

   
    # --- optional date-window slice (held-out-period FR robustness) ---------
    if (args.window_start is not None) or (args.window_end is not None):
        if args.window_name is None:
            raise ValueError("--window_name is required when a window is given.")
        n_before = len(df)
        dates = pd.to_datetime(df[args.date_col])
        mask = pd.Series(True, index=df.index)
        if args.window_start is not None:
            start = pd.Timestamp(args.window_start)
            if pd.isna(start):
                raise ValueError(f"Unparseable --window_start: {args.window_start!r}")
            mask &= dates >= start
        if args.window_end is not None:
            end = pd.Timestamp(args.window_end)
            if pd.isna(end):
                raise ValueError(f"Unparseable --window_end: {args.window_end!r}")
            mask &= dates <= end
        df = df.loc[mask].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"Date window [{args.window_start}, {args.window_end}] left zero "
                f"rows for split={split!r}. Check the window against the split."
            )
        print(f"[INFO] window {args.window_name}: {n_before:,} rows -> "
            f"{len(df):,} retained ({n_before - len(df):,} outside window)")
        split = f"{split}_{args.window_name}"   # Composite name used in every output path
    elif args.window_name is not None:
        raise ValueError("--window_name given without --window_start/--window_end.")

    # Read symbol group metadata.
    symbol_groups = load_symbol_groups(
        codebook_dir=args.codebook_dir,
        output_root=args.output_root,
        dataset=args.dataset,
    )
    
    # Build the groups to analyze.
    groups: dict[str, pd.DataFrame] = {"all": df}
    if symbol_groups is not None:
        symbols_with = symbol_groups["has_model_score"]
        symbols_without = symbol_groups["no_model_score"]

        df_with = df[df[args.symbol_col].isin(symbols_with)]
        df_without = df[df[args.symbol_col].isin(symbols_without)]

        if not df_with.empty:
            groups["has_model_score"] = df_with
        if not df_without.empty:
            groups["no_model_score"] = df_without

        print(f"[INFO] Symbol groups: "
              f"has_model_score={len(df_with[args.symbol_col].unique())}, "
              f"no_model_score={len(df_without[args.symbol_col].unique())}")

    # Compute metrics separately for each group.
    group_metrics: dict[str, Any] = {}
    for group_name, group_df in groups.items():
        print(f"[INFO] Computing metrics for group: {group_name} "
              f"({group_df[args.symbol_col].nunique()} symbols, {len(group_df)} rows)")
        group_metrics[group_name] = compute_metrics_for_group(group_df, args)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "iteration": args.iteration,
        "split": split,
        "label_dir": str(label_dir),
        "columns": {
            "symbol_col": args.symbol_col,
            "date_col": args.date_col,
            "state_col": args.state_col,
            "current_col": args.current_col,
            "future_col": args.future_col,
        },
        "group_metrics": group_metrics,
    }

    # Save JSON.
    summary_json_path = out_dir / f"single_iteration_{split}_summary.json"
    save_json(summary, summary_json_path)
    print(f"[INFO] Saved JSON summary -> {summary_json_path}")

    # Save CSV with one row per group.
    csv_rows = []
    for group_name, metrics in group_metrics.items():
        row = make_summary_row(summary, group_name, metrics)
        csv_rows.append(row)

    summary_csv_path = out_dir / f"single_iteration_{split}_summary.csv"
    save_csv(pd.DataFrame(csv_rows), summary_csv_path)
    print(f"[INFO] Saved CSV summary  -> {summary_csv_path}")

    if args.save_detail_tables:
        # Save detail tables only for the all group.
        all_df = groups["all"]

        segments = extract_segments(
            all_df,
            symbol_col=args.symbol_col,
            date_col=args.date_col,
            state_col=args.state_col,
        )
        bc_df, _ = compute_boundary_consistency(
            all_df,
            symbol_col=args.symbol_col,
            date_col=args.date_col,
            state_col=args.state_col,
            radius=args.bc_radius,
        )
        seg_feat_df = compute_segment_level_features(
            all_df,
            symbol_col=args.symbol_col,
            date_col=args.date_col,
            state_col=args.state_col,
            feature_cols=group_metrics["all"]["feature_cols"],
        )

        if not bc_df.empty:
            path = out_dir / f"boundary_consistency_{split}.csv"
            save_csv(bc_df, path)
            print(f"[INFO] Saved boundary table -> {path}")

        if not segments.empty:
            path = out_dir / f"segments_{split}.csv"
            save_csv(segments, path)
            print(f"[INFO] Saved segments -> {path}")

        if not seg_feat_df.empty:
            path = out_dir / f"segment_features_{split}.csv"
            save_csv(seg_feat_df, path)
            print(f"[INFO] Saved segment features -> {path}")

    return summary


def main() -> None:
    args = parse_args()

    if args.label_dir is not None:
        splits = [args.split or "manual"]
    else:
        splits = args.splits

    summaries: list[dict[str, Any]] = []

    for split in splits:
        summary = analyze_one_split(args, split)
        if summary:
            summaries.append(summary)

    if args.dry_run:
        print("\nDry run finished.")
        return

    if summaries:
        if args.out_dir is not None:
            root_out = ensure_dir(args.out_dir)
        else:
            if args.dataset is None or args.iteration is None:
                raise ValueError("Cannot resolve all-splits output dir.")
            root_out = ensure_dir(
                resolve_single_analysis_dir(
                    output_root=args.output_root,
                    dataset=args.dataset,
                    iteration=args.iteration,
                )
            )

        all_json = root_out / "all_splits_single_iteration_summary.json"
        all_csv = root_out / "all_splits_single_iteration_summary.csv"

        save_json(summaries, all_json)

        # Flatten all groups across all splits.
        csv_rows = []
        for s in summaries:
            for group_name, metrics in s.get("group_metrics", {}).items():
                csv_rows.append(make_summary_row(s, group_name, metrics))
        save_csv(pd.DataFrame(csv_rows), all_csv) 
       
        print(f"\n[INFO] Saved all-splits JSON summary -> {all_json}")
        print(f"[INFO] Saved all-splits CSV summary  -> {all_csv}")

    print("\nSingle-iteration label analysis finished.")


if __name__ == "__main__":
    main()