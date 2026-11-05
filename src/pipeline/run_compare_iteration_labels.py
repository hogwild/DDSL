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
    ensure_dir,
    load_label_dir,
    load_symbol_groups,
    resolve_iteration_label_dir,
    resolve_pairwise_analysis_dir,
    save_csv,
    save_json,
)
from analysis.metrics_pairwise import (
    align_base_new,
    compute_agreement,
    compute_boundary_overlap,
    confusion_table,
    distribution_similarity,
    transition_series,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare labels between two DDSL iterations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pipeline mode
    p.add_argument("--dataset", default=None)
    p.add_argument("--base_iteration", type=int, default=None)
    p.add_argument("--new_iteration", type=int, default=None)
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--adjust", default="hfq")
    p.add_argument("--splits", nargs="*", default=["test"]) #["train", "val", "test"]

    # Manual mode
    p.add_argument("--base_label_dir", default=None)
    p.add_argument("--new_label_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--split", default=None)

    # Columns
    p.add_argument("--date_col", default=DEFAULT_DATE_COL)
    p.add_argument("--symbol_col", default=DEFAULT_SYMBOL_COL)
    p.add_argument("--state_col", default=DEFAULT_STATE_COL)
    p.add_argument("--current_col", default=DEFAULT_CURRENT_COL)
    p.add_argument("--future_col", default=DEFAULT_FUTURE_COL)

    # File matching
    p.add_argument(
        "--pattern",
        default=None,
        help="CSV pattern. If omitted, tries '*_{adjust}.csv' first, then '*.csv'.",
    )

    # Output control
    p.add_argument(
        "--save_detail_tables",
        action="store_true",
        help="Save aligned label table, boundary overlap table, and confusion matrices.",
    )

    # Path to symbol list in codebook
    p.add_argument(
        "--codebook_dir",
        default=None,
        help="Path to codebook directory containing symbol_codebook_info.json. "
            "If provided, comparison is computed separately for symbols with and without codebook.",
    )

    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def resolve_base_label_dir(args: argparse.Namespace, split: str) -> Path:
    if args.base_label_dir is not None:
        return Path(args.base_label_dir)

    if args.dataset is None or args.base_iteration is None:
        raise ValueError("Pipeline mode requires --dataset and --base_iteration.")

    return resolve_iteration_label_dir(
        output_root=args.output_root,
        dataset=args.dataset,
        iteration=args.base_iteration,
        split=split,
    )


def resolve_new_label_dir(args: argparse.Namespace, split: str) -> Path:
    if args.new_label_dir is not None:
        return Path(args.new_label_dir)

    if args.dataset is None or args.new_iteration is None:
        raise ValueError("Pipeline mode requires --dataset and --new_iteration.")

    return resolve_iteration_label_dir(
        output_root=args.output_root,
        dataset=args.dataset,
        iteration=args.new_iteration,
        split=split,
    )


def resolve_out_dir_for_split(args: argparse.Namespace, split: str) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir)

    if args.dataset is None or args.new_iteration is None:
        raise ValueError("Pipeline mode requires --dataset and --new_iteration, or use --out_dir.")

    return resolve_pairwise_analysis_dir(
        output_root=args.output_root,
        dataset=args.dataset,
        new_iteration=args.new_iteration,
    )


def make_summary_row(
    summary: dict[str, Any],
    group_name: str,
    group_data: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    agreement = group_data.get("agreement", {})
    joint_agr = agreement.get(args.state_col, {})
    current_agr = agreement.get(args.current_col, {})
    future_agr = agreement.get(args.future_col, {})

    return {
        # Top-level metadata from summary
        "dataset": summary.get("dataset"),
        "base_iteration": summary.get("base_iteration"),
        "new_iteration": summary.get("new_iteration"),
        "split": summary.get("split"),
        "group": group_name,
        # The following fields come from group_data.
        "num_base_rows": group_data.get("num_base_rows"),
        "num_new_rows": group_data.get("num_new_rows"),
        "num_aligned_rows": group_data.get("num_aligned_rows"),
        "num_symbols_aligned": group_data.get("num_symbols_aligned"),
        "joint_agreement": joint_agr.get("agreement"),
        "joint_kappa": joint_agr.get("kappa"),
        "joint_change_rate": joint_agr.get("change_rate"),
        "current_agreement": current_agr.get("agreement"),
        "current_kappa": current_agr.get("kappa"),
        "current_change_rate": current_agr.get("change_rate"),
        "future_agreement": future_agr.get("agreement"),
        "future_kappa": future_agr.get("kappa"),
        "future_change_rate": future_agr.get("change_rate"),
        "state_jsd": group_data.get("state_distribution_similarity", {}).get("jsd"),
        "state_tv": group_data.get("state_distribution_similarity", {}).get("tv"),
        "transition_jsd": group_data.get("transition_distribution_similarity", {}).get("jsd"),
        "transition_tv": group_data.get("transition_distribution_similarity", {}).get("tv"),
        "boundary_base_num": group_data.get("boundary_overlap", {}).get("base_num_boundaries"),
        "boundary_new_num": group_data.get("boundary_overlap", {}).get("new_num_boundaries"),
        "boundary_jaccard": group_data.get("boundary_overlap", {}).get("jaccard"),
    }

def compare_one_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    base_dir = resolve_base_label_dir(args, split)
    new_dir = resolve_new_label_dir(args, split)
    out_dir = ensure_dir(resolve_out_dir_for_split(args, split))

    print("\n" + "=" * 80)
    print(f"Compare iteration labels: split={split}")
    print(f"base_label_dir: {base_dir}")
    print(f"new_label_dir : {new_dir}")
    print(f"out_dir       : {out_dir}")
    print("=" * 80)

    if args.dry_run:
        return {}

    base_df = load_label_dir(
        base_dir,
        adjust=args.adjust,
        pattern=args.pattern,
        date_col=args.date_col,
        symbol_col=args.symbol_col,
    )
    new_df = load_label_dir(
        new_dir,
        adjust=args.adjust,
        pattern=args.pattern,
        date_col=args.date_col,
        symbol_col=args.symbol_col,
    )

    base_df[args.symbol_col] = base_df[args.symbol_col].astype(str).str.zfill(6)
    new_df[args.symbol_col] = new_df[args.symbol_col].astype(str).str.zfill(6)

    # Read symbol group metadata.
    symbol_groups = load_symbol_groups(
        codebook_dir=args.codebook_dir,
        output_root=args.output_root,
        dataset=args.dataset,
    )

    # Non-codebook symbols keep the same labels across iterations; retain them only as a reference.
    groups: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "all": (base_df, new_df)
    }
    if symbol_groups is not None:
        symbols_with = symbol_groups["has_model_score"]
        symbols_without = symbol_groups["no_model_score"]

        base_with = base_df[base_df[args.symbol_col].isin(symbols_with)]
        new_with = new_df[new_df[args.symbol_col].isin(symbols_with)]

        base_without = base_df[base_df[args.symbol_col].isin(symbols_without)]
        new_without = new_df[new_df[args.symbol_col].isin(symbols_without)]

        if not base_with.empty and not new_with.empty:
            groups["has_model_score"] = (base_with, new_with)
        if not base_without.empty and not new_without.empty:
            groups["no_model_score"] = (base_without, new_without)

        print(f"[INFO] Symbol groups: "
              f"has_model_score={base_with[args.symbol_col].nunique()}, "
              f"no_model_score={base_without[args.symbol_col].nunique()}")

    # Compute comparison metrics separately for each group.
    group_comparisons: dict[str, Any] = {}
    for group_name, (g_base_df, g_new_df) in groups.items():
        print(f"[INFO] Comparing group: {group_name} "
              f"({g_base_df[args.symbol_col].nunique()} symbols)")

        label_cols = [args.state_col]
        if args.current_col in g_base_df.columns and args.current_col in g_new_df.columns:
            label_cols.append(args.current_col)
        if args.future_col in g_base_df.columns and args.future_col in g_new_df.columns:
            label_cols.append(args.future_col)

        aligned = align_base_new(
            g_base_df,
            g_new_df,
            symbol_col=args.symbol_col,
            date_col=args.date_col,
            cols=label_cols,
        )

        if aligned.empty:
            print(f"[WARN] No overlapping rows for group={group_name} split={split}")
            continue

        agreement = compute_agreement(aligned, label_cols=label_cols)

        state_similarity = distribution_similarity(
            aligned[f"base_{args.state_col}"],
            aligned[f"new_{args.state_col}"],
        )

        base_trans = transition_series(
            g_base_df,
            symbol_col=args.symbol_col,
            date_col=args.date_col,
            state_col=args.state_col,
        )
        new_trans = transition_series(
            g_new_df,
            symbol_col=args.symbol_col,
            date_col=args.date_col,
            state_col=args.state_col,
        )
        transition_similarity = distribution_similarity(base_trans, new_trans)

        boundary_overlap_df, boundary_overlap = compute_boundary_overlap(
            g_base_df,
            g_new_df,
            symbol_col=args.symbol_col,
            date_col=args.date_col,
            state_col=args.state_col,
        )

        group_comparisons[group_name] = {
            "num_base_rows": int(len(g_base_df)),
            "num_new_rows": int(len(g_new_df)),
            "num_aligned_rows": int(len(aligned)),
            "num_symbols_aligned": int(aligned[args.symbol_col].nunique()),
            "label_cols": label_cols,
            "agreement": agreement,
            "state_distribution_similarity": state_similarity,
            "transition_distribution_similarity": transition_similarity,
            "boundary_overlap": boundary_overlap,
        }

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "base_iteration": args.base_iteration,
        "new_iteration": args.new_iteration,
        "split": split,
        "base_label_dir": str(base_dir),
        "new_label_dir": str(new_dir),
        "group_comparisons": group_comparisons,
    }

    # Save JSON.
    summary_json_path = out_dir / f"compare_iterations_{split}_summary.json"
    save_json(summary, summary_json_path)
    print(f"[INFO] Saved JSON summary -> {summary_json_path}")

    # Save CSV with one row per group.
    csv_rows = [
        make_summary_row(summary, group_name, group_data, args)
        for group_name, group_data in group_comparisons.items()
    ]
    summary_csv_path = out_dir / f"compare_iterations_{split}_summary.csv"
    save_csv(pd.DataFrame(csv_rows), summary_csv_path)
    print(f"[INFO] Saved CSV summary  -> {summary_csv_path}")

    if args.save_detail_tables:
        # Detail tables cover codebook symbols; use all when group metadata is unavailable.
        target_group = "has_model_score" if "has_model_score" in groups else "all"
        g_base_df, g_new_df = groups[target_group]

        label_cols = group_comparisons.get(target_group, {}).get("label_cols", [args.state_col])

        aligned = align_base_new(
            g_base_df,
            g_new_df,
            symbol_col=args.symbol_col,
            date_col=args.date_col,
            cols=label_cols,
        )

        boundary_overlap_df, _ = compute_boundary_overlap(
            g_base_df,
            g_new_df,
            symbol_col=args.symbol_col,
            date_col=args.date_col,
            state_col=args.state_col,
        )

        if not aligned.empty:
            save_csv(aligned, out_dir / f"aligned_labels_{split}.csv")
            print(f"[INFO] Saved aligned labels -> {out_dir / f'aligned_labels_{split}.csv'}")

        if not boundary_overlap_df.empty:
            save_csv(boundary_overlap_df, out_dir / f"boundary_overlap_{split}.csv")
            print(f"[INFO] Saved boundary overlap -> {out_dir / f'boundary_overlap_{split}.csv'}")

        for col in label_cols:
            tab = confusion_table(aligned, col=col)
            if not tab.empty:
                path = out_dir / f"{col}_confusion_{split}.csv"
                save_csv(tab, path)
                print(f"[INFO] Saved confusion table -> {path}")

    return summary


def main() -> None:
    args = parse_args()

    if args.base_label_dir is not None or args.new_label_dir is not None:
        if args.base_label_dir is None or args.new_label_dir is None:
            raise ValueError("Manual mode requires both --base_label_dir and --new_label_dir.")
        splits = [args.split or "manual"]
    else:
        if args.dataset is None or args.base_iteration is None or args.new_iteration is None:
            raise ValueError(
                "Pipeline mode requires --dataset --base_iteration --new_iteration."
            )
        splits = args.splits

    summaries: list[dict[str, Any]] = []

    for split in splits:
        summary = compare_one_split(args, split)
        if summary:
            summaries.append(summary)

    if args.dry_run:
        print("\nDry run finished.")
        return

    if summaries:
        if args.out_dir is not None:
            root_out = ensure_dir(args.out_dir)
        else:
            if args.dataset is None or args.new_iteration is None:
                raise ValueError("Cannot resolve all-splits output dir.")
            root_out = ensure_dir(
                resolve_pairwise_analysis_dir(
                    output_root=args.output_root,
                    dataset=args.dataset,
                    new_iteration=args.new_iteration,
                )
            )

        all_json = root_out / "all_splits_compare_iterations_summary.json"
        all_csv = root_out / "all_splits_compare_iterations_summary.csv"

        save_json(summaries, all_json)
        # Flatten all groups across all splits.
        csv_rows = []
        for s in summaries:
            for group_name, group_data in s.get("group_comparisons", {}).items():
                csv_rows.append(make_summary_row(s, group_name, group_data, args))
        save_csv(pd.DataFrame(csv_rows), all_csv)

        print(f"\n[INFO] Saved all-splits JSON summary -> {all_json}")
        print(f"[INFO] Saved all-splits CSV summary  -> {all_csv}")

    print("\nIteration label comparison finished.")


if __name__ == "__main__":
    main()