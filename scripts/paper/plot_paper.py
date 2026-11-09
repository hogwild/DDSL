"""Render the final paper's figures from verified report data, never literal coordinates."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from paper_common import ITERATIONS, Report, one_row, parser, sha256

STATE_COLORS = {
    "BULL_TO_BULL": "#1a6e3c", "BULL_TO_NEUTRAL": "#6ab187", "BULL_TO_BEAR": "#b8dfc8",
    "NEUTRAL_TO_BULL": "#4e86c4", "NEUTRAL_TO_NEUTRAL": "#aec6e8", "NEUTRAL_TO_BEAR": "#c6dcf0",
    "BEAR_TO_BULL": "#f0c4b0", "BEAR_TO_NEUTRAL": "#e08060", "BEAR_TO_BEAR": "#c0392b",
}


def read_stage(report, directory, command):
    directory = Path(directory).resolve()
    receipt = report.json(directory / "report.json")
    if report.track(directory / "COMPLETE").read_text().strip() != sha256(directory / "report.json"):
        raise ValueError(f"Incomplete or changed report: {directory}")
    if receipt["command"] != command or receipt["config"] != report.cfg:
        raise ValueError(f"Report/config mismatch: {directory}")

    def read(name):
        path = report.track(directory / name)
        if receipt["outputs_sha256"].get(name) != sha256(path):
            raise ValueError(f"Report CSV checksum mismatch: {path}")
        return pd.read_csv(path, dtype={"symbol": str})
    return receipt, read


def _plot_trajectory(ax, frame, metric, label):
    """Draw one metric consistently in the combined and standalone figures."""
    for ds, rows in frame.groupby("dataset", sort=False):
        rows = rows.sort_values("iteration")
        if rows.iteration.tolist() != ITERATIONS:
            raise ValueError(f"Expected iterations 0..5 once per dataset: {ds}")
        x, y, sd = (rows[column].to_numpy(dtype=float) for column in
                    ["iteration", metric + "_mean", metric + "_std"])
        if not np.isfinite([x, y, sd]).all() or (sd < 0).any():
            raise ValueError(f"Invalid trajectory mean/SD: {ds}, {metric}")
        line, = ax.plot(x, y, marker="o", markersize=3, label=ds)
        ax.fill_between(x, y - sd, y + sd, color=line.get_color(), alpha=.15)
        ax.set(xlabel="Iteration", ylabel=label, xticks=ITERATIONS)
        ax.legend(fontsize=7)


def plot_trajectories(frame):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
    _plot_trajectory(axes[0], frame, "transformer_f1", "Transformer macro F1")
    _plot_trajectory(axes[1], frame, "segment_fisher", "Segment Fisher ratio")
    return fig


def plot_figure2a(frame):
    """Standalone Figure 2(a): Transformer macro F1, with mean +/- SD bands."""
    fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
    _plot_trajectory(ax, frame, "transformer_f1", "Transformer macro F1")
    return fig


def plot_figure2b(frame):
    """Standalone Figure 2(b): segment Fisher ratio, with mean +/- SD bands."""
    fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
    _plot_trajectory(ax, frame, "segment_fisher", "Segment Fisher ratio")
    return fig


def plot_frontier(frame):
    rows = frame[frame.dataset == "SSE50"].copy()
    iteration = rows[rows.sweep == "iteration"].sort_values("iteration")
    if iteration.iteration.tolist() != ITERATIONS:
        raise ValueError("SSE50 frontier requires all main iterations")
    weight = rows[rows.sweep == "weight"]
    dp = one_row(weight, {"condition": "dp"}, "frontier DP")
    finite = weight[weight.condition.str.startswith("beta_")].copy()
    finite["beta"] = finite.condition.str.removeprefix("beta_").astype(float)
    finite = finite.sort_values("beta")
    if finite.empty or finite.beta.duplicated().any():
        raise ValueError("Frontier requires unique finite-beta conditions")
    shared = one_row(finite, {"condition": "beta_0.3"}, "frontier intersection")
    columns = ["segment_fisher_mean", "transformer_f1_mean"]
    if not np.allclose(shared[columns].to_numpy(dtype=float),
                       iteration.iloc[-1][columns].to_numpy(dtype=float), rtol=0, atol=1e-10):
        raise ValueError("Iteration and beta sweeps disagree at their shared configuration")
    fig, ax = plt.subplots(figsize=(5, 3.8), layout="constrained")
    ax.plot(iteration[columns[0]], iteration[columns[1]], "o-", label="Iteration (beta=0.3)")
    weight_curve = pd.concat([dp.to_frame().T, finite], ignore_index=True)
    ax.plot(weight_curve[columns[0]], weight_curve[columns[1]], "s--", label="Weight (iteration 5)")
    for row in iteration.itertuples():
        ax.annotate(f"i{row.iteration}", (row.segment_fisher_mean, row.transformer_f1_mean),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    for row in finite.itertuples():
        ax.annotate(f"{row.beta:.1f}", (row.segment_fisher_mean, row.transformer_f1_mean),
                    xytext=(4, -10), textcoords="offset points", fontsize=7)
    pseudo = weight[weight.condition == "pseudo"]
    if len(pseudo) > 1:
        raise ValueError("Duplicate pseudo point")
    if len(pseudo):
        ax.scatter(pseudo[columns[0]], pseudo[columns[1]], marker="x", color="black", label="Pseudo")
    ax.scatter([dp[columns[0]]], [dp[columns[1]]], color="black", label="DP", zorder=5)
    ax.axvline(dp[columns[0]], color="grey", ls=":")
    ax.axhline(dp[columns[1]], color="grey", ls=":")
    ax.set(xlabel="Segment Fisher ratio", ylabel="Transformer macro F1", title="SSE50")
    ax.legend(fontsize=7)
    return fig


def example_data(report, candidates, receipt, dataset, seed):
    frames = []
    for iteration in [0, 1, 3, 5]:
        row = one_row(candidates, {"dataset": dataset, "seed": seed, "iteration": iteration}, "Figure 1")
        if not row.available or pd.isna(row.label_path):
            raise ValueError(f"Missing Figure 1 labels: {dataset}, seed={seed}, iteration={iteration}")
        path = report.track(row.label_path)
        if receipt["inputs_sha256"].get(str(path)) != sha256(path):
            raise ValueError(f"Figure 1 labels changed since main report: {path}")
        tf, _ = report.pair(dataset, need_gru=False)
        cols = tf["cfg"]["labeling"]["columns"]
        df = pd.read_csv(path).rename(columns={cols.get("date_col", "date"): "date",
                                              cols.get("price_col", "close"): "close"})
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        df = df[df.date.between(row.window_start, row.window_end)].sort_values("date")
        if len(df) < 2 or df.date.duplicated().any() or not df.joint_state.isin(STATE_COLORS).all():
            raise ValueError(f"Invalid or insufficient Figure 1 dates/states: {path}")
        if not np.isfinite(df.close).all() or (df.close <= 0).any():
            raise ValueError(f"Invalid Figure 1 prices: {path}")
        df = df[["date", "close", "joint_state"]].reset_index(drop=True)
        if frames and not df[["date", "close"]].equals(frames[0][["date", "close"]]):
            raise ValueError("Figure 1 dates/prices differ across iterations")
        frames.append(df.assign(iteration=iteration, dataset=dataset, seed=seed, symbol=str(row.symbol)))
    return pd.concat(frames, ignore_index=True)


def plot_example(frame):
    fig, axes = plt.subplots(4, 1, figsize=(9, 6), sharex=True, layout="constrained")
    for ax, (iteration, rows) in zip(axes, frame.groupby("iteration", sort=True)):
        rows = rows.reset_index(drop=True)
        ax.plot(rows.date, rows.close, color="#222222", lw=.9)
        states = rows.joint_state.to_numpy()
        starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
        for start, stop in zip(starts, np.r_[starts[1:], len(rows)]):
            right = rows.date.iloc[stop] if stop < len(rows) else rows.date.iloc[-1]
            ax.axvspan(rows.date.iloc[start], right, color=STATE_COLORS[states[start]], alpha=.65)
        ax.set_ylabel(f"iter {iteration}")
        ax.set_yticks([])
    fig.legend(handles=[Patch(color=c, label=s.replace("_TO_", " → ")) for s, c in STATE_COLORS.items()],
               loc="outside upper center", ncol=3, fontsize=7)
    axes[-1].set_xlabel(f"{frame.iloc[0].symbol} · seed {frame.iloc[0].seed}")
    return fig


def plot_example_poster(frame):
    """Figure 1 for the poster: 38.9 x 19.9 cm including the bottom legend.

    Uses the same verified prices, states and segment boundaries as the paper
    version. Save with bbox_inches=None to preserve the physical canvas size.
    The stock identifier belongs in the poster caption, outside this figure.
    """
    fig = plot_example(frame)
    fig.set_size_inches(38.9 / 2.54, 19.9 / 2.54)
    for legend in list(fig.legends):
        legend.remove()
    for ax, iteration in zip(fig.axes, [0, 1, 3, 5]):
        ax.set_ylabel("")
        ax.text(.01, .94, "iter 0 (DP baseline)" if iteration == 0 else f"iter {iteration}",
                transform=ax.transAxes, va="top", fontsize=18, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 2})
        ax.tick_params(axis="x", labelsize=18)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.lines[0].set_linewidth(1.2)
    fig.axes[-1].set_xlabel("Date", fontsize=18)
    fig.legend(handles=[Patch(color=c, alpha=.65, label=s.replace("_TO_", " → "))
                        for s, c in STATE_COLORS.items()],
               loc="outside lower center", ncol=3, fontsize=18, frameon=False)
    return fig


def run(args):
    report = Report(args.config)
    for ds in report.cfg["datasets"]:
        report.pair(ds)
    output = report.check_output(args.output_dir)
    for directory in [args.main_dir, args.ablation_dir]:
        directory = directory.resolve()
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError("Figures must be separate from source reports")
    main_receipt, main = read_stage(report, args.main_dir, "summarize_main")
    _, ablation = read_stage(report, args.ablation_dir, "summarize_ablation")
    figures = {}
    notes = []
    try:
        trajectories = main("main_mean_std.csv")
        figures["figure2_cross_market"] = plot_trajectories(trajectories)
        figures["figure2a_learnability"] = plot_figure2a(trajectories)
        figures["figure2b_price_separation"] = plot_figure2b(trajectories)
        sweeps = ablation("figure3_sweeps.csv")
        if "SSE50" in set(sweeps.dataset):
            figures["figure3_pareto_frontier"] = plot_frontier(sweeps)
        else:
            notes.append("Figure 3 skipped: selected report has no SSE50 ablation.")
        example = None
        if args.example_seed is not None:
            example = example_data(report, main("figure1_sources.csv"), main_receipt,
                                   args.example_dataset, args.example_seed)
            figures["figure1_stock_example"] = plot_example(example)
            figures["figure1_stock_example_poster"] = plot_example_poster(example)
        else:
            notes.append("Figure 1 skipped: specify --example-seed after confirming its source run.")
        report.verify_inputs()
        for note in notes:
            print(f"[INFO] {note}")
        if args.dry_run:
            print(f"[DRY RUN] Figures validated in memory: {', '.join(figures)}")
            return
        output.mkdir(parents=True, exist_ok=False)
        for name, figure in figures.items():
            for extension in ["png", "pdf"]:
                if name == "figure1_stock_example_poster":
                    # A tight crop would change the size measured for the poster.
                    with plt.rc_context({"savefig.bbox": None}):
                        figure.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches=None)
                else:
                    figure.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches="tight")
        if example is not None:
            example.to_csv(output / "figure1_source_data.csv", index=False)
        receipt = report.metadata("plot_paper")
        receipt["figure_notes"] = notes
        receipt["example"] = {"dataset": args.example_dataset, "seed": args.example_seed}
        receipt["outputs_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir())}
        (output / "report.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        report.verify_inputs()
        (output / "COMPLETE").write_text(sha256(output / "report.json") + "\n")
        print(f"[OK] Figures: {output}")
    finally:
        for figure in figures.values():
            plt.close(figure)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--main-dir", required=True, type=Path)
    p.add_argument("--ablation-dir", required=True, type=Path)
    p.add_argument("--example-seed", type=int)
    p.add_argument("--example-dataset", default="SSE50")
    run(p.parse_args())
