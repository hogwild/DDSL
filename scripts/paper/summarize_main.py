"""Collect the paper's main experiment: Tables 1–3, Figure 2 and text checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from report_statistics import rho
from paper_common import Report, ITERATIONS, METRICS, SCOPE, finite, mean_std, one_row, parser


def flatten(mapping, prefix=""):
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten(value, name)
        else:
            yield name, json.dumps(value, ensure_ascii=False)


def collect(report):
    observations, changes, params, coverage, examples = [], [], [], [], []
    for ds in report.cfg["datasets"]:
        tf, gru = report.pair(ds)
        for section in ["labeling", "splits", "train"]:
            for key, value in flatten(tf["cfg"][section], section):
                # Batch cells override this scalar with main_beta.
                if key == "labeling.ddsl.beta_model":
                    value = str(int(report.beta) / 10)
                params.append({**report.identity(ds, tf, gru), "parameter": key, "value": value})
        for key, value in flatten(gru["cfg"]["gru"], "gru"):
            params.append({**report.identity(ds, tf, gru), "parameter": key, "value": value})
        for seed in report.seeds:
            cell = report.cell(tf, report.beta, seed)
            for it in ITERATIONS:
                observations.append(report.observation(ds, tf, gru, report.beta, seed, it, states=True))
                bounds = report.label_coverage(ds, tf, seed, it)
                coverage.append(bounds)
                if it in [0, 1, 3, 5]:
                    # The paper names the stock/exchange, not its source batch.
                    # List candidates without selecting a dataset or seed.
                    sources = list(Path(bounds["label_dir"]).glob(f"600760*_{bounds['adjust']}.csv"))
                    sources = [p for p in sources if p.name.split("_", 1)[0] == "600760"]
                    examples.append({"dataset": ds, "seed": seed, "iteration": it,
                                     "symbol": "600760", "window_start": "2000-03-28", "window_end": "2000-11-08",
                                     "label_path": str(sources[0]) if len(sources) == 1 else "",
                                     "available": len(sources) == 1})
                if it == 0:
                    continue
                source = (Path(cell["output_root"]) / ds / f"iter_{it}" /
                          "label_analysis/compare_to_prev/all_splits_compare_iterations_summary.csv")
                row = one_row(report.csv(source), {"dataset": ds, "base_iteration": it - 1,
                              "new_iteration": it, "split": "test", "group": "all"}, source)
                changes.append({**report.identity(ds, tf, gru), "seed": seed, "base_iteration": it - 1,
                                "new_iteration": it, "label_scope": SCOPE,
                                "joint_change_rate": finite(row["joint_change_rate"], "change rate", 0, 1),
                                "joint_kappa": finite(row["joint_kappa"], "kappa", -1, 1),
                                "num_aligned_rows": finite(row["num_aligned_rows"], "aligned rows", 1)})
    obs = pd.DataFrame(observations)
    cov = pd.DataFrame(coverage)
    obs = obs.merge(cov[["dataset", "seed", "iteration", "label_date_min", "label_date_max", "label_rows", "label_symbols"]],
                    on=["dataset", "seed", "iteration"], validate="one_to_one")
    for ds, group in obs[obs.iteration == 0].groupby("dataset"):
        cols = [c for c in obs if c.startswith("state_dist.")] + ["segment_fisher"]
        if not np.allclose(group[cols].to_numpy(), group[cols].iloc[0].to_numpy(), rtol=0, atol=1e-9):
            raise ValueError(f"{ds}: DP baseline label statistics differ across seeds")
    baseline = obs[obs.iteration == 0].set_index(["dataset", "seed"])["entropy_norm"]
    obs["entropy_change_pct"] = [100 * (row.entropy_norm / finite(baseline.loc[(row.dataset, row.seed)],
                                 "baseline entropy", np.finfo(float).eps) - 1) for row in obs.itertuples()]
    keys = ["dataset", "adjust", "data_policy", "label_scope", "iteration"]
    stats = mean_std(obs, keys, METRICS + ["entropy_norm", "entropy_change_pct", "transformer_minus_gru"])
    table2, checks, gain_means = [], [], []
    for ds, group in stats.groupby("dataset", sort=False):
        group = group.set_index("iteration")
        row = {"dataset": ds, "adjust": group.iloc[0]["adjust"], "label_scope": SCOPE, "n_seeds": len(report.seeds)}
        for metric, iters in [("transformer_f1", [0, 5]), ("gru_f1", [0, 5]),
                              ("segment_fisher", [0, 1, 5]), ("entropy_change_pct", [5])]:
            for it in iters:
                for stat in ["mean", "std"]:
                    row[f"{metric}_iter{it}_{stat}"] = group.loc[it, f"{metric}_{stat}"]
        table2.append(row)
        f = group.transformer_f1_mean
        total = f.loc[5] - f.loc[0]
        gain_means.append({"dataset": ds, "basis": "ratio of seed-mean F1 gains",
                           "iter1_share_of_total_gain": (f.loc[1] - f.loc[0]) / total if total else None,
                           "iter2_share_of_total_gain": (f.loc[2] - f.loc[0]) / total if total else None,
                           "iter4_to_5_gain": f.loc[5] - f.loc[4],
                           "iter5_f1_std": group.loc[5, "transformer_f1_std"]})
    for (ds, seed), group in obs.groupby(["dataset", "seed"], sort=False):
        g = group.sort_values("iteration").set_index("iteration")
        fr = g.segment_fisher
        f1 = g.transformer_f1
        gf = g.gru_f1
        checks.append({"dataset": ds, "seed": seed, "label_scope": SCOPE,
                       "transformer_strictly_increasing": bool((f1.diff().dropna() > 0).all()),
                       "gru_strictly_increasing": bool((gf.diff().dropna() > 0).all()),
                       "gru_rho_vs_iteration": rho(ITERATIONS, gf.to_numpy()),
                       "fisher_peak_iteration": int(fr.idxmax()),
                       "fisher_strictly_decreasing_iter1_5": bool((fr.loc[1:].diff().dropna() < 0).all()),
                       "f1_fisher_rho_iter1_5": rho(f1.loc[1:].to_numpy(), fr.loc[1:].to_numpy()),
                       "transformer_gain_pct": 100 * (f1.loc[5] / f1.loc[0] - 1) if f1.loc[0] else None,
                       "fisher_drop_from_iter0_pct": 100 * (1 - fr.loc[5] / fr.loc[0]) if fr.loc[0] else None,
                       "fisher_drop_from_iter1_pct": 100 * (1 - fr.loc[5] / fr.loc[1]) if fr.loc[1] else None,
                       "min_state_share_all_iterations": g.min_state_share.min(),
                       "max_zero_states": int(g.n_zero_states.max()),
                       "entropy_change_iter5_pct": g.loc[5, "entropy_change_pct"]})
    changes = pd.DataFrame(changes)
    table3 = mean_std(changes, ["dataset", "adjust", "label_scope", "base_iteration", "new_iteration"],
                      ["joint_change_rate", "joint_kappa"])
    table3["CR/kappa"] = [f"{r.joint_change_rate_mean:.3f}/{r.joint_kappa_mean:.3f}" for r in table3.itertuples()]
    tables = {"main_per_seed.csv": obs, "main_mean_std.csv": stats,
            "table1_configuration.csv": pd.DataFrame(params), "table2_main.csv": pd.DataFrame(table2),
            "stability_per_seed.csv": changes, "table3_stability.csv": table3,
            "trajectory_checks_per_seed.csv": pd.DataFrame(checks),
            "saturation_from_mean_trajectory.csv": pd.DataFrame(gain_means),
            "label_coverage.csv": cov, "figure1_sources.csv": pd.DataFrame(examples,
                columns=["dataset", "seed", "iteration", "symbol", "window_start", "window_end", "label_path", "available"])}
    return tables


def main():
    args = parser(__doc__).parse_args()
    report = Report(args.config)
    report.write(args.output_dir, collect(report), "summarize_main", args.dry_run)


if __name__ == "__main__":
    main()
