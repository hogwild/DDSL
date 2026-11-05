"""Table 4 and Figure 3 data, preserving DP and pseudo as distinct conditions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from paper_common import ITERATIONS, METRICS, Report, mean_std, parser


def collect(report):
    rows, trajectories, checks, baseline_checks = [], [], [], []
    for ds, spec in report.cfg["datasets"].items():
        if not spec["ablation"]:
            continue
        tf, gru = report.pair(ds)
        for seed in report.seeds:
            dp = report.observation(ds, tf, gru, report.beta, seed, 0, states=True)
            for it in ITERATIONS:
                row = report.observation(ds, tf, gru, report.beta, seed, it)
                trajectories.append({**row, "sweep": "iteration", "condition": "beta_0.3"})
            rows.append({**dp, "condition": "dp", "labeler_kind": "dp", "beta_value": 0.0,
                         "baseline_source": "main beta03 iter0 (one observation per seed)"})
            cell_rows = []
            for beta in report.cfg["ablation_betas"]:
                base = report.observation(ds, tf, gru, beta, seed, 0, states=True)
                label_keys = [k for k in dp if k.startswith("state_dist.")] + ["segment_fisher"]
                if not np.allclose([base[k] for k in label_keys], [dp[k] for k in label_keys], rtol=0, atol=1e-9):
                    raise ValueError(f"{ds}, seed={seed}, beta={beta}: iter0 labels differ from main DP; do not combine")
                # Training currently enables cuDNN benchmarking/non-determinism.
                # Report repeated DP training differences; do not demand bitwise F1
                # equality or average repeated beta arms as extra independent seeds.
                baseline_checks.append({"dataset": ds, "seed": seed, "beta_tag": beta,
                                        "transformer_f1_minus_main_dp": base["transformer_f1"] - dp["transformer_f1"],
                                        "gru_f1_minus_main_dp": base["gru_f1"] - dp["gru_f1"],
                                        "label_statistics_match": True})
                row = report.observation(ds, tf, gru, beta, seed, 5)
                row.update(condition="pseudo" if beta == "inf" else f"beta_{int(beta)/10:.1f}",
                           beta_value=None if beta == "inf" else int(beta) / 10)
                rows.append(row)
                cell_rows.append(row)
            finite = sorted([r for r in cell_rows if r["condition"] != "pseudo"], key=lambda r: r["beta_value"])
            for metric in ["transformer_f1", "gru_f1"]:
                values = [dp[metric]] + [r[metric] for r in finite]
                check = {"dataset": ds, "seed": seed, "metric": metric,
                         "betas_checked": ",".join(["dp"] + [r["beta_tag"] for r in finite]),
                         "strictly_increasing": bool(np.all(np.diff(values) > 0))}
                pseudo = next((r for r in cell_rows if r["condition"] == "pseudo"), None)
                beta08 = next((r for r in finite if r["beta_tag"] == "08"), None)
                check["pseudo_minus_beta08"] = pseudo[metric] - beta08[metric] if pseudo and beta08 else None
                checks.append(check)
    if not rows:
        raise ValueError("No datasets selected with ablation: true")
    per_seed = pd.DataFrame(rows)
    # Condition, rather than numeric beta, is the key: pseudo must survive groupby.
    summary = mean_std(per_seed, ["dataset", "adjust", "data_policy", "condition", "labeler_kind"],
                       METRICS + ["transformer_minus_gru"])
    endpoints = mean_std(per_seed, ["dataset", "adjust", "condition", "iteration"], METRICS)
    endpoints["sweep"] = "weight"
    iteration = mean_std(pd.DataFrame(trajectories), ["dataset", "adjust", "condition", "iteration"], METRICS)
    iteration["sweep"] = "iteration"
    return {"ablation_per_seed.csv": per_seed, "table4_ablation.csv": summary,
            "figure3_sweeps.csv": pd.concat([iteration, endpoints], ignore_index=True),
            "beta_trend_checks_per_seed.csv": pd.DataFrame(checks),
            "dp_baseline_checks.csv": pd.DataFrame(baseline_checks)}


def main():
    args = parser(__doc__).parse_args()
    report = Report(args.config)
    report.write(args.output_dir, collect(report), "summarize_ablation", args.dry_run)


if __name__ == "__main__":
    main()
