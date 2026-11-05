"""Table 5 and predicted-segment duration checks from a completed event report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_common import ITERATIONS, Report, finite, integer, mean_std, one_row, parser, sha256

FOCUS = {"BEAR_TO_BULL": "BEAR_TO_BEAR", "NEUTRAL_TO_BULL": "NEUTRAL_TO_NEUTRAL"}
TRANSITIONS = {**FOCUS, "BULL_TO_BEAR": "BULL_TO_BULL", "NEUTRAL_TO_BEAR": "NEUTRAL_TO_NEUTRAL"}


def collect(report, event_dir):
    event_dir = Path(event_dir).resolve()
    receipt = report.json(event_dir / "report.json")
    if report.track(event_dir / "COMPLETE").read_text().strip() != sha256(event_dir / "report.json"):
        raise ValueError("Event report checksum mismatch or incomplete report")
    if receipt.get("command") != "prepare_events" or receipt.get("schema_version") != 1:
        raise ValueError("Use output from scripts/paper/prepare_events.py")
    for name, digest in receipt["outputs_sha256"].items():
        path = (event_dir / name).resolve()
        if not path.is_relative_to(event_dir) or path == event_dir:
            raise ValueError(f"Invalid event artifact: {name}")
        report.track(path)
        if report.inputs[str(path)] != digest:
            raise ValueError(f"Event artifact checksum mismatch: {path}")
    # Source hashes are recorded by prepare_events; no requirement that today's
    # analysis/training code still equals that historical version.
    sources = {(s["dataset"], s["seed"]): s for s in receipt["sources"]}
    if len(sources) != len(receipt["sources"]):
        raise ValueError("Duplicate event sources")
    transitions, segments, controls = [], [], []
    for ds in report.cfg["datasets"]:
        tf, _ = report.pair(ds, need_gru=False)
        for seed in report.seeds:
            source = sources.get((ds, seed))
            cell = report.cell(tf, report.beta, seed)
            split = tf["cfg"]["splits"]["test"]
            if (not source or source["transformer_manifest_sha256"] != tf["hash"] or
                    source["beta_tag"] != report.beta or source["cell_output_root"] != cell["output_root"] or
                    source["horizon"] != 20 or source["iterations"] != ITERATIONS or
                    source["test_start"] != split["start_date"] or source["test_end"] != split["end_date"]):
                raise ValueError(f"Event source does not match selected batch: {ds}, seed={seed}")
            prefix = (event_dir / source["prefix"]).resolve()
            if not prefix.is_relative_to(event_dir):
                raise ValueError("Event prefix outside report")
            for method, suffix in [("raw", ""), ("viterbi", "_viterbi")]:
                tp, sp = Path(f"{prefix}{suffix}_transitions.csv"), Path(f"{prefix}{suffix}_segments.csv")
                for p in [tp, sp]:
                    if str(p.relative_to(event_dir)) not in receipt["outputs_sha256"]:
                        raise ValueError(f"Unrecorded event result: {p}")
                t, s = report.csv(tp), report.csv(sp)
                identity = {**report.identity(ds, tf), "seed": seed, "method": method,
                            "test_start": source["test_start"], "test_end": source["test_end"]}
                for it in ITERATIONS:
                    for focus, control in TRANSITIONS.items():
                        row = one_row(t, {"iteration": it, "focus": focus}, tp)
                        if row["control"] != control:
                            raise ValueError(f"Wrong control: {tp}, {focus}")
                        f, c, diff = [finite(row[k], k) for k in ["cumret20_focus", "cumret20_control", "cumret20_diff"]]
                        if not np.isclose(f - c, diff, atol=1e-10, rtol=1e-7):
                            raise ValueError(f"Event difference disagrees with means: {tp}")
                        nf, nc = integer(row["n_focus"], "n_focus"), integer(row["n_control"], "n_control")
                        if min(nf, nc) < 1:
                            raise ValueError(f"Empty event/control group: {tp}")
                        observation = {**identity, "iteration": it, "focus": focus, "control": control,
                                       "delta_pct": 100 * diff, "n_focus": nf, "n_control": nc}
                        controls.append({**observation, "cumret20_focus": f, "cumret20_control": c})
                        if focus in FOCUS:
                            transitions.append(observation)
                    row = one_row(s, {"iteration": it, "future_axis": "BULL"}, sp)
                    segments.append({**identity, "iteration": it, "future_axis": "BULL",
                                     "median_duration": finite(row["median_duration"], "median duration", 1),
                                     "n_segments": finite(row["n_segments"], "n_segments", 1)})
    trans = pd.DataFrame(transitions)
    seg = pd.DataFrame(segments)
    subset = trans[(trans.method == "viterbi") & trans.iteration.isin([0, 1, 5])]
    keys = ["dataset", "adjust", "method", "test_start", "test_end", "iteration", "focus", "control"]
    table5 = mean_std(subset, keys, ["delta_pct"])
    counts = subset.groupby(keys, sort=False).agg(n_events_min=("n_focus", "min"), n_events_max=("n_focus", "max"))
    table5 = table5.merge(counts.reset_index(), on=keys, validate="one_to_one")
    table5["mean±std (%)"] = [f"{r.delta_pct_mean:+.2f} ± {r.delta_pct_std:.2f}" for r in table5.itertuples()]
    segstats = mean_std(seg, ["dataset", "adjust", "method", "iteration"], ["median_duration"])
    segstats = segstats.rename(columns={"median_duration_mean": "mean_of_seed_medians",
                                       "median_duration_std": "std_of_seed_medians"})
    tables = {"event_transitions_per_seed.csv": trans, "table5_events.csv": table5,
              "bull_duration_per_seed.csv": seg, "bull_duration_seed_summary.csv": segstats,
              "event_controls_per_seed.csv": pd.DataFrame(controls)}
    return tables


def main():
    p = parser(__doc__)
    p.add_argument("--event-dir", required=True, type=Path, help="Completed prepare_events output")
    args = p.parse_args()
    report = Report(args.config)
    event_dir, output = args.event_dir.resolve(), args.output_dir.resolve()
    if output == event_dir or output.is_relative_to(event_dir) or event_dir.is_relative_to(output):
        raise ValueError("Summary output must be separate from event inputs")
    report.write(output, collect(report, event_dir), "summarize_events", args.dry_run,
                 extra={"event_dir": str(event_dir)})


if __name__ == "__main__":
    main()
