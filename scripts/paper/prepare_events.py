"""Compute DDSL event-study inputs in a new directory, using frozen batch paths.

CPU analysis only. Reuses the existing dump, online Viterbi and event definitions.
No model training, TB, IC, codebook fitting or changes to experiment outputs.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import shlex
import subprocess
import sys

from paper_common import ITERATIONS, REPO, Report, parser, sha256


def plan(report, output):
    jobs, sources = [], []
    for ds in report.cfg["datasets"]:
        tf, _ = report.pair(ds, need_gru=False)
        adjust = tf["manifest"]["dataset"]["adjust"]
        split = tf["cfg"]["splits"]["test"]
        if not split["start_date"] or not split["end_date"]:
            raise ValueError("Event analysis requires explicit test start/end dates")
        horizon = tf["cfg"]["labeling"]["windows"]["future"]
        if horizon != 20:
            raise ValueError("This paper's event table requires future horizon=20")
        for seed in report.seeds:
            cell = report.cell(tf, report.beta, seed)
            root = Path(cell["output_root"])
            label_path = root / ds / "pipeline_config/label.yaml"
            label = report.yaml(label_path)
            if label["transition"] != tf["cfg"]["labeling"]["transition"]:
                raise ValueError(f"Cell grammar differs from batch: {label_path}")
            temperature = tf["cfg"]["labeling"]["ddsl"]["model_temperature"]
            if label["ddsl"]["model_temperature"] != temperature:
                raise ValueError(f"Cell temperature differs from batch: {label_path}")
            for it in ITERATIONS:
                report.track(root / ds / f"iter_{it}/logits/test/ALL_tokens_{adjust}.parquet")
                paths = sorted((root / ds / f"iter_{it}/labels/test").glob(f"*_{adjust}.csv"))
                if not paths or len({p.name.split("_", 1)[0] for p in paths}) != len(paths):
                    raise ValueError(f"Empty or duplicate-symbol labels: {root}, iter{it}")
                for path in paths:
                    report.track(path)
            prefix = output / ds / f"seed{seed}" / f"seed{seed}_test"
            points = [f"{prefix}_points_iter{it}.parquet" for it in ITERATIONS]
            decoded = [f"{prefix}_viterbi_points_iter{it}.parquet" for it in ITERATIONS]
            py = [sys.executable, "-u"]
            dump = py + [str(REPO / "src/analysis/run_dump_points.py"), "--output_root", str(root),
                         "--dataset", ds, "--iterations", ",".join(map(str, ITERATIONS)), "--adjust", adjust,
                         "--horizon", "20", "--split", "test", "--stride", "1", "--eval_start_date",
                         split["start_date"], "--eval_end_date", split["end_date"], "--out_prefix", str(prefix)]
            decode = py + [str(REPO / "src/analysis/run_causal_online_viterbi.py"), *points,
                           "--config", str(label_path), "--temperature", str(temperature), "--out_suffix", "_viterbi"]
            event_args = ["--price_root", str(root), "--dataset", ds, "--adjust", adjust,
                          "--price_split", "test", "--pre", "10", "--post", "20"]
            for name, cmd in [("dump", dump),
                              ("raw", py + [str(REPO / "src/analysis/run_transition_event_study.py"),
                                             *points, *event_args, "--out_prefix", str(prefix)]),
                              ("decode", decode),
                              ("viterbi", py + [str(REPO / "src/analysis/run_transition_event_study.py"),
                                                 *decoded, *event_args, "--out_prefix", f"{prefix}_viterbi"])]:
                jobs.append({"dataset": ds, "seed": seed, "stage": name, "command": cmd,
                             "log": str(prefix.parent / f"{name}.log")})
            sources.append({**report.identity(ds, tf), "seed": seed, "beta_tag": report.beta,
                            "prefix": str(prefix.relative_to(output)), "iterations": ITERATIONS,
                            "test_start": split["start_date"], "test_end": split["end_date"],
                            "horizon": horizon, "pre": 10, "temperature": temperature,
                            "cell_output_root": str(root), "label_config": str(label_path)})
    # Record the analysis implementation actually used, independently of training code.
    for path in (REPO / "src").rglob("*.py"):
        report.track(path)
    return jobs, sources


def run(args):
    report = Report(args.config)
    output = args.output_dir.resolve()
    jobs, sources = plan(report, output)
    report.check_output(output)
    report.verify_inputs()
    for job in jobs:
        print(shlex.join(job["command"]), flush=True)
    if args.dry_run:
        print("[DRY RUN] Inputs verified; no files written or subprocesses run.")
        return
    output.mkdir(parents=True, exist_ok=False)
    receipt = report.metadata("prepare_events")
    receipt.update(sources=sources, jobs=jobs)
    (output / "plan.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    for job in jobs:
        log = Path(job["log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {job['dataset']} seed{job['seed']} {job['stage']} -> {log}", flush=True)
        try:
            with log.open("w") as f:
                subprocess.run(job["command"], cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] {job['dataset']} seed{job['seed']} {job['stage']} failed "
                  f"(exit {exc.returncode}); full log: {log}", file=sys.stderr, flush=True)
            try:
                with log.open(encoding="utf-8", errors="replace") as f:
                    tail = "".join(deque(f, maxlen=60))
                print("[ERROR] Last 60 lines of the stage log:\n" + tail,
                      file=sys.stderr, flush=True)
            except OSError as log_error:
                print(f"[ERROR] Could not read stage log: {log_error}", file=sys.stderr, flush=True)
            raise
    for source in sources:
        prefix = output / source["prefix"]
        for suffix in ["", "_viterbi"]:
            expected = [Path(f"{prefix}{suffix}_{kind}.csv") for kind in ["transitions", "segments", "car_curves"]]
            expected += [Path(f"{prefix}{suffix}_points_iter{it}.parquet") for it in ITERATIONS]
            for path in expected:
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Event command did not produce a nonempty artifact: {path}")
    report.verify_inputs()
    receipt["outputs_sha256"] = {str(p.relative_to(output)): sha256(p)
                                 for p in sorted(output.rglob("*")) if p.is_file() and p.name != "plan.json"}
    (output / "report.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    (output / "COMPLETE").write_text(sha256(output / "report.json") + "\n")
    print(f"[OK] Event study prepared: {output}")


if __name__ == "__main__":
    run(parser(__doc__).parse_args())
