#!/usr/bin/env bash
# Generate paper Tables 1-5 and supporting details. Run with bash; do not source.
#
# Quick start (from the repository root):
# Copy configs/reports/paper_results.example.yaml to paper_results.yaml in the
# same directory and fill in the completed Transformer/GRU batch paths.
#   bash scripts/run_paper_reports.sh --config configs/reports/paper_results.yaml \
#     --output-dir outputs/paper_reports/report_001 --dry-run
# Remove --dry-run to generate reports in a new directory. Add --event-dir PATH
# to reuse completed event analysis, or --example-seed N for a verified Figure 1 seed.
# Add --python /path/to/python to select an interpreter; use --help for all options.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo '[ERROR] Run this script with bash; do not source it.' >&2
  return 2
fi
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Select report YAML with --config; the template is configs/reports/paper_results.example.yaml.
# Retain legacy PAPER_* variables; CLI takes precedence, and PYTHON_BIN overrides PAPER_PYTHON.
REPORT_CONFIG="${PAPER_CONFIG:-$REPO_ROOT/configs/reports/paper_results.yaml}"

# Use the active Python environment, create a fresh report directory and run CPU event analysis.
# Environment variables and CLI options can override these runtime settings.
PYTHON_BIN="${PYTHON_BIN:-${PAPER_PYTHON:-python}}"
OUTPUT_BASE="${PAPER_OUTPUT_BASE:-$REPO_ROOT/outputs/paper_reports}"
OUTPUT_DIR="${PAPER_OUTPUT_DIR:-}"
EVENT_DIR="${PAPER_EVENT_DIR:-}"
EXAMPLE_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run_paper_reports.sh [options]
Select report YAML with --config; analysis settings are read from YAML without editing this script.
Generate Tables 1-5, Figures 2/3, per-seed details and provenance records.

  --python PATH       Use Python from an existing environment (or set PYTHON_BIN)
  --config PATH       Report YAML; default: configs/reports/paper_results.yaml
  --output-dir PATH   New report directory; existing directories cannot be overwritten
  --event-dir PATH    Reuse completed prepare_events output with matching provenance
  --example-seed N    Select a verified seed for Figure 1; omit to skip Figure 1
  --example-dataset D Source dataset for Figure 1 (default: SSE50)
  --dry-run          Check inputs without creating a report directory or running event analysis
  -h, --help         Show help

Relative argument paths resolve from the repository root; paths inside YAML remain relative to that YAML file.
Interpreter precedence: --python > PYTHON_BIN > legacy PAPER_PYTHON > python.
Legacy PAPER_* environment variables remain supported; --output-root, --output_root and --dry_run are valid aliases.
Requires completed Transformer/GRU batches; no legacy beta aggregation step is needed.
No model training or changes to experiment statistics; when generating new events, Table 5 is validated after event generation.
EOF
}
fail() { echo "[ERROR] $*" >&2; exit 1; }
DRY_RUN=0
while (( $# > 0 )); do
  case "$1" in
    --python=*|--config=*|--output-dir=*|--output-root=*|--output_root=*|--event-dir=*|--example-seed=*|--example-dataset=*)
      option="${1%%=*}"; value="${1#*=}"; shift; set -- "$option" "$value" "$@" ;;
    -h|--help) usage; exit 0 ;;
    --dry-run|--dry_run) DRY_RUN=1; shift ;;
    --python|--config|--output-dir|--output-root|--output_root|--event-dir|--example-seed|--example-dataset)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires an argument"
      case "$1" in
        --python) PYTHON_BIN="$2" ;;
        --config) REPORT_CONFIG="$2" ;;
        --output-dir|--output-root|--output_root) OUTPUT_DIR="$2" ;;
        --event-dir) EVENT_DIR="$2" ;;
        --example-seed|--example-dataset) EXAMPLE_ARGS+=("$1" "$2") ;;
      esac
      shift 2 ;;
    *) fail "Unknown argument: $1 (use --help for usage)" ;;
  esac
done
cd -- "$REPO_ROOT"
[[ -n "$REPORT_CONFIG" ]] || fail 'Specify report YAML with --config'
[[ -f "$REPORT_CONFIG" ]] || fail "Report YAML not found: $REPORT_CONFIG; copy configs/reports/paper_results.example.yaml, fill in actual batch paths, and select it with --config"
command -v "$PYTHON_BIN" >/dev/null || fail 'Python not found; activate an existing environment or set --python'
export PYTHONUNBUFFERED=1
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${OUTPUT_BASE%/}/run_$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}"
fi
[[ "$OUTPUT_DIR" == /* ]] || OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
if [[ -n "$EVENT_DIR" && "$EVENT_DIR" != /* ]]; then
  EVENT_DIR="$REPO_ROOT/$EVENT_DIR"
fi

TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/ddsl-paper-config.XXXXXX")"
STAGE=preflight
cleanup() {
  local status=$?
  rm -f -- "$TEMP_CONFIG"
  if (( status != 0 )); then
    echo "[ERROR] Stage $STAGE failed; not all tables were completed." >&2
    if [[ -d "$OUTPUT_DIR" ]]; then
      echo "[INFO] Existing files are preserved in: $OUTPUT_DIR; check logs/ if analysis has started." >&2
    else
      echo '[INFO] No report directory was created.' >&2
    fi
  fi
}
trap cleanup EXIT

# Resolve absolute config paths and protect batch/raw/reused-event directories before writing reports.
# Pass arguments through argv; never interpolate user paths into Python code or execute them with eval.
"$PYTHON_BIN" - "$REPO_ROOT" "$TEMP_CONFIG" "$REPORT_CONFIG" "$OUTPUT_DIR" "$EVENT_DIR" \
  "$DRY_RUN" <<'PY'
import copy
from pathlib import Path
import sys

import yaml

repo, temporary, source, output, events, dry = sys.argv[1:]
sys.path.insert(0, str(Path(repo) / "scripts/paper"))
from paper_common import Report

original = Report(source)
cfg = copy.deepcopy(original.cfg)
for spec in cfg["datasets"].values():
    for key in ("transformer_batch", "gru_batch"):
        spec[key] = str(original.path(spec[key]))

if not any(spec["ablation"] for spec in cfg["datasets"].values()):
    raise ValueError("Table 4 requires at least one dataset with ablation: true")
Path(temporary).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
report = Report(temporary)
for dataset, spec in cfg["datasets"].items():
    tf, gru = report.pair(dataset)
    selected_betas = set(cfg["ablation_betas"] if spec["ablation"] else []) | {report.beta}
    for beta in sorted(selected_betas):
        for seed in report.seeds:
            tc, gc = report.cell(tf, beta, seed), report.cell(gru, beta, seed)
            if Path(gc["input_root"]).resolve() != Path(tc["output_root"]).resolve():
                raise ValueError(f"{dataset}: GRU input does not match the Transformer cell")
output = report.check_output(output)
# Protect entire standardized/workstation raw-data roots, not only the dataset directories in manifests.
for name in ("raw", "raw_workstation"):
    raw = (Path(repo) / "data" / name).resolve()
    if output == raw or output.is_relative_to(raw) or raw.is_relative_to(output):
        raise ValueError(f"The report directory must be outside the raw-data directory: {raw}")
if events:
    event_path = Path(events).resolve()
    if output == event_path or output.is_relative_to(event_path) or event_path.is_relative_to(output):
        raise ValueError("The report directory must be separate from the reused event directory")
    if not all((event_path / name).is_file() for name in ("report.json", "COMPLETE")):
        raise ValueError(f"Event analysis is not complete: {event_path}")
else:
    # New event analysis reads and writes parquet; check support in the existing Python environment first.
    from pandas.io.parquet import get_engine
    get_engine("auto")
report.verify_inputs()
print(f"[INFO] Datasets: {', '.join(cfg['datasets'])}; seeds: {cfg['seeds']}")
print(f"[INFO] Ablation datasets: {[d for d, s in cfg['datasets'].items() if s['ablation']]}; beta: {cfg['ablation_betas']}")
full = (set(cfg["datasets"]) == {"SSE50", "CSI300", "CSI500", "NASDAQ100"}
        and set(cfg["seeds"]) == {41, 42, 43}
        and set(cfg["ablation_betas"]) == {f"0{i}" for i in range(1, 9)} | {"inf"}
        and all(cfg["datasets"][d]["ablation"] for d in ("SSE50", "NASDAQ100")))
if not full:
    print("[INFO] These Tables 1-5 cover the selected subset and do not constitute a full paper reproduction.")
print(f"[INFO] Output: {output}")
if dry == "0":
    output.mkdir(parents=True, exist_ok=False)
    (output / "paper_results.yaml").write_bytes(Path(temporary).read_bytes())
    (output / "logs").mkdir()
PY

ACTIVE_CONFIG="$TEMP_CONFIG"
if (( ! DRY_RUN )); then
  ACTIVE_CONFIG="$OUTPUT_DIR/paper_results.yaml"
fi
run_stage() {
  STAGE="$1"
  local script="$2"
  shift 2
  local cmd=("$PYTHON_BIN" -u "$REPO_ROOT/scripts/paper/$script" --config "$ACTIVE_CONFIG" "$@")
  if (( DRY_RUN )); then
    cmd+=(--dry-run)
    printf '[CHECK] %s\n' "$STAGE"
    "${cmd[@]}"
  else
    {
      printf '[RUN] %s\n' "$STAGE"
      printf '[COMMAND] '; printf '%q ' "${cmd[@]}"; printf '\n'
      "${cmd[@]}"
    } 2>&1 | tee "$OUTPUT_DIR/logs/$STAGE.log"
  fi
}

run_stage main summarize_main.py --output-dir "$OUTPUT_DIR/main"
run_stage ablation summarize_ablation.py --output-dir "$OUTPUT_DIR/ablation"
if [[ -z "$EVENT_DIR" ]]; then
  EVENT_DIR="$OUTPUT_DIR/events"
  run_stage prepare_events prepare_events.py --output-dir "$EVENT_DIR"
  if (( DRY_RUN )); then
    echo '[DRY RUN] Main, ablation and event input checks passed; Table 5 will be validated after event data generation. No report directory was created.'
    exit 0
  fi
fi
run_stage event_summary summarize_events.py --event-dir "$EVENT_DIR" --output-dir "$OUTPUT_DIR/event_summary"
if (( DRY_RUN )); then
  echo '[DRY RUN] Tables 1-5 summary checks passed using event data from the specified completed directory. No report directory was created.'
  exit 0
fi

run_stage figures plot_paper.py --main-dir "$OUTPUT_DIR/main" \
  --ablation-dir "$OUTPUT_DIR/ablation" --output-dir "$OUTPUT_DIR/figures" "${EXAMPLE_ARGS[@]}"

STAGE=collect_tables
# Copy all five numerical tables unchanged, recording source paths and hashes without reaggregation or rounding.
"$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR" "$EVENT_DIR" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

repo, output, events = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo / "scripts/paper"))
from paper_common import sha256

output, events = output.resolve(), events.resolve()
stages = {"main": output / "main", "ablation": output / "ablation",
          "events": events, "event_summary": output / "event_summary", "figures": output / "figures"}
receipts = {}
for stage, path in stages.items():
    receipt = path / "report.json"
    digest = sha256(receipt)
    if (path / "COMPLETE").read_text().strip() != digest:
        raise ValueError(f"Stage did not complete successfully or the report was modified: {path}")
    receipts[stage] = {"path": str(receipt), "sha256": digest}

tables = {"table1_configuration.csv": "main", "table2_main.csv": "main",
          "table3_stability.csv": "main", "table4_ablation.csv": "ablation",
          "table5_events.csv": "event_summary"}
(output / "tables").mkdir()
table_sources = {}
for name, stage in tables.items():
    source = stages[stage] / name
    expected = json.loads((stages[stage] / "report.json").read_text())["outputs_sha256"][name]
    if sha256(source) != expected:
        raise ValueError(f"Table verification failed: {source}")
    destination = output / "tables" / name
    shutil.copyfile(source, destination)
    if sha256(destination) != expected:
        raise ValueError(f"Table copy verification failed: {destination}")
    table_sources[name] = {"source": str(source), "sha256": expected}
record = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
          "config": str(output / "paper_results.yaml"),
          "config_sha256": sha256(output / "paper_results.yaml"),
          "runner_sha256": sha256(repo / "scripts/run_paper_reports.sh"),
          "event_dir": str(events), "stage_reports": receipts, "tables": table_sources}
(output / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
(output / "COMPLETE").write_text(sha256(output / "run.json") + "\n")
print(f"[OK] Tables 1-5 data generated: {output / 'tables'}")
PY
