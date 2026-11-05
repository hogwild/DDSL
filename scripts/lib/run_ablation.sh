# Shared launcher for the model-specific Python batch CLIs.
set -euo pipefail

fail() { echo "[ERROR] $*" >&2; exit 2; }
usage() {
  cat <<EOF
Usage: bash scripts/run_${ABLATION_MODEL}_ablation.sh --config PATH [options...]

Default: prepare a new batch, then immediately train/evaluate its cells.
Add --dry-run to preview without creating files or starting training.

Advanced commands:
  start    Prepare and run a new batch (the default when no command is given).
  prepare  Validate and freeze a batch without starting training.
  run      Execute pending cells in an existing batch sequentially.
  cell     Execute one manifest-identified cell (advanced use).

Use 'start --help' for all default workflow options, or '<command> --help'.
Use 'prepare --dry-run' or 'run --dry-run' for a read-only preview.
Interpreter precedence: --python > PYTHON_BIN > python in the active environment.
--python PATH and --python=PATH work before or after the command.
Relative paths are resolved from the repository root, regardless of the caller's directory.
All other arguments are forwarded; --dry_run is an alias for --dry-run.
Preparation uses --config and --output-base (the parent of a generated batch directory).
Execution uses --batch-dir; batch logs and completion checks are managed by Python.
GRU preparation also requires --source-batch for its linked Transformer batch.
EOF
}

PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_ARGS=()
ARG_COUNT=0
while (( $# > 0 )); do
  case "$1" in
    --python)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail '--python requires a path'
      PYTHON_BIN="$2"; shift 2 ;;
    --python=*)
      PYTHON_BIN="${1#--python=}"
      [[ -n "$PYTHON_BIN" ]] || fail '--python requires a path'
      shift ;;
    --dry_run) BATCH_ARGS+=(--dry-run); ARG_COUNT=$((ARG_COUNT + 1)); shift ;;
    *) BATCH_ARGS+=("$1"); ARG_COUNT=$((ARG_COUNT + 1)); shift ;;
  esac
done
if (( ARG_COUNT == 0 )); then
  usage >&2
  fail 'Specify --config PATH; use --help for usage'
fi
case "${BATCH_ARGS[0]}" in
  -h|--help) usage; exit 0 ;;
  start|prepare|run|cell) ;;
  --*) BATCH_ARGS=(start "${BATCH_ARGS[@]}") ;;
  *) fail "Unknown command: ${BATCH_ARGS[0]}; use --help for usage" ;;
esac
case "$ABLATION_MODEL" in
  transformer|gru) ;;
  *) fail "Unknown model: $ABLATION_MODEL" ;;
esac

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$PROJECT_ROOT"
command -v "$PYTHON_BIN" >/dev/null || fail 'Activate a Python environment or select --python / PYTHON_BIN'
export PYTHONUNBUFFERED=1
# exec preserves exit status and signals; the wrapper creates no output or log directories.
exec "$PYTHON_BIN" -u "$PROJECT_ROOT/src/pipeline/${ABLATION_MODEL}_ablation_batch.py" "${BATCH_ARGS[@]}"
