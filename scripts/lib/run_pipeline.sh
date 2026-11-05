# Shared launcher implementation, sourced by the two public pipeline scripts.
set -euo pipefail

fail() { echo "[ERROR] $*" >&2; exit 1; }
usage() {
  echo "Usage: bash scripts/run_${PIPELINE_KIND}_pipeline.sh --config <pipeline.yaml> [options...]"
  echo 'A unified YAML configuration is required; --config=<pipeline.yaml> is also supported.'
  echo 'Preview with --dry-run or --dry_run; no logs or experiment files are created.'
  echo 'Example CLI overrides: --output-dir <new-dir> --seed 42 --start_iter 0 --end_iter 5'
  if [[ "$PIPELINE_KIND" == transformer ]]; then
    echo 'Override Transformer precision explicitly, for example with --train_precision fp32.'
  fi
  echo '--python PATH selects the interpreter and overrides PYTHON_BIN; the default is python.'
  echo '--output-dir PATH selects the output root; GRU --input-dir PATH selects an existing DDSL root.'
  echo 'Relative paths resolve from the repository root; explicit CLI options override YAML, and omitted settings retain their YAML values.'
  echo 'The --output_root, --input_root and --dry_run aliases remain supported.'
  echo 'The four-config mode and legacy experiment environment variables such as DATASET, ADJUST and TRAIN_PRECISION are no longer supported.'
}

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN=0
CONFIG_SEEN=0
PIPELINE_CONFIG=''
PIPELINE_ARGS=()
PIPELINE_ARG_COUNT=0
while (( $# > 0 )); do
  case "$1" in
    --python=*|--python_exec=*|--output-dir=*|--output-root=*|--input-dir=*)
      option="${1%%=*}"; value="${1#*=}"; shift; set -- "$option" "$value" "$@" ;;
    --python|--python_exec)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires an argument"
      PYTHON_BIN="$2"; shift 2 ;;
    --output-dir|--output-root|--input-dir)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires an argument"
      if [[ "$1" == --input-dir ]]; then
        [[ "$PIPELINE_KIND" == gru ]] || fail '--input-dir is only available for the GRU probe'
        PIPELINE_ARGS+=(--input_root "$2")
      else
        PIPELINE_ARGS+=(--output_root "$2")
      fi
      PIPELINE_ARG_COUNT=$((PIPELINE_ARG_COUNT + 2)); shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --dry-run|--dry_run) DRY_RUN=1; shift ;;
    --config)
      (( CONFIG_SEEN == 0 )) || fail '--config may only be specified once'
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail '--config requires a unified YAML file'
      CONFIG_SEEN=1; PIPELINE_CONFIG="$2"; shift 2 ;;
    --config=*)
      (( CONFIG_SEEN == 0 )) || fail '--config may only be specified once'
      CONFIG_SEEN=1; PIPELINE_CONFIG="${1#--config=}"; shift ;;
    *)
      PIPELINE_ARGS+=("$1"); PIPELINE_ARG_COUNT=$((PIPELINE_ARG_COUNT + 1)); shift ;;
  esac
done
(( CONFIG_SEEN )) || fail 'Specify a unified YAML file with --config <pipeline.yaml> (use --help for usage)'

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$PROJECT_ROOT"
[[ -n "$PIPELINE_CONFIG" && -f "$PIPELINE_CONFIG" ]] || fail "Unified YAML file not found: $PIPELINE_CONFIG"
# Reject legacy environment overrides so obsolete settings cannot appear to take effect.
for name in DATASET SEED BETA START_ITER END_ITER N_WORKERS ADJUST OUTPUT_ROOT \
    CODEBOOK_CONFIG LABEL_CONFIG BUILD_CONFIG TRAIN_CONFIG TRAIN_PRECISION; do
  [[ -z "${!name:-}" ]] || fail "Legacy experiment environment variable $name is no longer supported; unset it and use unified YAML or pipeline CLI options (see --help)"
done
export PYTHONUNBUFFERED=1
command -v "$PYTHON_BIN" >/dev/null || fail 'Activate an existing Python environment or set PYTHON_BIN to the absolute interpreter path'

case "$PIPELINE_KIND" in
  transformer) PIPELINE_SCRIPT=src/pipeline/run_transformer_pipeline.py; LOG_GROUP=workstation ;;
  gru) PIPELINE_SCRIPT=src/pipeline/run_gru_pipeline.py; LOG_GROUP=gru ;;
  *) fail "Unknown pipeline: $PIPELINE_KIND" ;;
esac
cmd=("$PYTHON_BIN" -u "$PIPELINE_SCRIPT" --config "$PIPELINE_CONFIG")
# Bash 3.2 with nounset does not allow empty-array expansion.
if (( PIPELINE_ARG_COUNT > 0 )); then
  cmd+=("${PIPELINE_ARGS[@]}")
fi
if (( DRY_RUN )); then
  exec "${cmd[@]}" --dry_run
fi

# Let Python validate output_root/dataset and save source/derived configs before creating it.
# Supported by BSD date (macOS) and GNU date; PID/random suffixes distinguish runs within a second.
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}"
LOG_DIR="logs/$LOG_GROUP/run_$RUN_ID"
mkdir -p -- "logs/$LOG_GROUP"
mkdir -- "$LOG_DIR"

run_pipeline() {
  echo "[INFO] Job started at $(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "[INFO] Project: $PROJECT_ROOT"
  echo "[INFO] Log: $LOG_DIR/pipeline.log"
  echo "[INFO] Git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  printf '[INFO] Command: '; printf '%q ' "${cmd[@]}"; printf '\n'
  "$PYTHON_BIN" -c 'import sys, torch; print("[INFO] Python:", sys.executable); print("[INFO] PyTorch:", torch.__version__, "ROCm:", torch.version.hip)'
  if command -v rocm-smi >/dev/null; then
    rocm-smi --showuse --showmemuse --showpids || echo '[WARN] Unable to read rocm-smi status'
  fi
  "${cmd[@]}"
  echo "[INFO] Job completed at $(date '+%Y-%m-%dT%H:%M:%S%z')"
}

# pipefail preserves Python failures; tee sends the full log to both the terminal and the file.
trap 'status=$?; if (( status != 0 )); then echo "[ERROR] Pipeline exited with status $status; log: $LOG_DIR/pipeline.log" >&2; fi' EXIT
run_pipeline 2>&1 | tee "$LOG_DIR/pipeline.log"
