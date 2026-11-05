#!/usr/bin/env bash
# Independent GRU probe on existing DDSL datasets; no feedback export.
# Run with bash; see README.md for configuration and examples.
#
# Quick start (from the repository root):
# Use a reviewed unified YAML and an existing Transformer run with token datasets.
#   bash scripts/run_gru_pipeline.sh --config experiment.yaml \
#     --input-dir outputs/transformer_run_001 \
#     --output-dir outputs/gru_run_001 --dry-run
# Input/output roots contain the dataset subdirectory. Remove --dry-run to train
# and evaluate the GRU probe. Add --python /path/to/python or use --help for options.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo '[ERROR] Run this script with bash; do not source it.' >&2
  return 2
fi
set -euo pipefail
PIPELINE_KIND=gru
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/run_pipeline.sh"
