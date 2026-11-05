#!/usr/bin/env bash
# Prepare and train independent GRU probe batches linked to Transformer experiments.
#
# Quick start (from the repository root, after the Transformer batch completes):
#   bash scripts/run_gru_ablation.sh --config experiment.yaml \
#     --source-batch PATH_TO_TRANSFORMER_BATCH
# This prepares a new GRU batch and immediately trains/evaluates it. Beta/seed
# selections default to the source batch. Add --dry-run for a read-only preview.
# Advanced: prepare only freezes a batch; run --batch-dir PATH_TO_GRU_BATCH runs it.
# Add --python /path/to/python; use start --help or run --help for all options.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo '[ERROR] Run this script with bash; do not source it.' >&2
  return 2
fi
set -euo pipefail
ABLATION_MODEL=gru
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/run_ablation.sh"
