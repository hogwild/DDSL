#!/usr/bin/env bash
# Prepare and train Transformer beta-ablation batches in one command.
#
# Quick start (from the repository root, using a reviewed unified YAML):
#   bash scripts/run_transformer_ablation.sh --config experiment.yaml \
#     --betas 01 02 03 04 05 06 07 08 inf --seeds 41 42 43 \
#     --output-base outputs/ablation_runs
# This prepares a new batch and immediately starts training. Add --dry-run for a
# read-only preview. The inf tag selects pseudo-labeling.
# Advanced: prepare only freezes a batch; run --batch-dir PATH executes that batch.
# Add --python /path/to/python; use start --help or run --help for all options.
# Current CSVs are inventoried automatically; no registration is required.
# Optional --data-registration or --conversion-report strictly checks a chosen record.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo '[ERROR] Run this script with bash; do not source it.' >&2
  return 2
fi
set -euo pipefail
ABLATION_MODEL=transformer
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/run_ablation.sh"
