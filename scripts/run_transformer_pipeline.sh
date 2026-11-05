#!/usr/bin/env bash
# DDSL labeling, Transformer training/testing and feedback export.
# Run with bash; see README.md for configuration and examples.
#
# Quick start (from the repository root):
# Copy and review a configs/pipeline/*.yaml file as experiment.yaml, including
# data paths, adjustment, precision and a separate blacklist path.
#   bash scripts/run_transformer_pipeline.sh --config experiment.yaml \
#     --output-dir outputs/transformer_run_001 --dry-run
# Remove --dry-run to execute the configured iteration chain.
# Add --python /path/to/python to select an interpreter; use --help for options.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo '[ERROR] Run this script with bash; do not source it.' >&2
  return 2
fi
set -euo pipefail
PIPELINE_KIND=transformer
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/run_pipeline.sh"
