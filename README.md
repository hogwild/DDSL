# DDSL: Iterative Self-Labeling for Financial Market States

DDSL combines structured market-state labeling with sequence-model learning.
It uses K/Q/G/M price and turnover tokens, initial dynamic-programming labels,
Transformer training, and model-informed Viterbi refinement. An independent GRU
probe measures how well another model learns the same labels.

This repository contains code, configurations and three small synthetic stock
CSVs for a local example. Real market data, trained models and experiment
outputs are not bundled. Running the example demonstrates execution, not
reproduction of the paper's numerical results.

**Hardware and reproducibility:** The results reported in the paper's tables
were obtained using an NVIDIA GeForce RTX 3080 Ti GPU on the NYU Shanghai HPC
cluster. Results obtained with different GPUs may differ from those reported
in the paper's tables.

## Installation

Use Python 3.12. You can reuse a suitable environment or create one:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install a PyTorch build suitable for your OS and CPU/GPU using the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then install
and check the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` pins the directly used non-PyTorch packages, including
PyArrow for Parquet files. The preparation environment used Python 3.12.13 and
PyTorch `2.14.0+rocm7.14`; that ROCm build is not a portable installation
requirement. The bundled example has been run on CPU in that existing
environment; a fresh dependency installation on another machine has not been
validated. Save environment versions with your experiments.

Run the commands below from the repository root in the activated environment.
The Bash launchers accept `--python /path/to/python` to select another interpreter
and `--help` for usage. No editable package installation is required.

## Small training example

The three bundled CSVs in `data/example/SSE50/` each contain 780 synthetic rows.
No data download, conversion or registration is needed. See the
[data description](data/example/README.md) for their origin and limitations.

```bash
bash scripts/run_transformer_pipeline.sh \
  --config configs/examples/small_run.yaml \
  --output-dir outputs/examples/transformer_001
```

This runs codebook/token construction, initial DP labeling and label analysis,
one Transformer training epoch, test evaluation and logit export. The example
uses seed 42, iteration 0, batch size 64 and FP32, retaining the SSE50 model,
token definitions, labeling settings and date splits. It does not exercise the
later iterative feedback loop, and its metrics are not paper results.

The pipeline selects an available CUDA/ROCm GPU, then MPS, otherwise CPU. On
Linux, prefix the command with
`CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=` to hide GPUs
for a CPU run. Add `--dry-run` only to preview commands; it does not train or
validate the full execution.

After the Transformer finishes, optionally train and test the independent GRU
on the same token datasets:

```bash
bash scripts/run_gru_pipeline.sh \
  --config configs/examples/small_run.yaml \
  --input-dir outputs/examples/transformer_001 \
  --output-dir outputs/examples/gru_001
```

Transformer test metrics are saved at
`outputs/examples/transformer_001/SSE50/iter_0/eval/seed_42/test_metrics.json`;
GRU metrics are at
`outputs/examples/gru_001/SSE50/iter_0/eval_gru/seed_42/test_metrics.json`.
Use a new output directory for each run. See [outputs and logs](docs/experiments.md#outputs-and-logs)
for checkpoints, derived datasets and log locations.

## Documentation

| Guide | Contents |
| --- | --- |
| [Data](docs/data.md) | Input format, conversion and using your own stock data |
| [Experiments](docs/experiments.md) | Configuration, Transformer/GRU training, beta ablation and failure handling |
| [Reporting](docs/reporting.md) | Completed batch inputs, result summaries, paper tables and figures |

## Project structure

| Path | Purpose |
| --- | --- |
| `src/data/`, `src/states/`, `src/labeling/` | Tokenization, state definitions and labeling |
| `src/models/`, `src/training/` | Transformer/GRU models, training and evaluation |
| `src/pipeline/`, `src/analysis/` | Experiment orchestration and analyses |
| `configs/` | Four dataset pipelines, one small example and one report template |
| `data/example/` | Three synthetic stock CSVs and their description |
| `scripts/` | Public launchers, data preparation and report entry points |
| `docs/` | Data, experiment and reporting guides |

## Citation and license

The code uses the [MIT license](LICENSE). The copyright holder remains to be
filled in. Final paper citation metadata is pending; `CITATION.cff.example` is
a template, not an active citation record.
