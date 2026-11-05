# DDSL: Iterative Self-Labeling for Financial Market States

DDSL combines structured market-state labeling with sequence-model learning.
It uses K/Q/G/M price and turnover tokens, initial dynamic-programming labels,
Transformer training, and model-informed Viterbi refinement. An independent GRU
probe measures how well another model learns the same labels.

This repository contains code, configurations and three small synthetic stock
CSVs for a local example. Real market data, trained models and experiment
outputs are not bundled. Running the example demonstrates execution, not
reproduction of the paper's numerical results.

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

The main outputs are:

| Path under the selected output root | Contents |
| --- | --- |
| `SSE50/pipeline_config/` | Effective Transformer configuration |
| `SSE50/codebook/` | Fitted token representation |
| `SSE50/iter_0/labels/`, `SSE50/iter_0/dataset/` | Labels and token datasets |
| `SSE50/iter_0/model/seed_42/` | Transformer checkpoint and training records |
| `SSE50/iter_0/eval/seed_42/test_metrics.json` | Transformer test metrics |
| `SSE50/iter_0/logits/` | Transformer predictions |
| `SSE50/iter_0/model_gru/seed_42/` | GRU checkpoint, in the GRU output root |
| `SSE50/iter_0/eval_gru/seed_42/test_metrics.json` | GRU test metrics, in the GRU output root |

Launchers print their log paths under `logs/workstation/` or `logs/gru/`.
Use a new output directory for each run, such as `transformer_002`; existing
runs are protected against overwrite. The example writes derived data and its
blacklist under `outputs/`, leaving the bundled CSVs unchanged.

## Data and configuration

For your own dataset, leave the example files unchanged and place standardized
CSVs in a separate folder, such as `data/my_stocks/`. Use UTF-8 and filenames
`{symbol}_{hfq|qfq}.csv`. Expected columns are:

```text
date,symbol,open,high,low,close,volume,amount,turnover
```

The K/Q/G/M representation requires `date,open,high,low,close,turnover`.
Preserve string symbols (including leading zeros), source units and other
available fields. Dates must be ascending and unique per symbol. Use positive
prices and enough valid history in every split for tokenization, 60-step
sequences and labeling windows. Label analysis also needs sufficient segments
per state for Fisher-ratio estimation; very short or uniform inputs may fail
that analysis.

Copy the appropriate unified YAML from `configs/pipeline/` for SSE50, CSI300,
CSI500 or NASDAQ100, then review these settings:

| Setting | What to configure |
| --- | --- |
| `dataset.name`, `dataset.market`, `dataset.adjust` | Dataset identity, market and actual price adjustment |
| `data.raw_dir` | Directory containing your standardized CSVs |
| `data.blacklist_path` | A dedicated path, e.g. `outputs/my_stocks_blacklist.txt` |
| `splits` | Training, validation and test date ranges |
| `pipeline` | Output root, iteration range, seed and worker count |
| `train.training`, `gru.training` | Epochs, batch size and model-specific training settings |

The supplied experiment YAMLs select HFQ; SSE50 uses BF16 training and the other
three use FP32. Choose supported precision for your hardware before starting.
They enable blacklist regeneration and point to shared `configs/symbols/`
files, so set a separate blacklist path for your data. A fresh output directory
alone does not isolate that blacklist. Use a new blacklist path when changing
data or quality-check settings.

Default splits train through 2018-12-31, validate in 2019-2022 and test in
2023-2025. The sequence length is 60, labeling windows are 10/10/20 and token
sizes are K=16, Q=8, G=8, M=4. The unified YAML controls the full pipeline;
`configs/base/` contains separate configurations for individual stages and does
not automatically follow edits to a unified YAML.

Only `data/example/` is included in Git. Other data directories and the author's
local `data/raw`, `data/raw_icaif2026_paper` and `data/raw_workstation` symlinks
are ignored. You can use your own directories without those links.

### Optional input conversion

Already standardized CSVs can be used directly. `scripts/prepare_data.py`
converts supported Chinese workstation exports, with source folders such as
`SSE50_hfq/` and filenames such as `600028.SH.CSV`. It is not a market-data
downloader or a general NASDAQ100 converter. Preview one file first:

```bash
python scripts/prepare_data.py \
  --source-root /path/to/source_exports \
  --output-dir data/converted_001 \
  --datasets SSE50 --adjust hfq --limit 1 --dry-run
```

Remove both `--limit 1` and `--dry-run` for a full conversion. Without filtering
flags, rows are preserved subject to any requested date bounds. The optional
`--drop-suspension-placeholders --suspension-policy complete-kqgm-v3` removes
rows missing any required OHLC/turnover field. Historical policies are available
through `--help`; select the policy required by your experiment explicitly.
The converter preserves source values and units, does not recompute price
adjustment, refuses to overwrite differing files and writes conversion reports.
Review errors and reports before training. Changing filtering or date bounds
creates a new input version and requires a fresh experiment.

## Training and ablation

Save your reviewed configuration as `configs/pipeline/experiment.yaml`, then run:

```bash
bash scripts/run_transformer_pipeline.sh \
  --config configs/pipeline/experiment.yaml \
  --output-dir outputs/pipeline_runs/experiment_001

# Run after the Transformer finishes.
bash scripts/run_gru_pipeline.sh \
  --config configs/pipeline/experiment.yaml \
  --input-dir outputs/pipeline_runs/experiment_001 \
  --output-dir outputs/gru_runs/experiment_001
```

Input/output roots contain the dataset subdirectory. Iteration 0 uses DP labels;
subsequent iterations use the preceding Transformer's predictions for
relabeling. GRU consumes those token datasets and does not provide feedback
logits. A GRU dry run requires the existing input datasets.

Effective stage configurations are saved under `<output_root>/<DATASET>/pipeline_config/`.
Continue a chain only with its original output root, preceding artifacts and
compatible settings. Changed data, tokenization, labeling or precision requires
a new experiment from iteration 0.

### Beta ablation batches

Use a Git checkout with an existing commit for actual batch preparation: batches
record Git identity as well as code, inputs and configurations. A GitHub clone
provides this history; an extracted ZIP needs its own initialized repository
and initial commit. The small single-run example does not require Git history.

```bash
bash scripts/run_transformer_ablation.sh \
  --config configs/pipeline/experiment.yaml \
  --betas 01 02 --seeds 41 42 --output-base outputs/ablation_runs

# After the selected Transformer cells finish, use the printed batch directory.
bash scripts/run_gru_ablation.sh \
  --config configs/pipeline/experiment.yaml \
  --source-batch outputs/ablation_runs/EDIT_TRANSFORMER_BATCH
```

Each command prepares a new batch and immediately trains its cells sequentially.
`--output-base` is the parent of the generated batch directory. GRU inherits the
source beta/seed matrix unless a subset is selected explicitly. Add `--dry-run`
for a preview without creating a batch or starting training. Beta tags `01`
through `08` mean 0.1 through 0.8; `inf` selects the separate pseudo-label
condition, not a numerical infinity for structured decoding.

Separate data registration is optional: batches save the current CSV hashes in
`data_inventory.json`. If supplied, `--conversion-report` or `--data-registration`
must match all selected inputs. Advanced `prepare` stops before training;
`run --batch-dir PATH` executes an existing batch. Use `start --help` or
`run --help` for the complete options.

Batch logs and execution receipts are under `logs/cell_<index>.log` and
`logs/cell_<index>.json` inside the batch. Execution stops on the first failure
and preserves artifacts. Rerunning skips cells with matching success markers,
but refuses unfinished cells with existing outputs/logs; inspect the failure
and prepare a new batch when such artifacts remain. Keep inputs and code stable
throughout a batch and use a persistent terminal such as tmux for long runs.

The paper scope is four datasets, seeds 41/42/43, iterations 0-5 and main beta
0.3. SSE50 and NASDAQ100 additionally use
`--betas 01 02 03 04 05 06 07 08 inf`; CSI300 and CSI500 need `--betas 03` for
the main results. Confirm original inputs and frozen settings before comparing
results with the manuscript.

## Paper tables and figures

Reports require completed, linked Transformer and GRU **batch manifests**.
The small example and plain single-run directories are not substitutes.
Copy `configs/reports/paper_results.example.yaml` to
`configs/reports/paper_results.yaml`, replace every `EDIT_*` batch path and review
the selected datasets, seeds and beta conditions. Paths inside this YAML are
relative to the YAML file; command-line paths are relative to the repository root.

```bash
# Check inputs first.
bash scripts/run_paper_reports.sh \
  --config configs/reports/paper_results.yaml --dry-run

# Generate reports in a new directory.
bash scripts/run_paper_reports.sh \
  --config configs/reports/paper_results.yaml \
  --output-dir outputs/paper_reports/review_001
```

The runner reads saved experiment results, performs event analysis and generates
Tables 1-5, Figures 2/3 and supporting details without training models. Outputs
include `tables/`, `figures/`, `main/`, `ablation/`, `events/`, `event_summary/`,
`logs/` and `run.json`; `COMPLETE` is written only after the full workflow succeeds.
A subset covers only the selected scope. Figure 3 requires SSE50 ablation data.

Figure 1 requires an explicitly verified `--example-seed N` (and optionally
`--example-dataset`, default SSE50). Its example is stock 600760,
2000-03-28 through 2000-11-08, iterations 0/1/3/5; missing inputs cause failure
rather than automatic substitution. Omit the option to skip Figure 1.
Use `--event-dir PATH` to reuse compatible completed event analysis.
A dry run with new events checks inputs but cannot verify Table 5 before those
events are generated. If a report fails, inspect its retained logs and retry in
a new output directory.

Cross-seed summaries use equal weights and sample standard deviations. F1 and
Fisher come from saved metrics; event-minus-control 20-day log-return differences
are computed per seed and reported in percentage points. Keep
`label_scope: existing_test_prefix`: label stability/distribution summaries use
the existing cumulative history, whereas evaluation/event analysis uses its
specified test dates. Do not combine incompatible data versions, symbol sets,
precision settings or metric scopes.

## Project structure

| Path | Purpose |
| --- | --- |
| `src/data/`, `src/states/`, `src/labeling/` | Tokenization, state definitions and labeling |
| `src/models/`, `src/training/` | Transformer/GRU models, training and evaluation |
| `src/pipeline/`, `src/analysis/` | Experiment orchestration and analyses |
| `configs/` | Stage, pipeline, example and report configurations |
| `data/example/` | Three synthetic stock CSVs and their description |
| `scripts/` | Public launchers, data preparation and report entry points |

## Citation and license

The code uses the [MIT license](LICENSE). The copyright holder remains to be
filled in. Final paper citation metadata is pending; `CITATION.cff.example` is
a template, not an active citation record.
