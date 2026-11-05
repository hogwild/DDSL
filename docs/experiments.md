# Training and experiments

[Quick start](../README.md#small-training-example) · [Data](data.md) · [Reporting](reporting.md)

Run commands from the repository root in the environment described in the
[installation instructions](../README.md#installation). Public Bash launchers
accept `--python /path/to/python` and `--help`.

## Configuration

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
sizes are K=16, Q=8, G=8, M=4. The unified YAML controls the full pipeline and
contains both Transformer and GRU settings. The public project includes six
YAMLs: four dataset pipelines, the small example and the report template.

Stage YAMLs are generated automatically inside each run: Transformer writes
`<output_root>/<DATASET>/pipeline_config/{codebook,label,build,train}.yaml`;
GRU writes `train_gru.yaml` under
`<output_root>/<DATASET>/gru_pipeline_config/seed_<S>/iter_<start>_<end>/`.
The old separate stage and beta-specific templates are no longer distributed.
Beta sweeps select weights through `--betas` below.

For advanced standalone stage commands, use an appropriate generated stage
YAML or an explicitly supplied custom stage YAML, not the unified pipeline
YAML. The Python Transformer runner still accepts all four explicit
`--codebook_config`, `--label_config`, `--build_config` and `--train_config`
paths together; the GRU runner accepts an explicit `--train_config`. Neither
falls back to a bundled stage file. The public Bash launchers use `--config`.

Prepare your inputs as described in [the data guide](data.md). Save the reviewed
YAML as `configs/pipeline/experiment.yaml`; this filename is a user-created
configuration, not a bundled experiment.

## Transformer and GRU

Using the reviewed configuration, run:

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

## Beta ablation batches

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

## Outputs and logs

For the [small example](../README.md#small-training-example), outputs are:

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

For full experiments, substitute the configured dataset, iteration and seed
in these paths. GRU reads the existing Transformer token datasets; it does not
fit a new codebook or relabel data. Complete paired batches can be passed to
the [report runner](reporting.md).
