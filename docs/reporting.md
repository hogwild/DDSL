# Experiment reports and paper figures

[Quick start](../README.md#small-training-example) · [Data](data.md) · [Experiments](experiments.md)

## Inputs and commands

Run commands from the repository root in the installed Python environment.
Use `--python /path/to/python` to select another interpreter.

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

## Outputs

The runner reads saved experiment results, performs event analysis and generates
Tables 1-5, Figures 2/3 and supporting details without training models. Outputs
include `tables/`, `figures/`, `main/`, `ablation/`, `events/`, `event_summary/`,
`logs/` and `run.json`; `COMPLETE` is written only after the full workflow succeeds.
A subset covers only the selected scope. Figure 3 requires SSE50 ablation data.

The figure stage exports Figure 2 in both combined and separate forms, each as
PDF and PNG:

- `figure2_cross_market`: the original two-panel figure.
- `figure2a_learnability`: Transformer macro F1 across iterations.
- `figure2b_price_separation`: segment-level Fisher Ratio across iterations.

The standalone panels use 4 × 3 inch canvases before the export crop and share
the combined figure's dataset colors, iteration coordinates, means and ±1 SD
bands. They are generated automatically from the same `main_mean_std.csv`;
no extra command-line option or example seed is required.

## Figure 1 and event reuse

Figure 1 requires an explicitly verified `--example-seed N` (and optionally
`--example-dataset`, default SSE50). Its example is stock 600760,
2000-03-28 through 2000-11-08, iterations 0/1/3/5; missing inputs cause failure
rather than automatic substitution. Omit the option to skip Figure 1.
Use `--event-dir PATH` to reuse compatible completed event analysis.
A dry run with new events checks inputs but cannot verify Table 5 before those
events are generated. If a report fails, inspect its retained logs and retry in
a new output directory.

When `--example-seed` is supplied, the figure stage also exports
`figure1_stock_example_poster.pdf` and `.png`, using the same verified source
data as the paper version. This poster variant has no title, puts the nine-state
legend below the four panels in three columns, and uses 18 pt labels and a fixed
**38.9 × 19.9 cm** canvas including
the legend and axes. Place the stock description in the poster caption. It is
saved without a tight crop so that the intended dimensions are preserved.

To generate figures from existing completed reports, use a new output directory:

```bash
python scripts/paper/plot_paper.py \
  --config configs/reports/paper_results.yaml \
  --main-dir outputs/paper_reports/review_001/main \
  --ablation-dir outputs/paper_reports/review_001/ablation \
  --output-dir outputs/paper_reports/figures_with_poster_001 \
  --example-dataset SSE50 --example-seed VERIFIED_SEED
```

Replace `VERIFIED_SEED` with the confirmed source seed. The existing paper
figures are exported alongside the new poster variant; no training is run.

## Statistical scope

Cross-seed summaries use equal weights and sample standard deviations. F1 and
Fisher come from saved metrics; event-minus-control 20-day log-return differences
are computed per seed and reported in percentage points. Keep
`label_scope: existing_test_prefix`: label stability/distribution summaries use
the existing cumulative history, whereas evaluation/event analysis uses its
specified test dates. Do not combine incompatible data versions, symbol sets,
precision settings or metric scopes.
