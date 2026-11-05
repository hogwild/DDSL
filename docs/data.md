# Data inputs

[Quick start](../README.md#small-training-example) · [Experiments](experiments.md) · [Reporting](reporting.md)

The public example uses three synthetic CSVs described in
[data/example/README.md](../data/example/README.md). Real market datasets are
not included or downloaded automatically.

## Use your own data

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

Copy the appropriate configuration from `configs/pipeline/`, set `data.raw_dir`
to your CSV directory, and select a dedicated `data.blacklist_path`, for example
`outputs/my_stocks_blacklist.txt`. Review the actual adjustment, symbol universe
and date coverage before training. See [experiment configuration](experiments.md#configuration)
for the remaining settings and commands.

Only `data/example/` is included in Git. Other data directories and the author's
local `data/raw`, `data/raw_icaif2026_paper` and `data/raw_workstation` symlinks
are ignored. You can use your own directories without those links.

## Optional input conversion

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

Optional `--start-date` and `--end-date` bounds filter the original input rows;
they are separate from the downstream train/validation/test splits. Conversion
does not fill missing dates, impute values or select index membership. Turnover
remains in the source units, including percentage values when supplied that way.

## Registering an unchanged historical copy

Separate registration is optional. Standardized historical CSVs can be used
directly; batch preparation saves their current hashes in `data_inventory.json`
and does not infer historical preprocessing from filenames.

If you explicitly provide `--conversion-report` or `--data-registration`, the
selected record must match every input CSV. Use `python scripts/register_data.py --help`
for the optional registration interface. Keep original inputs and existing
experiment outputs unchanged when moving to a new data version, and start the
new experiment from iteration 0 in a fresh output directory.
