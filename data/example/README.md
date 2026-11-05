# Synthetic stock example

These CSVs are generated demonstration inputs, not downloaded market data.
They can be distributed with the project and used to run the normal pipeline
without a data provider account. Follow [the small training example](../../README.md#small-training-example).

| File | Rows | Synthetic identifier |
| --- | ---: | --- |
| `SSE50/990001_hfq.csv` | 780 | 990001 |
| `SSE50/990002_hfq.csv` | 780 | 990002 |
| `SSE50/990003_hfq.csv` | 780 | 990003 |

Each file contains `date,symbol,open,high,low,close,volume,amount,turnover`.
Dates are the first 260 weekdays of each of 2018, 2019 and 2023, providing data
in the existing training, validation and test periods. Exchange holidays are
not modeled. The sequence deliberately skips intervening years to keep the
example small; it is not a continuous market history.

`SSE50` selects an existing pipeline configuration, and `_hfq` matches its
expected filename format. These identifiers do not represent actual securities,
SSE50 membership or genuinely adjusted market prices. Labels and model outputs
are generated during execution. This data is suitable for checking execution,
not measuring predictive quality or reproducing paper results.

## Generation recipe

The files were generated deterministically using Python's standard library,
without random sampling or external inputs. For each stock, let the row index
`i` start at zero. The `(phase_shift, base_price)` pairs are `(0, 80)`,
`(10, 100)` and `(20, 120)` in the table's order. Set `t = i + phase_shift`.
Start a cumulative log trend at zero and update it on each row by `0.006` when
`t % 120 < 30`, by `-0.006` when `60 <= t % 120 < 90`, and by zero otherwise.
Then calculate:

```python
close = base_price * exp(cumulative_log_trend + 0.0001 * sin(t))
open_price = close * exp(0.004 * cos(t / 3))
high = max(open_price, close) * (1.01 + 0.001 * sin(t / 4))
low = min(open_price, close) * (0.99 - 0.001 * cos(t / 5))
volume = 100000 + 20000 * sin(t / 5)
amount = close * volume
turnover = 1 + 0.5 * cos(t / 7)
```

Numeric fields use ten decimal places. Alternating upward, flat and downward
periods provide varied input for the unchanged labeling and analysis stages.
The example configuration writes its blacklist, models and derived datasets
outside this directory.
