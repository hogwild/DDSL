"""
run_transition_event_study.py
=============================
Event study of PREDICTED joint-state transitions (the external test of the
9-state current x future design).

All triggers and windows are causal and label-free:
  - Events are onsets in the model's MARGINAL FUTURE call: day t is an event
    if pred_future[t] != pred_future[t-1]. The event is typed
    "{current}_TO_{new_future}", where the current axis is read from
    pred_joint at t. The marginal future argmax (pred_future) -- not the
    future axis implied by the 9-class joint argmax (pred_joint) -- defines
    the call, because the two can disagree and the marginal is the
    decision-relevant one. The model saw only history up to t, so the
    trigger is causal.
  - Outcomes are realized RAW cumulative log-returns
    cum_ret(h) = log(C_{t+h} / C_t) for h in [-pre, +post], from raw close
    prices. These are NOT abnormal returns: no expected-return model is
    subtracted. The de-drifting role is carried by (a) the event-minus-
    control difference and (b) the ALL_DAYS unconditional baseline.
  - Test labels are never touched. Label-defined segments encode future
    information (the Viterbi decode saw the future window), so they serve as
    neither trigger nor evaluation window; both come from model predictions
    and raw prices only.

Three layers:
  1. Transition cum_ret curves with same-current-axis controls. Each focus
     transition is paired with a control onset sharing the SAME current axis
     but not calling a reversal, so the difference removes the mechanical
     mean-reversion that follows falls/rises. The controls are matched on the
     current-axis conditioning ONLY -- not on volatility, size, or prior
     return -- so they are "same-current-axis controls", not fully matched
     controls:
        BEAR_TO_BULL     vs BEAR_TO_BEAR
        BULL_TO_BEAR     vs BULL_TO_BULL
        NEUTRAL_TO_BULL  vs NEUTRAL_TO_NEUTRAL
        NEUTRAL_TO_BEAR  vs NEUTRAL_TO_NEUTRAL
     An ALL_DAYS row gives the unconditional drift baseline.
  2. Predicted-segment holding returns. For each maximal run of pred_future
     in {BULL, BEAR, NEUTRAL}: holding return from the onset-day close to the
     EXIT-day close (the first day the model has switched), so a one-day run
     earns one day's return rather than zero. The leading run of each symbol
     (boundary start, not a model switch) and the truncated final run (no
     exit within the sample) are excluded and counted.
  3. Summary at h = post: event-vs-control mean cum_ret difference, a
     directional difference (signed by the focus call), and own-direction
     hit rates for focus and control, with event counts.

Inference caveat (printed and intentional): events of the same symbol have
overlapping outcome windows and same-day events are cross-sectionally
correlated, so the plain SE = std/sqrt(n) understates uncertainty. SEs are
descriptive scale only; no p-values are attached. A symbol-block bootstrap
is left to future work.

Strict failure policy (project convention): unexpected conditions raise.
Permitted exclusions are counted and reported: events without a full
[-pre, +post] price window (head/tail), and for Layer 2 the leading and
truncated runs. A gap between a run's last day and its exit day raises.

Inputs:
  - points parquet files from run_dump_points.py (or the online Viterbi decoder)
    (stride MUST be 1: run-lengths need consecutive days). Must contain
    pred_joint and pred_future (dumps written after the signal update).
  - raw prices, read from the iter_0 label CSVs of --price_root (prices are
    label-independent; iter_0 is used purely as a price source):
      {price_root}/{dataset}/iter_0/labels/{price_split}/*_{adjust}.csv
    Match the exact symbol before the first underscore, as in the prediction
    dump loader. Both {symbol}_{adjust}.csv and legacy
    {symbol}_daily_{adjust}.csv are supported; multiple matches are rejected.

Outputs:
  {out_prefix}_car_curves.csv    iteration, group, h, n, mean_cum_ret, median_cum_ret, se
  {out_prefix}_transitions.csv   per (iteration, focus state): event vs control
  {out_prefix}_segments.csv      per (iteration, future axis): holding stats

Usage:
  python src/analysis/run_transition_event_study.py \
      outputs/outputs_seed42/SSE50/external_anchor/seed42_test_points_iter*.parquet \
      --price_root outputs/outputs_seed42 --dataset SSE50 \
      --adjust hfq --price_split test \
      --out_prefix outputs/outputs_seed42/SSE50/external_anchor/seed42_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATE_NAMES = [
    "BEAR_TO_BEAR", "BEAR_TO_NEUTRAL", "BEAR_TO_BULL",
    "NEUTRAL_TO_BEAR", "NEUTRAL_TO_NEUTRAL", "NEUTRAL_TO_BULL",
    "BULL_TO_BEAR", "BULL_TO_NEUTRAL", "BULL_TO_BULL",
]
TREND_NAMES = ["BEAR", "NEUTRAL", "BULL"]

# focus transition -> matched control (same current axis, no reversal call)
CONTROL_OF = {
    "BEAR_TO_BULL": "BEAR_TO_BEAR",
    "BULL_TO_BEAR": "BULL_TO_BULL",
    "NEUTRAL_TO_BULL": "NEUTRAL_TO_NEUTRAL",
    "NEUTRAL_TO_BEAR": "NEUTRAL_TO_NEUTRAL",
}
FOCUS_STATES = list(CONTROL_OF.keys())
CONTROL_STATES = sorted(set(CONTROL_OF.values()))

REQUIRED_COLS = [
    "dataset", "iteration", "split", "group_by", "horizon", "stride",
    "eval_start_date", "symbol", "date", "pred_joint", "pred_future",
]
META_COLS = ["dataset", "split", "group_by", "horizon", "stride", "eval_start_date"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_points(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Points file not found: {path}")
        df = pd.read_parquet(path)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path}: missing columns {missing}. pred_joint/pred_future "
                f"require dumps written by the updated eval script."
            )
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    for col in META_COLS:
        uniq = sorted(out[col].astype(str).unique())
        if len(uniq) != 1:
            raise ValueError(f"Inconsistent '{col}' across points files: {uniq}")
    if int(out["stride"].iloc[0]) != 1:
        raise ValueError("Event study requires stride=1 points (consecutive days).")

    out["date"] = pd.to_datetime(out["date"])
    if out.duplicated(["iteration", "symbol", "date"]).any():
        raise ValueError("Duplicated (iteration, symbol, date) rows.")

    bad_joint = set(out["pred_joint"].unique()) - set(STATE_NAMES)
    if bad_joint:
        raise ValueError(f"Unknown pred_joint values: {sorted(bad_joint)}")
    bad_fut = set(out["pred_future"].unique()) - set(TREND_NAMES)
    if bad_fut:
        raise ValueError(f"Unknown pred_future values: {sorted(bad_fut)}")

    iters = sorted(out["iteration"].unique())
    counts = out.groupby("iteration").size()
    if counts.nunique() != 1:
        raise ValueError(f"Iterations differ in point counts:\n{counts.to_string()}")
    print(f"[INFO] Loaded {len(out):,} points, iterations {iters}, "
          f"{counts.iloc[0]:,} points each")
    return out


def load_prices(
    price_root: Path, dataset: str, price_split: str, adjust: str,
    symbols: set[str],
) -> dict[str, tuple[dict, np.ndarray]]:
    """Per symbol: ({date -> position}, log_close array) over the FULL price file."""
    d = price_root / dataset / "iter_0" / "labels" / price_split
    if not d.exists():
        raise FileNotFoundError(f"Price source directory not found: {d}")
    # Labeling preserves input filenames. Match the dump loader's symbol and
    # adjustment rules instead of assuming the legacy "_daily_" filename.
    paths_by_symbol = {symbol: [] for symbol in symbols}
    for path in sorted(d.glob(f"*_{adjust}.csv")):
        symbol = path.name.split("_", 1)[0]
        if symbol in paths_by_symbol and path.is_file():
            paths_by_symbol[symbol].append(path)
    out: dict[str, tuple[dict, np.ndarray]] = {}
    for symbol in sorted(symbols):
        paths = paths_by_symbol[symbol]
        if not paths:
            raise FileNotFoundError(f"Price file missing for symbol {symbol}, adjust={adjust}: {d}")
        if len(paths) != 1:
            raise ValueError(f"Multiple price files for symbol {symbol}, adjust={adjust}: {paths}")
        path = paths[0]
        df = pd.read_csv(path, usecols=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].isna().any():
            raise ValueError(f"{path}: unparseable dates.")
        df = df.sort_values("date").reset_index(drop=True)
        if df["date"].duplicated().any():
            raise ValueError(f"{path}: duplicated dates.")
        close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(close)) or np.any(close <= 0):
            raise ValueError(f"{path}: non-finite or non-positive close.")
        pos = {ts: i for i, ts in enumerate(df["date"])}
        out[symbol] = (pos, np.log(close))
    return out


# ---------------------------------------------------------------------------
# Layer 1: cumulative-return matrix around events
# ---------------------------------------------------------------------------

def car_matrix_for_events(
    events: pd.DataFrame,
    prices: dict[str, tuple[dict, np.ndarray]],
    pre: int, post: int,
) -> tuple[np.ndarray, int, int]:
    """CAR(h) matrix (n_events x (pre+post+1)); h=0 column is zero.

    Returns (car, n_tail_excluded, n_head_excluded)."""
    hs = np.arange(-pre, post + 1)
    rows = []
    n_tail = 0
    n_head = 0
    for symbol, date in zip(events["symbol"], events["date"]):
        pos_map, logc = prices[symbol]
        if date not in pos_map:
            raise ValueError(f"Event date {date.date()} of {symbol} not in price file.")
        p = pos_map[date]
        if p + post >= len(logc):
            n_tail += 1
            continue
        if p - pre < 0:
            n_head += 1
            continue
        rows.append(logc[p + hs] - logc[p])
    if rows:
        car = np.vstack(rows)
    else:
        car = np.empty((0, len(hs)))
    return car, n_tail, n_head


def summarize_car(
    car: np.ndarray,
    pre: int,
    post: int,
) -> pd.DataFrame:
    hs = np.arange(-pre, post + 1)
    n = car.shape[0]
    rows = []

    for j, h in enumerate(hs):
        col = car[:, j]
        rows.append({
            "h": int(h),
            "n": n,
            "mean_cum_ret": (
                float(np.mean(col))
                if n else float("nan")
            ),
            "median_cum_ret": (
                float(np.median(col))
                if n else float("nan")
            ),
            "se": (
                float(
                    np.std(col, ddof=1) / np.sqrt(n)
                )
                if n > 1 else float("nan")
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 2: predicted-segment holding returns (future-axis run lengths)
# ---------------------------------------------------------------------------

def segment_holding_stats(
    points_iter: pd.DataFrame,
    prices: dict[str, tuple[dict, np.ndarray]],
) -> pd.DataFrame:
    """Holding return of each predicted future-axis run.

    Semantics: enter at the close of the onset day, hold while the model
    maintains the call, exit at the close of the EXIT day -- the first
    trading day on which the model has switched (run end + 1 bar in the
    price series). A one-day run therefore earns one day's return, not zero.

    Exclusions (counted, never silent):
      - leading run of each symbol: its start is the sample boundary, not a
        model switch, so the entry has no onset semantics;
      - truncated run at the sample end: the model never switched within the
        sample, so the exit day does not exist and the return is undefined.
    A gap between a run's last day and its exit day (missing/suspended day)
    raises rather than being absorbed into the holding return.
    """
    recs = []
    n_leading = 0
    n_truncated = 0
    for symbol, g in points_iter.groupby("symbol", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        fut = g["pred_future"].to_numpy()
        dates = g["date"].to_numpy()
        pos_map, logc = prices[symbol]
        start = 0
        first_run = True
        for i in range(1, len(fut) + 1):
            if i == len(fut) or fut[i] != fut[start]:
                is_leading = first_run
                is_truncated = i == len(fut)

                if is_leading:
                    n_leading += 1

                if is_truncated:
                    n_truncated += 1

                if not is_leading and not is_truncated:
                    d0 = pd.Timestamp(dates[start])       # onset day
                    d_last = pd.Timestamp(dates[i - 1])   # last day in the run
                    d_exit = pd.Timestamp(dates[i])       # first day after switch
                    p0 = pos_map[d0]
                    p_last = pos_map[d_last]
                    p_exit = pos_map[d_exit]
                    # The exit day must be the trading day immediately after the
                    # run's last day. If the points series has a gap here (a
                    # missing/suspended day), p_exit would jump past it and the
                    # holding return would span the gap. Fail loudly instead.
                    if p_exit != p_last + 1:
                        raise ValueError(
                            f"{symbol}: exit day {d_exit.date()} is not the "
                            f"trading day immediately after run end "
                            f"{d_last.date()} (price-series gap? "
                            f"p_last={p_last}, p_exit={p_exit}). Holding return "
                            f"would span a gap."
                        )
                    recs.append({
                        "symbol": symbol,
                        "future_axis": fut[start],
                        "onset": d0,
                        "duration": i - start,
                        "ret": float(logc[p_exit] - logc[p0]),
                    })
                first_run = False
                start = i
    seg = pd.DataFrame(recs)
    if seg.empty:
        raise ValueError("No predicted segments found; check inputs.")

    rows = []
    for axis, g in seg.groupby("future_axis", sort=True):
        r = g["ret"].to_numpy(dtype=float)
        if axis == "BULL":
            hit = float(np.mean(r > 0))
        elif axis == "BEAR":
            hit = float(np.mean(r < 0))
        else:
            hit = float("nan")
        rows.append({
            "future_axis": axis,
            "n_segments": len(g),
            "mean_duration": float(g["duration"].mean()),
            "median_duration": float(g["duration"].median()),
            "mean_ret": float(np.mean(r)),
            "median_ret": float(np.median(r)),
            "hit_rate": hit,
        })
    out = pd.DataFrame(rows)
    out["n_excl_leading"] = n_leading
    out["n_excl_truncated"] = n_truncated
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("points_files", nargs="+",
                   help="*_points_iter{k}.parquet (stride=1, updated dump).")
    p.add_argument("--price_root", required=True,
                   help="Output root whose iter_0 label CSVs supply raw prices.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--adjust", default="hfq")
    p.add_argument("--price_split", default="test",
                   help="Which split's label files to read prices from "
                        "(test files span the full history).")
    p.add_argument("--pre", type=int, default=10,
                   help="Pre-event window length in bars.")
    p.add_argument("--post", type=int, default=20,
                   help="Post-event window length in bars.")
    p.add_argument("--out_prefix", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.pre < 0 or args.post < 1:
        raise ValueError(f"Need pre >= 0 and post >= 1, got {args.pre}, {args.post}")

    points = load_points([Path(p) for p in args.points_files])
    symbols = set(points["symbol"].unique())
    prices = load_prices(Path(args.price_root), args.dataset,
                         args.price_split, args.adjust, symbols)

    iterations = sorted(points["iteration"].unique())
    car_rows, trans_rows, seg_rows = [], [], []

    print("\n[NOTE] Plain SEs ignore overlapping event windows and same-day "
          "cross-sectional correlation; they are descriptive scale only.\n")

    for k in iterations:
        pk = points.loc[points["iteration"] == k]

        # --- onsets: days where the MARGINAL future call changes ----------
        # Event = pred_future[t] != pred_future[t-1] (first row per symbol is
        # a sample boundary, not a switch, and is never an event). The event
        # type is "{current}_TO_{new_future}", where the current axis is read
        # from pred_joint at the onset day; pred_joint's own future axis is
        # deliberately NOT used (joint argmax and marginal argmax can
        # disagree; the marginal is the decision-correct future call).
        onset_frames = []
        for _, g in pk.groupby("symbol", sort=True):
            g = g.sort_values("date").reset_index(drop=True)
            fut = g["pred_future"].to_numpy()
            is_onset = np.zeros(len(g), dtype=bool)
            is_onset[1:] = fut[1:] != fut[:-1]
            ev = g.loc[is_onset].copy()
            cur_axis = ev["pred_joint"].str.split("_TO_").str[0]
            ev["event_state"] = cur_axis + "_TO_" + ev["pred_future"]
            onset_frames.append(ev)
        onsets = pd.concat(onset_frames, ignore_index=True)

        groups: dict[str, pd.DataFrame] = {}
        for state in FOCUS_STATES + CONTROL_STATES:
            groups[state] = onsets.loc[onsets["event_state"] == state]

        # ALL_DAYS unconditional baseline (every sample day is an "event")
        groups["ALL_DAYS"] = pk

        stats = {}
        mats = {}
        for name, ev in groups.items():
            mat, n_tail, n_head = car_matrix_for_events(
                ev, prices, args.pre, args.post)
            summ = summarize_car(mat, args.pre, args.post)
            summ.insert(0, "group", name)
            summ.insert(0, "iteration", k)
            summ["n_tail_excluded"] = n_tail
            summ["n_head_excluded"] = n_head
            car_rows.append(summ)
            stats[name] = summ.set_index("h")
            mats[name] = mat

        # --- event vs control at h = post ---------------------------------
        post_col = args.pre + args.post  # column index of h = +post
        for focus, control in CONTROL_OF.items():
            f, c = stats[focus].loc[args.post], stats[control].loc[args.post]

            def own_direction_hit(
                mat: np.ndarray,
                state: str,
            ) -> float:
                if mat.shape[0] == 0:
                    return float("nan")

                col = mat[:, post_col]
                future_axis = state.split("_TO_")[1]

                if future_axis == "BULL":
                    return float(np.mean(col > 0))

                if future_axis == "BEAR":
                    return float(np.mean(col < 0))

                return float("nan")

            hit_f = own_direction_hit(mats[focus], focus)
            hit_c = own_direction_hit(mats[control], control)
            direction = 1.0 if focus.endswith("_BULL") else -1.0
            trans_rows.append({
                "iteration": k,
                "focus": focus,
                "control": control,
                "n_focus": int(f["n"]),
                "n_control": int(c["n"]),
                f"cumret{args.post}_focus": float(f["mean_cum_ret"]),
                f"cumret{args.post}_control": float(c["mean_cum_ret"]),
                f"cumret{args.post}_diff": float(f["mean_cum_ret"] - c["mean_cum_ret"]),
                "directional_diff": float(
                    direction * (f["mean_cum_ret"] - c["mean_cum_ret"])),
                "hit_rate_focus": hit_f,
                "hit_rate_control": hit_c,
            })
            print(f"[INFO] iter_{k} {focus:<16} n={int(f['n']):>5}  "
                  f"ret{args.post}={f['mean_cum_ret']:+.4f} (hit {hit_f:.2f})  "
                  f"vs {control} ret{args.post}={c['mean_cum_ret']:+.4f} "
                  f"(hit {hit_c:.2f})  diff={f['mean_cum_ret'] - c['mean_cum_ret']:+.4f}")

        # --- layer 2: predicted-segment holding ---------------------------
        seg = segment_holding_stats(pk, prices)
        seg.insert(0, "iteration", k)
        seg_rows.append(seg)

    car_curves = pd.concat(car_rows, ignore_index=True)
    transitions = pd.DataFrame(trans_rows)
    segments = pd.concat(seg_rows, ignore_index=True)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in [("car_curves", car_curves),
                        ("transitions", transitions),
                        ("segments", segments)]:
        path = out_prefix.parent / f"{out_prefix.name}_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    print("\n[INFO] ===== segment holding (per iteration x future axis) =====")
    with pd.option_context("display.width", 220):
        print(segments.to_string(index=False))
    print()
    for name, path in paths.items():
        print(f"[INFO] Saved {name:<12} -> {path}")


if __name__ == "__main__":
    main()
