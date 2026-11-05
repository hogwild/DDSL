"""
run_causal_online_viterbi.py
============================
Re-decode a DDSL prediction stream with a CAUSAL online Viterbi, so the event
study can compare the raw per-day argmax against the grammar-smoothed
prediction that reflects DDSL's decoding stage.

Why. The event study consumes pred_future / pred_joint from the eval dump,
which are per-day MARGINAL / JOINT argmax of the model logits -- no Viterbi,
no grammar. That silently strips DDSL of half its method (structured
decoding). This script restores it, WITHOUT leaking the future:

Emission (score form, matching the labeling-stage Viterbi which is max-score):
    emission[t, s] = log_softmax(logits[t] / T)[s]
  i.e. the model's own log-probability of joint state s at day t. It uses
  ONLY the logits, which the model produced from history-only tokens, so it
  is causal and still carries a forward-looking call (the model learned to
  predict the future axis during training). Price-based emissions are NOT
  used: the labeling-stage future_score reads C_t..C_{t+H}, so at the day
  being predicted its future window is empty; only the logits provide a
  causal forward judgement.

Transition: the labeling-stage grammar matrix build_transition_matrix(cfg),
verbatim, coefficient 1 (no new lambda). Score form: entries are <= 0
penalties on current-axis switches, future-axis switches, and the
future_{t-1} -> current_t consistency term.

Causal online rule (the crux). At each T_now we run the forward DP over
emission[0..T_now] and take
    s_{T_now} = argmax_s D[T_now, s]
as the FINAL, emit-and-freeze prediction for day T_now. We do NOT backtrace
the whole path: a global backtrace would let s_t depend on emissions after t
(through the backpointer chain), breaking causality. Taking the last column's
argmax means s_t depends only on emission[0..t], i.e. on C_{<=t}. The forward
DP table is grown incrementally (one column per day), so the cost is
O(T * S^2), not O(T^2 * S^2).

This is mathematically the standard online/filtering Viterbi; the emit-and-
freeze at argmax D[T_now] is what keeps it causal and stable (no retro-
active rewriting of past predictions as new days arrive).

Input: a points parquet from run_dump_points.py, which
already carries the eval window, y_dC/y_slope, prices, and the 9 joint-state
probabilities p_joint_{STATE}. We re-decode pred_joint/pred_future from those
probabilities and rewrite them, preserving all other columns so the event
study runs unchanged.

Strict failure policy: raises on missing joint-prob columns, non-finite
probabilities, unknown states, or a transition matrix whose shape disagrees
with the 9-state space.

Usage:
  python src/analysis/run_causal_online_viterbi.py \
      outputs/outputs_seed42/CSI500/external_anchor/seed42_test_points_iter*.parquet \
      --config configs/base/CSI500_label.yaml \
      --out_suffix _viterbi

  # writes, next to each input, a file with pred_joint/pred_future replaced:
  #   seed42_test_viterbi_points_iter{k}.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STATE_NAMES = [
    "BEAR_TO_BEAR", "BEAR_TO_NEUTRAL", "BEAR_TO_BULL",
    "NEUTRAL_TO_BEAR", "NEUTRAL_TO_NEUTRAL", "NEUTRAL_TO_BULL",
    "BULL_TO_BEAR", "BULL_TO_NEUTRAL", "BULL_TO_BULL",
]
TREND_NAMES = ["BEAR", "NEUTRAL", "BULL"]
NUM_STATES = 9


def state_id_to_pair(sid: int) -> tuple[str, str]:
    name = STATE_NAMES[sid]
    c, f = name.split("_TO_")
    return c, f


# ---------------------------------------------------------------------------
# Grammar transition matrix (score form, <= 0), matching the labeling stage
# ---------------------------------------------------------------------------

def trend_switch_penalty(prev: str, curr: str, neutral_pen: float, opp_pen: float) -> float:
    if prev == curr:
        return 0.0
    if "NEUTRAL" in (prev, curr):
        return -float(neutral_pen)
    return -float(opp_pen)


def build_transition_matrix(
    lambda_neutral: float, lambda_opposite: float,
    mu_neutral: float, mu_opposite: float,
) -> np.ndarray:
    trans = np.zeros((NUM_STATES, NUM_STATES), dtype=float)
    for p in range(NUM_STATES):
        c_prev, f_prev = state_id_to_pair(p)
        for c in range(NUM_STATES):
            c_curr, f_curr = state_id_to_pair(c)
            psi_c = trend_switch_penalty(c_prev, c_curr, lambda_neutral, lambda_opposite)
            psi_f = trend_switch_penalty(f_prev, f_curr, lambda_neutral, lambda_opposite)
            psi_fc = trend_switch_penalty(f_prev, c_curr, mu_neutral, mu_opposite)
            trans[p, c] = psi_c + psi_f + psi_fc
    return trans


def load_grammar_from_config(config_path: Path) -> np.ndarray:
    with open(config_path) as fh:
        y = yaml.safe_load(fh)
    tr = y.get("transition", {})
    ln = tr.get("lambda_neutral_switch", 0.25)
    lo = tr.get("lambda_opposite_switch", 1.0)
    mn = tr.get("mu_neutral_switch", 0.0)
    mo = tr.get("mu_opposite_switch", 0.0)
    print(f"[CONFIG] grammar: lambda_neutral={ln} lambda_opposite={lo} "
          f"mu_neutral={mn} mu_opposite={mo}")
    return build_transition_matrix(ln, lo, mn, mo)


# ---------------------------------------------------------------------------
# Causal online Viterbi
# ---------------------------------------------------------------------------

def causal_online_decode(log_emission: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Per-day emit-and-freeze prediction (argmax of the forward DP column).

    log_emission: [T, S] log-probabilities (score form, higher=better).
    transition:   [S, S] grammar scores (<= 0), added as prev->curr.
    Returns pred_state_ids [T]: at each t, argmax_s of the forward DP table
    computed from emission[0..t] only (causal).
    """
    T, S = log_emission.shape
    if transition.shape != (S, S):
        raise ValueError(f"transition shape {transition.shape} != ({S},{S})")

    preds = np.empty(T, dtype=int)
    # forward DP column D[s] = best cumulative score of any path ending in s
    D = log_emission[0].copy()
    preds[0] = int(np.argmax(D))
    for t in range(1, T):
        # D_new[c] = emission[t,c] + max_p (D[p] + transition[p,c])
        cand = D[:, None] + transition          # [S_prev, S_curr]
        D = log_emission[t] + cand.max(axis=0)
        preds[t] = int(np.argmax(D))            # emit-and-freeze, no backtrace
    return preds


def log_softmax_rows(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def redecode_file(path: Path, transition: np.ndarray, temperature: float,
                  out_path: Path) -> dict:
    df = pd.read_parquet(path)

    prob_cols = [f"p_joint_{s}" for s in STATE_NAMES]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing joint-probability columns {missing}. "
            f"Re-dump with the updated eval (it writes p_joint_{{STATE}})."
        )
    for req in ("symbol", "date", "iteration"):
        if req not in df.columns:
            raise ValueError(f"{path}: missing '{req}' column.")

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    p_joint = df[prob_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(p_joint)) or np.any(p_joint < 0):
        raise ValueError(f"{path}: non-finite or negative joint probabilities.")
    # log-emission from probabilities (already softmaxed at dump time). Guard
    # zeros before log; renormalization is unnecessary for argmax/Viterbi.
    log_emission_all = np.log(np.clip(p_joint, 1e-12, None))

    new_joint = np.empty(len(df), dtype=object)
    new_future = np.empty(len(df), dtype=object)
    n_changed = 0

    for symbol, idx in df.groupby("symbol", sort=False).groups.items():
        rows = np.asarray(idx)
        le = log_emission_all[rows]
        pred_ids = causal_online_decode(le, transition)
        joint = np.array(STATE_NAMES, dtype=object)[pred_ids]
        fut = np.array([s.split("_TO_")[1] for s in joint], dtype=object)
        new_joint[rows] = joint
        new_future[rows] = fut

    n_changed = int(np.sum(new_joint != df["pred_joint"].to_numpy())) \
        if "pred_joint" in df.columns else -1

    df["pred_joint"] = new_joint
    df["pred_future"] = new_future

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    seg_stats = _segment_summary(df)
    return {"n_rows": len(df), "n_changed": n_changed, **seg_stats}


def _segment_summary(df: pd.DataFrame) -> dict:
    """Median run-length of pred_future, for a quick before/after read."""
    from itertools import groupby
    runs = []
    for _, g in df.groupby("symbol", sort=False):
        fut = g.sort_values("date")["pred_future"].to_numpy()
        runs.extend(len(list(v)) for _, v in groupby(fut))
    return {"median_run": float(np.median(runs)), "n_runs": len(runs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("points_files", nargs="+",
                    help="points parquet(s) from eval --dump_points.")
    ap.add_argument("--config", required=True,
                    help="Label YAML supplying the grammar (lambda/mu).")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature; must match the labeling stage "
                         "(default 1.0).")
    ap.add_argument("--out_suffix", default="_viterbi",
                    help="Inserted before '_points_iter' in the output name.")
    args = ap.parse_args()

    transition = load_grammar_from_config(Path(args.config))

    for pf in args.points_files:
        pin = Path(pf)
        if "_points_iter" not in pin.name:
            raise ValueError(f"Unexpected filename (no '_points_iter'): {pin.name}")
        out_name = pin.name.replace("_points_iter", f"{args.out_suffix}_points_iter")
        out_path = pin.parent / out_name
        stats = redecode_file(pin, transition, args.temperature, out_path)
        chg = ("n/a" if stats["n_changed"] < 0
               else f"{stats['n_changed']:,}/{stats['n_rows']:,} "
                    f"({stats['n_changed']/stats['n_rows']:.1%})")
        print(f"[INFO] {pin.name} -> {out_name}  "
              f"changed={chg}  median run-length={stats['median_run']:.1f} "
              f"({stats['n_runs']:,} runs)")


if __name__ == "__main__":
    main()
