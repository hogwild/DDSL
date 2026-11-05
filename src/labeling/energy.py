from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from states import NUM_STATES, STATE_NAMES, state_id_to_pair_names, score_column_candidates
from states.grammar import trend_switch_penalty
from .config import Semantic9Config
from common.symbols import normalize_symbol_series, validate_symbol_series


def ewma(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def linear_regression_slope(prices: np.ndarray, start: int, end: int, eps: float = 1e-8, normalize: bool = True) -> float:
    if end <= start:
        return 0.0
    y = prices[start:end + 1].astype(float)
    n = len(y)
    if n <= 1:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < eps:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    if normalize:
        return float(slope / (abs(y_mean) + eps))
    return float(slope)


def window_returns(prices: np.ndarray, start: int, end: int, eps: float = 1e-8) -> np.ndarray:
    if end <= start:
        return np.array([], dtype=float)
    seg = prices[start:end + 1].astype(float)
    return np.diff(seg) / (seg[:-1] + eps)


def compute_flip_rate(returns: np.ndarray, eps: float = 1e-12) -> float:
    if len(returns) <= 1:
        return 0.0
    signs = np.zeros(len(returns), dtype=int)
    signs[returns > eps] = 1
    signs[returns < -eps] = -1
    nz = signs[signs != 0]
    if len(nz) <= 1:
        return 0.0
    return float(np.sum(nz[1:] != nz[:-1]) / (len(nz) - 1))


def compute_pos_ratio(returns: np.ndarray, eps: float = 1e-12) -> float:
    if len(returns) == 0:
        return 0.5
    nz = returns[np.abs(returns) > eps]
    if len(nz) == 0:
        return 0.5
    return float(np.mean(nz > 0))


def trend_scores(slope_value: float, flip_rate: float, pos_ratio: float, cfg: Semantic9Config) -> dict[str, float]:
    bull_score = cfg.w_slope * (slope_value - cfg.tau) - cfg.w_flip * flip_rate + cfg.w_pos * (pos_ratio - 0.5)
    bear_score = cfg.w_slope * (-slope_value - cfg.tau) - cfg.w_flip * flip_rate + cfg.w_pos * (0.5 - pos_ratio)
    neutral_score = -cfg.w_slope * abs(slope_value) + cfg.w_flip_neutral * flip_rate - cfg.w_pos_neutral * abs(pos_ratio - 0.5)
    return {"BULL": bull_score, "BEAR": bear_score, "NEUTRAL": neutral_score}


def build_transition_matrix(cfg: Semantic9Config) -> np.ndarray:
    trans = np.zeros((NUM_STATES, NUM_STATES), dtype=float)
    for prev_id in range(NUM_STATES):
        c_prev, f_prev = state_id_to_pair_names(prev_id)
        for curr_id in range(NUM_STATES):
            c_curr, f_curr = state_id_to_pair_names(curr_id)
            psi_c = trend_switch_penalty(c_prev, c_curr, cfg.lambda_neutral_switch, cfg.lambda_opposite_switch)
            psi_f = trend_switch_penalty(f_prev, f_curr, cfg.lambda_neutral_switch, cfg.lambda_opposite_switch)
            psi_fc = trend_switch_penalty(f_prev, c_curr, cfg.mu_neutral_switch, cfg.mu_opposite_switch)
            trans[prev_id, curr_id] = psi_c + psi_f + psi_fc
    return trans


def compute_emission_scores(price_series: np.ndarray, cfg: Semantic9Config) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    T = len(price_series)
    emissions = np.full((T, NUM_STATES), -1e9, dtype=float)

    diagnostics = {
        "current_slope": np.zeros(T, dtype=float),
        "future_slope": np.zeros(T, dtype=float),
        "current_flip_rate": np.zeros(T, dtype=float),
        "future_flip_rate": np.zeros(T, dtype=float),
        "current_pos_ratio": np.full(T, 0.5, dtype=float),
        "future_pos_ratio": np.full(T, 0.5, dtype=float),
        "current_trend_raw": np.full(T, None, dtype=object),
        "future_trend_raw": np.full(T, None, dtype=object),
    }

    for t in range(T):
        c_start = max(0, t - cfg.current_left_window)
        c_end = min(T - 1, t + cfg.current_right_window)
        f_start = t + 1
        f_end = min(T - 1, t + cfg.future_window)

        if (c_end - c_start + 1) < 2 or (f_end - f_start + 1) < 2:
            continue

        current_slope = linear_regression_slope(price_series, c_start, c_end, eps=cfg.eps, normalize=True)
        future_slope = linear_regression_slope(price_series, f_start, f_end, eps=cfg.eps, normalize=True)
        current_rets = window_returns(price_series, c_start, c_end, eps=cfg.eps)
        future_rets = window_returns(price_series, f_start, f_end, eps=cfg.eps)
        current_flip = compute_flip_rate(current_rets)
        future_flip = compute_flip_rate(future_rets)
        current_pos = compute_pos_ratio(current_rets)
        future_pos = compute_pos_ratio(future_rets)

        diagnostics["current_slope"][t] = current_slope
        diagnostics["future_slope"][t] = future_slope
        diagnostics["current_flip_rate"][t] = current_flip
        diagnostics["future_flip_rate"][t] = future_flip
        diagnostics["current_pos_ratio"][t] = current_pos
        diagnostics["future_pos_ratio"][t] = future_pos

        current_scores = trend_scores(current_slope, current_flip, current_pos, cfg)
        future_scores = trend_scores(future_slope, future_flip, future_pos, cfg)
        diagnostics["current_trend_raw"][t] = max(current_scores, key=current_scores.get)
        diagnostics["future_trend_raw"][t] = max(future_scores, key=future_scores.get)

        for state_id in range(NUM_STATES):
            current_trend, future_trend = state_id_to_pair_names(state_id)
            emissions[t, state_id] = (
                cfg.lambda_current * current_scores[current_trend]
                + cfg.lambda_future * future_scores[future_trend]
            )

    return emissions, diagnostics


def _detect_model_score_columns(score_df: pd.DataFrame, cfg: Semantic9Config) -> tuple[str, list[str]]:
    requested = cfg.model_score_kind.lower().strip()
    kinds = [requested] if requested != "auto" else ["logprob", "prob", "logit"]
    for kind in kinds:
        cols = []
        ok = True
        for state_name in STATE_NAMES:
            found = next((c for c in score_column_candidates(state_name, kind) if c in score_df.columns), None)
            if found is None:
                ok = False
                break
            cols.append(found)
        if ok:
            return kind, cols
    raise ValueError(f"Could not detect model score columns for states {STATE_NAMES}. Available columns: {list(score_df.columns)}")


def _log_softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("model_temperature must be positive.")
    z = logits.astype(float) / temperature
    z = z - np.max(z, axis=1, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=1, keepdims=True) + 1e-12)


def load_model_log_probs(
    score_path_or_df: str | Path | pd.DataFrame,
    target_df: pd.DataFrame,
    cfg: Semantic9Config,
) -> np.ndarray:
    # Read the DataFrame or use the supplied one directly.
    if isinstance(score_path_or_df, pd.DataFrame):
        score_df = score_path_or_df
        score_source = "in-memory DataFrame"
    else:
        score_path = Path(score_path_or_df)
        score_source = str(score_path)
        if not score_path.exists():
            raise FileNotFoundError(f"Model score file not found: {score_path}")
        if score_path.suffix.lower() == ".csv":
            score_df = pd.read_csv(score_path, dtype={cfg.symbol_col: str})
        elif score_path.suffix.lower() in {".parquet", ".pq"}:
            score_df = pd.read_parquet(score_path)
        else:
            raise ValueError(f"Unsupported model score format: {score_path}")

    key_cols = [cfg.symbol_col, cfg.date_col]
    for frame_name, frame in [("score_df", score_df), ("target_df", target_df)]:
        missing = set(key_cols) - set(frame.columns)
        if missing:
            raise ValueError(f"{frame_name} missing columns required for alignment: {missing}")

    market = getattr(cfg, "market", "cn")

    target_keys = target_df[key_cols].copy()
    target_keys[cfg.symbol_col] = normalize_symbol_series(
        target_keys[cfg.symbol_col],
        market=market,
    )
    target_keys[cfg.date_col] = pd.to_datetime(
        target_keys[cfg.date_col],
        errors="coerce",
    )

    score_df = score_df.copy()
    score_df[cfg.symbol_col] = score_df[cfg.symbol_col].astype(str).str.strip()
    validate_symbol_series(
        score_df[cfg.symbol_col],
        market=market,
        name="model_score.symbol",
    )
    score_df[cfg.date_col] = pd.to_datetime(
        score_df[cfg.date_col],
        errors="coerce",
    )
    
    kind, cols = _detect_model_score_columns(score_df, cfg)
    if score_df.duplicated(key_cols).any():
        raise ValueError(f"Duplicated symbol-date keys in {score_source}")

    merged = target_keys.merge(score_df[key_cols + cols], on=key_cols, how="left", validate="one_to_one")
    missing_mask = merged[cols].isna().any(axis=1)
    if missing_mask.any():
        missing_idx = np.flatnonzero(missing_mask.to_numpy())
        n_missing = len(missing_idx)

        # Check where gaps occur, not merely whether they exist.
        #
        # The first seq_len-1 rows of each symbol have no logits: the sliding-window
        # dataset emits targets only for end_pos >= seq_len-1, as does logit export.
        # This is expected for every symbol at iter_1 and later, not an anomaly.
        #
        # Gaps elsewhere mean label and logit dates are misaligned:
        # this is a data error and must raise an exception.
        #
        # Previously cfg.require_model_scores selected whether to raise or fill,
        # without distinguishing these cases. True failed every cell because warm-up gaps
        # always exist; False also concealed real alignment errors.
        if missing_idx[0] != 0 or not np.array_equal(missing_idx, np.arange(n_missing)):
            raise ValueError(
                f"load_model_log_probs: {n_missing}/{len(target_df)} rows have no "
                f"matching model score, and they do not form a leading prefix "
                f"(first missing index {missing_idx[0]}, last {missing_idx[-1]}).\n"
                f"Only the leading seq_len-1 warm-up rows are expected to be missing. "
                f"A gap elsewhere means the logits file and the price data are not "
                f"aligned on dates."
            )

        # Fill a valid warm-up prefix with a uniform prior.
        #
        # For DDSLLabeler, a uniform prior is harmless:
        # add_model_feedback_to_emissions adds the same constant to all nine states at that time.
        # The optimal Viterbi path is unchanged; these rows depend only on E_price + E_grammar.
        #
        # For PseudoLabeler, argmax would always return index 0 for a nine-way tie.
        # PseudoLabeler.label_df detects tied rows and substitutes DP labels separately.
        print(
            f"[INFO] Leading warm-up prefix: {n_missing}/{len(target_df)} rows "
            f"without logits, filled with uniform prior.",
            flush=True,
        )
        if kind == "prob":
            merged.loc[missing_mask, cols] = 1.0 / NUM_STATES
        elif kind == "logprob":
            merged.loc[missing_mask, cols] = -np.log(NUM_STATES)
        else:
            merged.loc[missing_mask, cols] = 0.0

    values = merged[cols].to_numpy(dtype=float)
    if kind == "prob":
        probs = np.clip(values, cfg.eps, 1.0)
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), cfg.eps, None)
        return np.log(probs)
    if kind == "logprob":
        return values
    if kind == "logit":
        return _log_softmax_np(values, cfg.model_temperature)
    raise ValueError(f"Unsupported model score kind: {kind}")


def add_model_feedback_to_emissions(
    emissions: np.ndarray,
    model_log_probs: Optional[np.ndarray],
    cfg: Semantic9Config,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if cfg.beta_model == 0.0:
        return emissions, None
    if model_log_probs is None:
        raise ValueError("beta_model > 0 but model_log_probs is None")
    if model_log_probs.shape != emissions.shape:
        raise ValueError(f"model_log_probs shape {model_log_probs.shape} != emissions shape {emissions.shape}")
    model_term = cfg.beta_model * model_log_probs
    return emissions + model_term, model_term
