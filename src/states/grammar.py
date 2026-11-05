from __future__ import annotations

from .current_future_9states import state_id_to_pair_names


def trend_switch_penalty(
    prev_trend: str,
    curr_trend: str,
    neutral_switch_penalty: float,
    opposite_switch_penalty: float,
) -> float:
    """Return a non-positive transition score penalty."""
    if prev_trend == curr_trend:
        return 0.0
    if "NEUTRAL" in (prev_trend, curr_trend):
        return -float(neutral_switch_penalty)
    return -float(opposite_switch_penalty)


def future_current_consistency_penalty(
    prev_state_id: int,
    curr_state_id: int,
    neutral_switch_penalty: float,
    opposite_switch_penalty: float,
) -> float:
    """Penalty for violating future_{t-1} ~= current_t grammar consistency."""
    _, prev_future = state_id_to_pair_names(prev_state_id)
    curr_current, _ = state_id_to_pair_names(curr_state_id)
    return trend_switch_penalty(
        prev_future,
        curr_current,
        neutral_switch_penalty,
        opposite_switch_penalty,
    )
