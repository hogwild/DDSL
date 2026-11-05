from __future__ import annotations

from .base import LabelSchemeSpec

# One source of truth for the current/future 9-state system.
# Labeling output should use names, while Viterbi/model arrays use the index
# induced by STATE_NAMES.
TREND_NAMES = ["BEAR", "NEUTRAL", "BULL"]

AXIS1_TO_ID = {"BEAR": 0, "NEUTRAL": 1, "BULL": 2}   # current_trend
AXIS2_TO_ID = {"BEAR": 0, "NEUTRAL": 1, "BULL": 2}   # future_trend

STATE_NAMES = [
    "BEAR_TO_BEAR",
    "BEAR_TO_NEUTRAL",
    "BEAR_TO_BULL",
    "NEUTRAL_TO_BEAR",
    "NEUTRAL_TO_NEUTRAL",
    "NEUTRAL_TO_BULL",
    "BULL_TO_BEAR",
    "BULL_TO_NEUTRAL",
    "BULL_TO_BULL",
]

JOINT_TO_ID = {name: i for i, name in enumerate(STATE_NAMES)}
ID_TO_JOINT = {i: name for name, i in JOINT_TO_ID.items()}
NUM_STATES = len(STATE_NAMES)

SPEC = LabelSchemeSpec(
    name="current_future_9states",
    axis1_col="current_trend",
    axis2_col="future_trend",
    joint_col="joint_state",
    axis1_to_id=AXIS1_TO_ID,
    axis2_to_id=AXIS2_TO_ID,
    joint_to_id=JOINT_TO_ID,
)


def validate_state_name(state_name: str) -> None:
    if state_name not in JOINT_TO_ID:
        raise ValueError(f"Unknown state name: {state_name!r}")


def state_name_to_id(state_name: str) -> int:
    validate_state_name(state_name)
    return JOINT_TO_ID[state_name]


def state_id_to_name(state_id: int) -> str:
    state_id = int(state_id)
    if state_id not in ID_TO_JOINT:
        raise ValueError(f"Unknown state id: {state_id}")
    return ID_TO_JOINT[state_id]


def split_state_name(state_name: str) -> tuple[str, str]:
    validate_state_name(state_name)
    current, future = state_name.split("_TO_")
    return current, future


def state_id_to_pair_names(state_id: int) -> tuple[str, str]:
    return split_state_name(state_id_to_name(state_id))


def state_id_to_pair_ids(state_id: int) -> tuple[int, int]:
    current_name, future_name = state_id_to_pair_names(state_id)
    return AXIS1_TO_ID[current_name], AXIS2_TO_ID[future_name]


def pair_names_to_state_name(current: str, future: str) -> str:
    state_name = f"{current}_TO_{future}"
    validate_state_name(state_name)
    return state_name


def pair_names_to_state_id(current: str, future: str) -> int:
    return state_name_to_id(pair_names_to_state_name(current, future))


def pair_ids_to_state_id(current_id: int, future_id: int) -> int:
    return SPEC.combine_axes_to_joint_id(current_id, future_id)


def score_column_candidates(state_name: str, kind: str) -> list[str]:
    """Supported recognizer-score column names for one state."""
    validate_state_name(state_name)
    if kind == "prob":
        return [f"prob_{state_name}", f"p_{state_name}", f"{state_name}_prob", f"{state_name}_p"]
    if kind == "logprob":
        return [f"logprob_{state_name}", f"logp_{state_name}", f"{state_name}_logprob", f"{state_name}_logp"]
    if kind == "logit":
        return [f"logit_{state_name}", f"logits_{state_name}", f"{state_name}_logit", f"{state_name}_logits"]
    raise ValueError(f"Unsupported score kind: {kind!r}")
