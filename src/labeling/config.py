from __future__ import annotations

from dataclasses import dataclass
from common.config_utils import infer_market

@dataclass
class Semantic9Config:
    
    # input columns
    date_col: str = "date"
    symbol_col: str = "symbol"
    price_col: str = "close"

    market: str = "cn"
    
    # smoothing and windows
    smooth_span: int = 10
    current_left_window: int = 10
    current_right_window: int = 10
    future_window: int = 20

    # price/semantic emission weights
    tau: float = 0.0005
    lambda_current: float = 1.0
    lambda_future: float = 1.2

    # DDSL recognizer feedback. beta_model=0 recovers pure DP labeling.
    beta_model: float = 0.0
    model_score_kind: str = "auto"  # auto | prob | logprob | logit
    model_temperature: float = 1.0
    require_model_scores: bool = True

    # trend scoring
    w_slope: float = 1.0
    w_flip: float = 0.8
    w_pos: float = 0.6
    w_flip_neutral: float = 0.4
    w_pos_neutral: float = 0.6

    # transition grammar penalties
    lambda_neutral_switch: float = 0.25
    lambda_opposite_switch: float = 1.0
    mu_neutral_switch: float = 0.0
    mu_opposite_switch: float = 0.0

    output_encoding: str = "utf-8-sig"
    eps: float = 1e-8


def cfg_from_yaml(y: dict) -> Semantic9Config:
    return Semantic9Config(
        date_col=y.get("columns", {}).get("date_col", Semantic9Config.date_col),
        symbol_col=y.get("columns", {}).get("symbol_col", Semantic9Config.symbol_col),
        price_col=y.get("columns", {}).get("price_col", Semantic9Config.price_col),
        market = infer_market(y),
        smooth_span=y.get("smoothing", {}).get("smooth_span", Semantic9Config.smooth_span),
        current_left_window=y.get("windows", {}).get("current_left", Semantic9Config.current_left_window),
        current_right_window=y.get("windows", {}).get("current_right", Semantic9Config.current_right_window),
        future_window=y.get("windows", {}).get("future", Semantic9Config.future_window),
        tau=y.get("scoring", {}).get("tau", Semantic9Config.tau),
        lambda_current=y.get("scoring", {}).get("lambda_current", Semantic9Config.lambda_current),
        lambda_future=y.get("scoring", {}).get("lambda_future", Semantic9Config.lambda_future),
        beta_model=y.get("ddsl", {}).get("beta_model", Semantic9Config.beta_model),
        model_score_kind=y.get("ddsl", {}).get("model_score_kind", Semantic9Config.model_score_kind),
        model_temperature=y.get("ddsl", {}).get("model_temperature", Semantic9Config.model_temperature),
        require_model_scores=y.get("ddsl", {}).get("require_model_scores", Semantic9Config.require_model_scores),
        w_slope=y.get("scoring", {}).get("w_slope", Semantic9Config.w_slope),
        w_flip=y.get("scoring", {}).get("w_flip", Semantic9Config.w_flip),
        w_pos=y.get("scoring", {}).get("w_pos", Semantic9Config.w_pos),
        w_flip_neutral=y.get("scoring", {}).get("w_flip_neutral", Semantic9Config.w_flip_neutral),
        w_pos_neutral=y.get("scoring", {}).get("w_pos_neutral", Semantic9Config.w_pos_neutral),
        lambda_neutral_switch=y.get("transition", {}).get("lambda_neutral_switch", Semantic9Config.lambda_neutral_switch),
        lambda_opposite_switch=y.get("transition", {}).get("lambda_opposite_switch", Semantic9Config.lambda_opposite_switch),
        mu_neutral_switch=y.get("transition", {}).get("mu_neutral_switch", Semantic9Config.mu_neutral_switch),
        mu_opposite_switch=y.get("transition", {}).get("mu_opposite_switch", Semantic9Config.mu_opposite_switch),
        output_encoding=y.get("output", {}).get("encoding", Semantic9Config.output_encoding),
    )
