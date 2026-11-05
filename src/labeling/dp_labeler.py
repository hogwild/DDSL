from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from states import SPEC, STATE_NAMES, state_id_to_pair_names
from common.symbols import (
    normalize_symbol_series,
    normalize_symbol_value,
    validate_symbol_series,
)
from .config import Semantic9Config
from .energy import compute_emission_scores, build_transition_matrix, ewma
from .viterbi import viterbi_decode
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------
# Module-level worker function
# Define it at module scope, outside the class, so ProcessPoolExecutor can pickle it.
# ---------------------------------------------------------------
def _dp_label_one(args: tuple) -> str:
    """
    ProcessPoolExecutor worker for DP labeling of one stock.
    Raise errors directly so they propagate to and terminate the main process.
    """
    cfg_dict, in_path_str, out_path_str, start_date_str, end_date_str = args

    import pandas as pd
    from pathlib import Path
    from labeling.config import Semantic9Config
    from labeling.dp_labeler import DPLabeler

    cfg     = Semantic9Config(**cfg_dict)
    labeler = DPLabeler(cfg)

    start_date = pd.Timestamp(start_date_str) if start_date_str else None
    end_date   = pd.Timestamp(end_date_str)   if end_date_str   else None

    labeler.label_csv_file(
        Path(in_path_str),
        Path(out_path_str),
        start_date=start_date,
        end_date=end_date,
    )
    return in_path_str


def _compute_log_returns(
    out: pd.DataFrame,
    eps: float = 1e-12,
    long_h: int = 5,
) -> None:
    """
    Compute log-return feature columns in place for label analysis.
 
      dC   = log(close_t / close_{t-1})        daily close log return
      gap  = log(open_t  / close_{t-1})        overnight gap log return
      dC5  = log(close_t / close_{t-long_h})   five-day log return
 
    The first row has no prior observation; retain its natural NaN without filling.
    Only close and open are required; both are always present in the label CSV.
    """
    c = pd.to_numeric(out["close"], errors="coerce").astype(float)
    o = pd.to_numeric(out["open"],  errors="coerce").astype(float)
 
    c_prev   = c.shift(1)
    c_prev_h = c.shift(long_h)
 
    out["dC"]          = np.log((c + eps) / (c_prev   + eps))
    out["gap"]         = np.log((o + eps) / (c_prev   + eps))
    out[f"dC{long_h}"] = np.log((c + eps) / (c_prev_h + eps))


class DPLabeler:
    """Pure DP/Viterbi labeler. Outputs string labels, not numeric IDs."""

    def __init__(self, cfg: Semantic9Config):
        self.cfg = cfg

    def label_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        out = df.copy()
        out[cfg.date_col] = pd.to_datetime(out[cfg.date_col])
        out = out.sort_values(cfg.date_col).reset_index(drop=True)

        close = pd.to_numeric(out[cfg.price_col], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(close)):
            close = pd.Series(close).ffill().bfill().to_numpy(dtype=float)

        smooth = ewma(close, cfg.smooth_span)
        emissions, diagnostics = compute_emission_scores(smooth, cfg)
        transition = build_transition_matrix(cfg)
        path, dp_scores = viterbi_decode(emissions, transition)

        self._attach_outputs(out, smooth, path, dp_scores, diagnostics)
        out["ddsl_beta_model"] = 0.0
        return out

    def _attach_outputs(
        self,
        out: pd.DataFrame,
        smooth: np.ndarray,
        path: np.ndarray,
        dp_scores: np.ndarray,
        diagnostics: dict[str, np.ndarray],
    ) -> None:
        cfg = self.cfg
        out["smooth"] = smooth
        out[SPEC.joint_col] = pd.array([STATE_NAMES[int(s)] for s in path], dtype="string")

        decoded_current = []
        decoded_future = []
        for s in path:
            current, future = state_id_to_pair_names(int(s))
            decoded_current.append(current)
            decoded_future.append(future)

        out[SPEC.axis1_col] = pd.array(decoded_current, dtype="string")
        out[SPEC.axis2_col] = pd.array(decoded_future, dtype="string")

        for col in [
            "current_slope",
            "future_slope",
            "current_flip_rate",
            "future_flip_rate",
            "current_pos_ratio",
            "future_pos_ratio",
        ]:
            out[col] = diagnostics[col]

        out["current_trend_raw"] = pd.array(diagnostics["current_trend_raw"], dtype="string")
        out["future_trend_raw"] = pd.array(diagnostics["future_trend_raw"], dtype="string")
        out["dp_best_score"] = np.max(dp_scores, axis=1)

        T = len(smooth)
        complete = np.ones(T, dtype=bool)
        for t in range(T):
            f_end = min(T - 1, t + cfg.future_window)
            if (f_end - (t + 1) + 1) < 2:
                complete[t] = False
        out["label_is_complete"] = complete

        # Log-return features for Fisher Ratio calculations in label analysis
        # The calculation uses only close/open and requires no external files.
        _compute_log_returns(out)

    def label_csv_file(
        self,
        in_path: str | Path,
        out_path: str | Path,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        in_path = Path(in_path)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        market = getattr(self.cfg, "market", "cn")

        df = pd.read_csv(in_path, dtype={self.cfg.symbol_col: str})
        if self.cfg.symbol_col not in df.columns:
            symbol = in_path.stem.split("_", 1)[0]
            df[self.cfg.symbol_col] = normalize_symbol_value(symbol, market=market)
        else:
            df[self.cfg.symbol_col] = normalize_symbol_series(
                df[self.cfg.symbol_col],
                market=market,
            )
        # end_date is applied before labeling to avoid future leakage.
        if end_date is not None:
            dates = pd.to_datetime(df[self.cfg.date_col], errors="coerce")
            df = df.loc[dates <= end_date].reset_index(drop=True)

        if df.empty:
            print(f"Skip {in_path.name}: no data before end_date")
            return pd.DataFrame()

        labeled = self.label_df(df)

        # start_date is applied after labeling, preserving historical context.
        if start_date is not None:
            dates = pd.to_datetime(labeled[self.cfg.date_col], errors="coerce")
            labeled = labeled.loc[dates >= start_date].reset_index(drop=True)
        # Before writing in label_csv_file
        labeled[self.cfg.symbol_col] = normalize_symbol_series(
            labeled[self.cfg.symbol_col],
            market=market,
        )
        validate_symbol_series(
            labeled[self.cfg.symbol_col],
            market=market,
            name=f"{out_path.name}:{self.cfg.symbol_col}",
        )
        labeled.to_csv(out_path, index=False, encoding=self.cfg.output_encoding)
        print(f"labeled -> {out_path} rows={len(labeled)}")
        return labeled

    def label_folder(
        self,
        in_dir: str | Path,
        out_dir: str | Path,
        pattern: str = "*.csv",
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        blacklist: Optional[set[str]] = None, 
        n_workers: int = 1,           
    ) -> None:
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matched {pattern} under {in_dir}")
        
        if blacklist:
            before = len(files)
            files = [f for f in files if f.name.split("_", 1)[0] not in blacklist]
            print(f"[INFO] Blacklist filtered {before - len(files)} files, "
                f"{len(files)} remaining.")
        
        # Serial path (n_workers=1): preserve the original behavior.
        if n_workers == 1 or n_workers == 0:
            for fp in files:
                self.label_csv_file(fp, out_dir / fp.name, start_date=start_date, end_date=end_date)
            return 
        
        # Parallel path
        import dataclasses
        cfg_dict       = dataclasses.asdict(self.cfg)
        start_date_str = str(start_date) if start_date is not None else None
        end_date_str   = str(end_date)   if end_date   is not None else None

        tasks = [
            (cfg_dict, str(fp), str(out_dir / fp.name), start_date_str, end_date_str)
            for fp in files
        ]

        print(f"[INFO] DPLabeler: parallel labeling {len(tasks)} files "
            f"with n_workers={n_workers}")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for result in executor.map(_dp_label_one, tasks):
                pass  # Exceptions propagate to the main process while iterating over executor.map.

