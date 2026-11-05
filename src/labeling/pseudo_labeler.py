# src/labeling/pseudo_labeler.py
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from states import SPEC, STATE_NAMES
from common.symbols import normalize_symbol_value

from .ddsl_labeler import DDSLLabeler
from .dp_labeler import DPLabeler, _compute_log_returns
from .energy import ewma


# ---------------------------------------------------------------
# Module-level worker function for the PseudoLabeler parallel path only
# ---------------------------------------------------------------
def _pseudo_label_one(args: tuple) -> str:
    (
        cfg_dict,
        in_path_str,
        out_path_str,
        start_date_str,
        end_date_str,
        use_dp,
        symbol,          # Symbol string for this stock
        score_file_str,  # Parquet file path string, or None
    ) = args

    import pandas as pd
    from pathlib import Path
    from labeling.config import Semantic9Config
    from labeling.dp_labeler import DPLabeler
    from labeling.pseudo_labeler import PseudoLabeler

    cfg = Semantic9Config(**cfg_dict)
    start_date = pd.Timestamp(start_date_str) if start_date_str else None
    end_date = pd.Timestamp(end_date_str) if end_date_str else None

    if use_dp:
        labeler = DPLabeler(cfg)
        labeler.label_csv_file(
            Path(in_path_str),
            Path(out_path_str),
            start_date=start_date,
            end_date=end_date,
        )
    else:
        labeler = PseudoLabeler(cfg)

        if score_file_str is None:
            raise ValueError(
                f"{symbol!r}: score_file_str is None in _pseudo_label_one. "
                "PseudoLabeler requires model logits."
            )

        score_df = pd.read_parquet(score_file_str)
        sym_score_df = score_df[
            score_df[cfg.symbol_col] == symbol
        ].reset_index(drop=True)

        if sym_score_df.empty:
            raise ValueError(
                f"{symbol!r} is in codebook but has no logits in {score_file_str}.\n"
                f"Fix: add {symbol!r} to the blacklist and re-run from codebook fitting."
            )

        labeler.label_csv_file(
            Path(in_path_str),
            Path(out_path_str),
            start_date=start_date,
            end_date=end_date,
            model_score_path=sym_score_df,
        )

    return in_path_str


class PseudoLabeler(DDSLLabeler):
    """
    Pseudo-label baseline: assigns labels via argmax over previous-iteration
    logits, bypassing Viterbi decoding and all hard constraints.

    Inherits label_csv_file from DDSLLabeler unchanged, so logit loading and
    file I/O work identically.

    label_df is overridden: instead of running Viterbi, takes the argmax of
    model_log_probs directly.

    label_folder is overridden: parallel path uses _pseudo_label_one (which
    instantiates PseudoLabeler in each worker) instead of _ddsl_label_one.

    Used to demonstrate that Viterbi + hard constraints in DDSL are necessary
    to prevent label collapse.

    Warm-up rows: each symbol's leading seq_len-1 rows have no logits, because
    the sliding-window dataset only produces targets from end_pos >= seq_len-1
    and export_logits inherits that. load_model_log_probs fills them with a
    uniform prior, and argmax over 9 tied values returns index 0 for every one
    of them -- fabricating a constant STATE_NAMES[0] run at the start of each
    symbol. label_df below detects those rows and falls back to the DP label,
    which is exactly what DDSLLabeler effectively does there (a uniform prior
    is a constant shift across states and leaves the Viterbi path unchanged).
    """

    # argmax requires logits regardless of beta_model. See the attribute description
    # in DDSLLabeler.
    needs_model_scores: bool = True

    # Tolerance for detecting nine equal log-probabilities in a row. Uniform-prior
    # filling gives exact ties: load_model_log_probs inserts 0.0 logits, and
    # _log_softmax_np maps each state to -log(9+1e-12). Only floating-point error
    # needs tolerance; a trained classifier is extremely unlikely to produce nine equal logits.
    #
    # Do not count rows using seq_len: that would duplicate a configuration-dependent constant.
    # Detect whether the model provided information; this adapts automatically to seq_len.
    TIE_TOL: float = 1e-12

    def label_df(
        self,
        df: pd.DataFrame,
        model_log_probs: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if model_log_probs is None:
            raise ValueError(
                "PseudoLabeler.label_df: model_log_probs is required. "
                "Pseudo-labeling needs previous-iteration logits."
            )

        if model_log_probs.shape[1] != len(STATE_NAMES):
            raise ValueError(
                f"PseudoLabeler.label_df: model_log_probs has "
                f"{model_log_probs.shape[1]} columns, "
                f"expected {len(STATE_NAMES)} (one per state)."
            )

        if model_log_probs.shape[0] != len(df):
            raise ValueError(
                f"PseudoLabeler.label_df: model_log_probs has "
                f"{model_log_probs.shape[0]} rows but df has {len(df)} rows."
            )

        out = df.copy()
        out[self.cfg.date_col] = pd.to_datetime(out[self.cfg.date_col])
        out = out.sort_values(self.cfg.date_col).reset_index(drop=True)

        # Keep smooth consistent with DPLabeler._attach_outputs for seg_slope in metrics_single.py.
        close = pd.to_numeric(out[self.cfg.price_col], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(close)):
            close = pd.Series(close).ffill().bfill().to_numpy(dtype=float)
        out["smooth"] = ewma(close, self.cfg.smooth_span)

        # argmax: no Viterbi, no E_price, no E_grammar
        pred_ids = np.argmax(model_log_probs, axis=1)
        joint_states = [STATE_NAMES[i] for i in pred_ids]

        out["joint_state"] = joint_states
        out["current_trend"] = [s.split("_TO_")[0] for s in joint_states]
        out["future_trend"] = [s.split("_TO_")[1] for s in joint_states]
        out["pseudo_argmax"] = True  # Record the source so analyses can distinguish it.

        # ---- Use DP labels for warm-up rows. ----
        tied = np.ptp(model_log_probs, axis=1) <= self.TIE_TOL

        if tied.all():
            raise ValueError(
                f"PseudoLabeler.label_df: all {len(out)} rows have tied "
                f"log-probs. The logits cover none of this symbol's dates -- "
                f"that is a coverage failure, not warm-up."
            )

        if tied.any():
            tied_idx = np.flatnonzero(tied)

            # Warm-up must form a contiguous leading prefix. Gaps elsewhere indicate that
            # label and logit dates differ; do not silently conceal this data issue with DP labels.
            if tied_idx[0] != 0 or not np.array_equal(tied_idx, np.arange(len(tied_idx))):
                raise ValueError(
                    f"PseudoLabeler.label_df: {len(tied_idx)} rows have tied "
                    f"log-probs but they do not form a leading prefix "
                    f"(first tied index {tied_idx[0]}, last {tied_idx[-1]}). "
                    f"Expected only the leading seq_len-1 warm-up rows. "
                    f"This is a label/logit date-alignment problem."
                )

            # Run DP on the full sequence and take its prefix; do not run DP on the prefix alone.
            # Viterbi is global dynamic programming, so the path for a truncated sequence
            # can differ from the corresponding part of the full-sequence path.
            #
            # Like this method, DPLabeler.label_df sorts by date and resets the index,
            # so their row positions correspond exactly.
            dp_out = DPLabeler.label_df(self, df)

            if len(dp_out) != len(out):
                raise ValueError(
                    f"PseudoLabeler.label_df: DP fallback produced {len(dp_out)} "
                    f"rows but pseudo produced {len(out)}."
                )
            if not dp_out[self.cfg.date_col].equals(out[self.cfg.date_col]):
                raise ValueError(
                    "PseudoLabeler.label_df: DP fallback dates do not match "
                    "pseudo dates after sorting."
                )

            # DPLabeler._attach_outputs names label columns through SPEC.*, while this method
            # uses literals. Pair the columns explicitly rather than assuming identical names.
            for dp_col, pseudo_col in (
                (SPEC.joint_col, "joint_state"),
                (SPEC.axis1_col, "current_trend"),
                (SPEC.axis2_col, "future_trend"),
            ):
                if dp_col not in dp_out.columns:
                    raise ValueError(
                        f"PseudoLabeler.label_df: DP output has no column "
                        f"{dp_col!r}. Columns: {list(dp_out.columns)}"
                    )
                out.loc[tied, pseudo_col] = dp_out.loc[tied, dp_col].to_numpy()

            out.loc[tied, "pseudo_argmax"] = False

            print(
                f"[INFO] PseudoLabeler: {len(tied_idx)} leading warm-up rows "
                f"fell back to DP labels (no logits coverage).",
                flush=True,
            )

        # Add dC / gap / dC5 consistently with DPLabeler._attach_outputs
        # for ROW_FISHER_COLS in metrics_single.py.
        _compute_log_returns(out)

        return out

    def label_folder(
        self,
        in_dir: str | Path,
        out_dir: str | Path,
        pattern: str = "*.csv",
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        model_score_path: Optional[str | Path] = None,
        codebook_dir: Optional[str | Path] = None,
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
            print(
                f"[INFO] Blacklist filtered {before - len(files)} files, "
                f"{len(files)} remaining."
            )

        market = getattr(self.cfg, "market", "cn")

        # Read symbol_codebook_info.json to identify symbols with a codebook.
        symbols_with_codebook: set[str] | None = None
        if codebook_dir is not None:
            info_path = Path(codebook_dir) / "symbol_codebook_info.json"
            if info_path.exists():
                with open(info_path, "r", encoding="utf-8") as f:
                    symbol_info = json.load(f)
                symbols_with_codebook = {
                    normalize_symbol_value(s, market=market)
                    for s in symbol_info.get("symbols_with_codebook", [])
                }
                print(
                    f"[INFO] Loaded symbol_codebook_info: "
                    f"{len(symbols_with_codebook)} symbols with codebook."
                )
            else:
                print(
                    f"[WARN] symbol_codebook_info.json not found under {codebook_dir}, "
                    f"all symbols will use pseudo labeling."
                )

        # Read model scores once and group them by symbol.
        if model_score_path is None:
            raise ValueError(
                "PseudoLabeler.label_folder: model_score_path is required. "
                "Pseudo-labeling needs previous-iteration logits."
            )

        from common.symbols import normalize_symbol_series, validate_symbol_series

        p = Path(model_score_path)
        if not (p.is_file() and p.suffix.lower() in {".parquet", ".pq"}):
            raise ValueError(
                f"PseudoLabeler.label_folder: model_score_path must be a .parquet file, "
                f"got: {model_score_path}"
            )

        score_df = pd.read_parquet(p)
        score_df[self.cfg.symbol_col] = normalize_symbol_series(
            score_df[self.cfg.symbol_col], market=market
        )
        validate_symbol_series(
            score_df[self.cfg.symbol_col],
            market=market,
            name=f"{p.name}:{self.cfg.symbol_col}",
        )
        score_by_symbol: dict[str, pd.DataFrame] = {
            sym: group.reset_index(drop=True)
            for sym, group in score_df.groupby(self.cfg.symbol_col, sort=False)
        }

        # --- Serial path ---
        if n_workers <= 1:
            dp_labeler = DPLabeler(self.cfg)

            for fp in files:
                symbol = normalize_symbol_value(fp.stem.split("_", 1)[0], market=market)

                if symbols_with_codebook is not None and symbol not in symbols_with_codebook:
                    print(f"[INFO] {symbol!r} not in codebook, using DP labeling.")
                    dp_labeler.label_csv_file(
                        fp,
                        out_dir / fp.name,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    continue

                sym_score_df = score_by_symbol.get(symbol)
                if sym_score_df is None:
                    raise ValueError(
                        f"{symbol!r} is in codebook (symbols_with_codebook) "
                        f"but has no logits in {model_score_path}.\n"
                        f"Root cause: this symbol likely has fewer than seq_len rows "
                        f"in the train split, so export_logits skipped it.\n"
                        f"Fix: add {symbol!r} to the blacklist and re-run from codebook fitting."
                    )

                self.label_csv_file(
                    fp,
                    out_dir / fp.name,
                    start_date=start_date,
                    end_date=end_date,
                    model_score_path=sym_score_df,
                )
            return

        # --- Parallel path ---
        import dataclasses

        cfg_dict = dataclasses.asdict(self.cfg)
        start_date_str = str(start_date) if start_date is not None else None
        end_date_str = str(end_date) if end_date is not None else None

        tasks = []
        for fp in files:
            symbol = normalize_symbol_value(fp.stem.split("_", 1)[0], market=market)

            use_dp = (
                symbols_with_codebook is not None
                and symbol not in symbols_with_codebook
            )

            # Validate early: fail immediately for codebook symbols without logits.
            if not use_dp and score_by_symbol.get(symbol) is None:
                raise ValueError(
                    f"{symbol!r} is in codebook but has no logits in {model_score_path}.\n"
                    f"Fix: add {symbol!r} to the blacklist and re-run from codebook fitting."
                )

            tasks.append((
                cfg_dict,
                str(fp),
                str(out_dir / fp.name),
                start_date_str,
                end_date_str,
                use_dp,
                symbol,
                str(model_score_path),  # Workers filter by symbol; do not pass a DataFrame.
            ))

        print(
            f"[INFO] PseudoLabeler: parallel labeling {len(tasks)} files "
            f"with n_workers={n_workers}"
        )

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for _ in executor.map(_pseudo_label_one, tasks):
                pass  # Exceptions propagate to the main process during iteration.
