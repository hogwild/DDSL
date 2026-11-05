from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import numpy as np
import pandas as pd

from states import STATE_NAMES
from common.symbols import (
    normalize_symbol_series,
    normalize_symbol_value,
    validate_symbol_series,
)

from .config import Semantic9Config
from .dp_labeler import DPLabeler
from .energy import (
    add_model_feedback_to_emissions,
    build_transition_matrix,
    compute_emission_scores,
    ewma,
    load_model_log_probs,
)
from .viterbi import viterbi_decode
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------
# Module-level worker function
# ---------------------------------------------------------------
def _ddsl_label_one(args: tuple) -> str:
    (cfg_dict, in_path_str, out_path_str,
     start_date_str, end_date_str,
     use_dp,
     symbol,             # Symbol string for this stock
     score_file_str,     # Parquet file path string, or None
    ) = args

    import pandas as pd
    from pathlib import Path
    from labeling.config import Semantic9Config
    from labeling.dp_labeler import DPLabeler
    from labeling.ddsl_labeler import DDSLLabeler

    cfg        = Semantic9Config(**cfg_dict)
    start_date = pd.Timestamp(start_date_str) if start_date_str else None
    end_date   = pd.Timestamp(end_date_str)   if end_date_str   else None

    if use_dp:
        labeler = DPLabeler(cfg)
        labeler.label_csv_file(
            Path(in_path_str),
            Path(out_path_str),
            start_date=start_date,
            end_date=end_date,
        )
    else:
        labeler = DDSLLabeler(cfg)

        sym_score_df = None
        if score_file_str is not None:
            # Each worker reads the parquet file and selects rows for its own symbol.
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


class DDSLLabeler(DPLabeler):
    """DDSL labeler: DP emissions + beta_model * recognizer log-probabilities."""

    # Each labeler declares whether it requires prior-iteration logits independently of beta_model.
    #
    #   DDSLLabeler: beta_model == 0 reduces to pure DP (see config.py), which
    #                requires no logits, so this flag is False. Loading is instead
    #                controlled by the beta_model != 0 branch in label_csv_file.
    #   PseudoLabeler: argmax requires logits regardless of beta_model,
    #                  so it overrides this flag with True.
    #
    # Use a class attribute rather than isinstance(self, PseudoLabeler) to avoid
    # a circular import between pseudo_labeler and ddsl_labeler.
    #
    # Distinguish this from cfg.require_model_scores, which controls whether incomplete
    # logit coverage raises an error or receives a uniform prior; the meanings differ.
    needs_model_scores: bool = False

    def label_df(self, df: pd.DataFrame, model_log_probs: Optional[np.ndarray] = None) -> pd.DataFrame:
        cfg = self.cfg
        if cfg.beta_model == 0.0:
            return super().label_df(df)

        out = df.copy()
        out[cfg.date_col] = pd.to_datetime(out[cfg.date_col])
        out = out.sort_values(cfg.date_col).reset_index(drop=True)

        close = pd.to_numeric(out[cfg.price_col], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(close)):
            close = pd.Series(close).ffill().bfill().to_numpy(dtype=float)

        smooth = ewma(close, cfg.smooth_span)
        base_emissions, diagnostics = compute_emission_scores(smooth, cfg)
        emissions, model_term = add_model_feedback_to_emissions(base_emissions, model_log_probs, cfg)
        transition = build_transition_matrix(cfg)
        path, dp_scores = viterbi_decode(emissions, transition)

        self._attach_outputs(out, smooth, path, dp_scores, diagnostics)
        out["ddsl_beta_model"] = cfg.beta_model
        if model_term is not None:
            out["model_logprob_path"] = model_log_probs[np.arange(len(path)), path]
            out["model_term_path"] = model_term[np.arange(len(path)), path]
            out["base_emission_path"] = base_emissions[np.arange(len(path)), path]
            out["ddsl_emission_path"] = emissions[np.arange(len(path)), path]
        return out

    def label_csv_file(
        self,
        in_path: str | Path,
        out_path: str | Path,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        model_score_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        in_path = Path(in_path)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # print(f"[DEBUG] beta_model: {self.cfg.beta_model}")

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

        if end_date is not None:
            dates = pd.to_datetime(df[self.cfg.date_col], errors="coerce")
            df = df.loc[dates <= end_date].reset_index(drop=True)

        if df.empty:
            print(f"Skip {in_path.name}: no data before end_date")
            return pd.DataFrame()

        # model_log_probs is a positional array built in the current row order of df,
        # while label_df applies sort_values(date_col).reset_index(drop=True). There is
        # no intervening realignment, so df must already be sorted by date; otherwise
        # logits attach to the wrong dates without a shape mismatch or an error.
        #
        # The load_model_log_probs check that missing rows form a contiguous leading
        # prefix also depends on this: a temporal prefix requires date-ordered rows.
        dates = pd.to_datetime(df[self.cfg.date_col], errors="coerce")
        if dates.isna().any():
            raise ValueError(
                f"{in_path.name}: {int(dates.isna().sum())} unparseable values "
                f"in date column {self.cfg.date_col!r}."
            )
        if not dates.is_monotonic_increasing:
            raise ValueError(
                f"{in_path.name}: rows are not in ascending date order. "
                f"model_log_probs is positional and label_df sorts afterwards, "
                f"so unsorted input would silently misalign logits with dates."
            )
        # Load logits according to the labeler requirements. The original implementation
        # checked only self.cfg.beta_model != 0.0, before PseudoLabeler existed:
        # beta_model also acted as the feedback switch. PseudoLabeler always needs
        # logits regardless of beta_model, so a pseudo run with beta_model=0 would
        # fail in label_df because model_log_probs was None.
        model_log_probs = None
        if self.needs_model_scores or self.cfg.beta_model != 0.0:
            if model_score_path is None:
                raise ValueError(
                    f"{type(self).__name__} needs model scores "
                    f"(needs_model_scores={self.needs_model_scores}, "
                    f"beta_model={self.cfg.beta_model}) but model_score_path is None"
                )
            model_log_probs = load_model_log_probs(model_score_path, df, self.cfg)

        labeled = self.label_df(df, model_log_probs=model_log_probs)

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

        # Apply the blacklist filter.
        if blacklist:
            before = len(files)
            files = [f for f in files if f.name.split("_", 1)[0] not in blacklist]
            print(f"[INFO] Blacklist filtered {before - len(files)} files, "
                f"{len(files)} remaining.")

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
                    f"all symbols will use DDSL labeling."
                )
       
        # Preload a merged model-score file once and group its rows by symbol.
        score_by_symbol: dict[str, pd.DataFrame] | None = None
        if model_score_path is not None:
            p = Path(model_score_path)
            if p.is_file() and p.suffix.lower() in {".parquet", ".pq"}:
                score_df = pd.read_parquet(p)
                if self.cfg.symbol_col in score_df.columns and score_df[self.cfg.symbol_col].nunique() > 1:
                    score_df[self.cfg.symbol_col] = normalize_symbol_series(
                        score_df[self.cfg.symbol_col],
                        market=market,
                    )
                    validate_symbol_series(
                        score_df[self.cfg.symbol_col],
                        market=market,
                        name=f"{p.name}:{self.cfg.symbol_col}",
                    )
                    score_by_symbol = {
                        sym: group.reset_index(drop=True)
                        for sym, group in score_df.groupby(self.cfg.symbol_col, sort=False)
                    }
        # --- Serial path (n_workers=1 or 0): preserve the original behavior. ---
        if n_workers == 1 or n_workers == 0:
            dp_labeler = DPLabeler(self.cfg)

            for fp in files:
                symbol = normalize_symbol_value(
                    fp.stem.split("_", 1)[0],
                    market=market,
                )

                # Use DPLabeler directly for symbols without a codebook.
                if symbols_with_codebook is not None and symbol not in symbols_with_codebook:
                    print(f"[INFO] {symbol!r} not in codebook, using DP labeling.")
                    dp_labeler.label_csv_file(
                        fp,
                        out_dir / fp.name,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    continue

                if score_by_symbol is not None:
                    sym_score_df = score_by_symbol.get(symbol)
                    if sym_score_df is None:
                        raise ValueError(
                            f"{symbol!r} is in codebook (symbols_with_codebook) "
                            f"but has no logits in {model_score_path}.\n"
                            f"Root cause: this symbol likely has fewer than seq_len rows "
                            f"in the train split, so export_logits skipped it.\n"
                            f"Fix: add {symbol!r} to the blacklist "
                            f"(configs/symbols/CSI300_blacklist.txt) and re-run from codebook fitting."
                        )
                    resolved = sym_score_df
                else:
                    resolved = resolve_model_score_path(model_score_path, fp.stem)

                self.label_csv_file(
                    fp,
                    out_dir / fp.name,
                    start_date=start_date,
                    end_date=end_date,
                    model_score_path=resolved,
                )
            return
        
         # --- Parallel path ---
        import dataclasses
        cfg_dict       = dataclasses.asdict(self.cfg)
        start_date_str = str(start_date) if start_date is not None else None
        end_date_str   = str(end_date)   if end_date   is not None else None

        tasks = []
        for fp in files:
            symbol = normalize_symbol_value(
                fp.stem.split("_", 1)[0], market=market)

            use_dp = (symbols_with_codebook is not None
                    and symbol not in symbols_with_codebook)

            # Fail early if a symbol has a codebook but no logits.
            if not use_dp and score_by_symbol is not None:
                if score_by_symbol.get(symbol) is None:
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
                symbol,                                              # Pass the symbol.
                str(model_score_path) if model_score_path else None, # Pass the path rather than the data.
            ))

        print(f"[INFO] DDSLLabeler: parallel labeling {len(tasks)} files "
            f"with n_workers={n_workers}")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for result in executor.map(_ddsl_label_one, tasks):
                pass  # Exceptions propagate to the main process during iteration.

def resolve_model_score_path(model_score_source: Optional[str | Path], stock_stem: str) -> Optional[Path]:
    if model_score_source is None:
        return None
    p = Path(model_score_source)
    if p.is_file():
        return p
    if p.is_dir():
        for candidate in [p / f"{stock_stem}.parquet", p / f"{stock_stem}.pq", p / f"{stock_stem}.csv"]:
            if candidate.exists():
                return candidate
        return p / f"{stock_stem}.csv"
    return p


def expected_model_score_columns(kind: str = "logit") -> list[str]:
    return [f"{kind}_{state_name}" for state_name in STATE_NAMES]