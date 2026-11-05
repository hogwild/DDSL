from __future__ import annotations

import re
import pandas as pd


CN_MARKETS = {"cn", "china", "a_share", "ashare"}
US_MARKETS = {"us", "usa", "us_stock"}


def normalize_symbol_value(x, market: str = "cn", width: int = 6) -> str:
    s = str(x).strip()

    if market in CN_MARKETS:
        if not s.isdigit():
            raise ValueError(f"Invalid CN symbol: {x!r}")
        if len(s) > width:
            raise ValueError(f"CN symbol too long: {x!r}")
        return s.zfill(width)

    if market in US_MARKETS:
        s = s.upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]*", s):
            raise ValueError(f"Invalid US symbol: {x!r}")
        return s

    return s


def normalize_symbol_series(
    s: pd.Series,
    market: str = "cn",
    width: int = 6,
) -> pd.Series:
    out = s.astype(str).str.strip()

    if market in CN_MARKETS:
        bad = out[~out.str.fullmatch(r"\d+")]
        if not bad.empty:
            examples = bad.drop_duplicates().head(10).tolist()
            raise ValueError(f"Invalid CN symbols: {examples}")
        too_long = out[out.str.len() > width]
        if not too_long.empty:
            examples = too_long.drop_duplicates().head(10).tolist()
            raise ValueError(f"CN symbols longer than {width}: {examples}")
        return out.str.zfill(width)

    if market in US_MARKETS:
        out = out.str.upper()
        bad = out[~out.str.fullmatch(r"[A-Z][A-Z0-9.-]*")]
        if not bad.empty:
            examples = bad.drop_duplicates().head(10).tolist()
            raise ValueError(f"Invalid US symbols: {examples}")
        return out

    return out


def validate_symbol_series(
    s: pd.Series,
    market: str = "cn",
    width: int = 6,
    name: str = "symbol",
) -> None:
    out = s.astype(str).str.strip()

    if market in CN_MARKETS:
        ok = out.str.fullmatch(rf"\d{{{width}}}")

    elif market in US_MARKETS:
        ok = out.str.fullmatch(r"[A-Z][A-Z0-9.-]*")

    else:
        return

    if not ok.all():
        examples = out.loc[~ok].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{name} contains non-normalized symbols for market={market}: {examples}"
        )


def validate_symbol_value(
    x,
    market: str = "cn",
    width: int = 6,
    name: str = "symbol",
) -> None:
    validate_symbol_series(pd.Series([x]), market=market, width=width, name=name)