"""
Global Energy Stress OS

Market Data Layer

Author: Super Curve
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from engine.utils import (
    ROOT,
    get_logger,
)

logger = get_logger()


# ==========================================================
# Download
# ==========================================================

def safe_download(
    ticker: str,
    start: str,
) -> pd.Series | None:
    """
    Safe Yahoo Finance downloader.
    """

    try:

        data = yf.download(
            ticker,
            start=start,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data.empty:

            logger.warning(
                "No data: %s",
                ticker,
            )

            return None

        close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close.name = ticker

        logger.info(
            "Downloaded %s (%d rows)",
            ticker,
            len(close),
        )

        return close.dropna()

    except Exception as e:

        logger.exception(
            "Download failed: %s",
            ticker,
        )

        return None


# ==========================================================
# Validation
# ==========================================================

def validate_market(
    px: pd.DataFrame,
) -> pd.DataFrame:
    """
    Basic validation.
    """

    px = px.sort_index()

    px = px.ffill()

    required = [

        "WTI",
        "BRENT",
        "DXY",
        "SPX",
        "USDJPY",

    ]

    cols = [

        c
        for c in required
        if c in px.columns

    ]

    px = px.dropna(
        subset=cols
    )

    return px


# ==========================================================
# Cache
# ==========================================================

def save_cache(
    px: pd.DataFrame,
) -> None:

    cache = ROOT / "data" / "cache"

    cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    px.to_csv(
        cache / "market.csv"
    )


def load_cache():

    f = ROOT / "data" / "cache" / "market.csv"

    if not f.exists():

        return None

    return pd.read_csv(
        f,
        index_col=0,
        parse_dates=True,
    )


# ==========================================================
# Fetch
# ==========================================================

def fetch_market(
    cfg: dict,
    use_cache: bool = False,
) -> pd.DataFrame:

    if use_cache:

        cache = load_cache()

        if cache is not None:

            logger.info(
                "Loaded cache."
            )

            return cache

    start = cfg["project"][
        "start_date"
    ]

    tickers = cfg[
        "market"
    ]["tickers"]

    series = {}

    for name, ticker in tickers.items():

        s = safe_download(
            ticker,
            start,
        )

        if s is not None:

            series[name] = s

    if len(series) == 0:

        raise RuntimeError(
            "No market data downloaded."
        )

    px = pd.concat(
        series,
        axis=1,
        sort=False,
    )

    px = validate_market(
        px,
    )

    save_cache(
        px,
    )

    logger.info(
        "Market Layer completed."
    )

    return px
