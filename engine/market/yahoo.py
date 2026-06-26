"""
Global Energy Stress OS

Yahoo Finance Market Layer

Author: Super Curve
"""

from __future__ import annotations

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
    Download one ticker from Yahoo Finance.
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
                "No data returned: %s",
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

    except Exception:

        logger.exception(
            "Yahoo download failed: %s",
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
    Clean and validate market dataframe.
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

    cols = [c for c in required if c in px.columns]

    if cols:

        px = px.dropna(subset=cols)

    return px


# ==========================================================
# Cache
# ==========================================================


CACHE = ROOT / "data" / "cache" / "market.csv"


def save_cache(
    px: pd.DataFrame,
) -> None:

    CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    px.to_csv(CACHE)

    logger.info("Market cache saved.")


def load_cache():

    if not CACHE.exists():

        return None

    logger.info("Loading market cache.")

    return pd.read_csv(
        CACHE,
        index_col=0,
        parse_dates=True,
    )


# ==========================================================
# Fetch
# ==========================================================


def fetch_market_data(
    tickers: dict[str, str],
    start: str,
) -> pd.DataFrame:
    """
    Download all configured market series.
    """

    series = {}

    for name, ticker in tickers.items():

        s = safe_download(
            ticker,
            start,
        )

        if s is not None:

            series[name] = s

    if len(series) == 0:

        raise RuntimeError("No market data downloaded.")

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

    return px


def fetch_market(
    cfg: dict,
    use_cache: bool = False,
) -> pd.DataFrame:
    """
    Main Market Layer interface.
    """

    if use_cache:

        cache = load_cache()

        if cache is not None:

            return cache

    start = cfg["project"]["start_date"]

    tickers = cfg["market"]["tickers"]

    px = fetch_market_data(
        tickers,
        start,
    )

    logger.info("Market Layer completed.")

    return px
