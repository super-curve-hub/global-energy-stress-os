"""
Tests for Market Layer
"""

import pandas as pd

from engine.market import fetch_market_data


def test_fetch_market_data():

    tickers = {
        "SPX": "^GSPC",
    }

    px = fetch_market_data(
        tickers,
        "2025-01-01",
    )

    assert isinstance(px, pd.DataFrame)

    assert "SPX" in px.columns

    assert len(px) > 100
