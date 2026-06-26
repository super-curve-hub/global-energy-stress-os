from __future__ import annotations

import pandas as pd

from engine.market import yahoo
from engine.market.yahoo import (
    fetch_market_data,
    safe_download,
    validate_market,
)


def test_import_fetch_market_data():
    assert callable(fetch_market_data)


# ==========================================================
# safe_download
# ==========================================================


def test_safe_download_ok(monkeypatch):
    def fake_download(*args, **kwargs):
        idx = pd.date_range("2026-01-01", periods=3)

        return pd.DataFrame(
            {
                "Close": [70.0, 71.0, 72.0],
            },
            index=idx,
        )

    monkeypatch.setattr(
        "engine.market.yahoo.yf.download",
        fake_download,
    )

    s = safe_download(
        "WTI",
        "2026-01-01",
    )

    assert s is not None
    assert len(s) == 3
    assert s.name == "WTI"


def test_safe_download_empty(monkeypatch):
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(
        "engine.market.yahoo.yf.download",
        fake_download,
    )

    assert (
        safe_download(
            "WTI",
            "2026-01-01",
        )
        is None
    )


def test_safe_download_exception(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("network")

    monkeypatch.setattr(
        "engine.market.yahoo.yf.download",
        broken,
    )

    assert (
        safe_download(
            "WTI",
            "2026-01-01",
        )
        is None
    )


# ==========================================================
# validate_market
# ==========================================================


def test_validate_market():
    idx = pd.date_range(
        "2026-01-01",
        periods=3,
    )

    df = pd.DataFrame(
        {
            "WTI": [70, None, 72],
            "BRENT": [73, 74, 75],
            "DXY": [100, 101, 102],
            "SPX": [6000, 6001, 6002],
            "USDJPY": [150, 151, 152],
        },
        index=idx,
    )

    out = validate_market(df)

    assert len(out) == 3
    assert out["WTI"].isna().sum() == 0


# ==========================================================
# cache
# ==========================================================


def test_save_and_load_cache(tmp_path, monkeypatch):
    cache = tmp_path / "market.csv"

    monkeypatch.setattr(
        yahoo,
        "CACHE",
        cache,
    )

    df = pd.DataFrame(
        {
            "WTI": [70, 71],
        },
        index=pd.date_range(
            "2026-01-01",
            periods=2,
        ),
    )

    yahoo.save_cache(df)

    out = yahoo.load_cache()

    assert out is not None
    assert len(out) == 2
    assert list(out.columns) == ["WTI"]


def test_load_cache_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        yahoo,
        "CACHE",
        tmp_path / "missing.csv",
    )

    assert yahoo.load_cache() is None


# ==========================================================
# fetch_market_data
# ==========================================================


def test_fetch_market_data(monkeypatch):
    idx = pd.date_range(
        "2026-01-01",
        periods=2,
    )

    def fake_safe_download(ticker, start):
        return pd.Series(
            [1.0, 2.0],
            index=idx,
            name=ticker,
        )

    monkeypatch.setattr(
        yahoo,
        "safe_download",
        fake_safe_download,
    )

    monkeypatch.setattr(
        yahoo,
        "save_cache",
        lambda df: None,
    )

    tickers = {
        "WTI": "CL=F",
        "BRENT": "BZ=F",
        "DXY": "DX-Y.NYB",
        "SPX": "^GSPC",
        "USDJPY": "JPY=X",
    }

    out = yahoo.fetch_market_data(
        tickers,
        "2026-01-01",
    )

    assert len(out) == 2
    assert "WTI" in out.columns


def test_fetch_market_data_empty(monkeypatch):
    monkeypatch.setattr(
        yahoo,
        "safe_download",
        lambda *args, **kwargs: None,
    )

    try:
        yahoo.fetch_market_data(
            {"WTI": "CL=F"},
            "2026-01-01",
        )
    except RuntimeError:
        return

    assert False, "RuntimeError was expected"


# ==========================================================
# fetch_market
# ==========================================================


def test_fetch_market_use_cache(monkeypatch):
    df = pd.DataFrame(
        {
            "WTI": [1],
        },
        index=pd.date_range(
            "2026-01-01",
            periods=1,
        ),
    )

    monkeypatch.setattr(
        yahoo,
        "load_cache",
        lambda: df,
    )

    out = yahoo.fetch_market(
        {},
        use_cache=True,
    )

    assert out.equals(df)


def test_fetch_market_download(monkeypatch):
    df = pd.DataFrame(
        {
            "WTI": [1],
        },
        index=pd.date_range(
            "2026-01-01",
            periods=1,
        ),
    )

    monkeypatch.setattr(
        yahoo,
        "load_cache",
        lambda: None,
    )

    monkeypatch.setattr(
        yahoo,
        "fetch_market_data",
        lambda tickers, start: df,
    )

    cfg = {
        "project": {
            "start_date": "2026-01-01",
        },
        "market": {
            "tickers": {
                "WTI": "CL=F",
            },
        },
    }

    out = yahoo.fetch_market(cfg)

    assert out.equals(df)
