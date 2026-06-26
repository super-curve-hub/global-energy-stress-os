from __future__ import annotations

import pandas as pd
import yfinance as yf


def safe_download(ticker: str, start: str) -> pd.Series | None:
    try:
        data = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
        if data is None or len(data) == 0 or "Close" not in data:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = ticker
        return close.dropna()
    except Exception as e:
        print(f"[market] download failed {ticker}: {e}")
        return None


def fetch_market_data(tickers: dict[str, str], start: str) -> pd.DataFrame:
    px = {}
    for name, ticker in tickers.items():
        s = safe_download(ticker, start)
        if s is not None:
            px[name] = s
    if not px:
        raise RuntimeError("No market data downloaded")
    df = pd.concat(px, axis=1).sort_index()
    return df.ffill().dropna(how="all")
