from __future__ import annotations

import random
import time
from urllib.parse import quote

import pandas as pd
import requests


def gdelt_timeline(
    keyword: str, timespan: str = "90d", retries: int = 3
) -> dict | None:
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc?"
        f"query={quote(keyword)}&mode=timelinevolraw&format=json&timespan={timespan}"
    )
    for i in range(retries):
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 429:
                wait = 1.5 * (2**i) + random.random()
                print(f"[gdelt] 429 {keyword}, retry in {wait:.1f}s")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                print(f"[gdelt] status {r.status_code} {keyword}")
                return None
            return r.json()
        except Exception as e:
            print(f"[gdelt] error {keyword}: {e}")
            time.sleep(1)
    return None


def fetch_gdelt_news_index(
    keywords: list[str], index: pd.DatetimeIndex
) -> pd.DataFrame:
    rows = []
    for kw in keywords:
        data = gdelt_timeline(kw)
        if not data:
            continue
        for item in data.get("timeline", []):
            dt = item.get("date")
            val = item.get("value", 0)
            if dt:
                rows.append(
                    {
                        "Date": pd.to_datetime(dt).normalize(),
                        "keyword": kw,
                        "value": float(val),
                    }
                )
    if not rows:
        return pd.DataFrame({"GDELT_News": 0.0}, index=index)
    df = pd.DataFrame(rows)
    s = df.groupby("Date")["value"].sum().sort_index()
    out = pd.DataFrame(index=pd.date_range(index.min(), index.max(), freq="D"))
    out["GDELT_News"] = s.reindex(out.index).fillna(0.0)
    return out.reindex(index, method="ffill").fillna(0.0)
