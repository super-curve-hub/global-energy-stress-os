from __future__ import annotations

import pandas as pd

from .google import fetch_google_news
from .rss import fetch_rss
from .schema import normalize_news


def fetch_news(cfg: dict) -> pd.DataFrame:
    """
    Aggregate all news sources.

    Phase 4
    --------
    - Google News RSS
    - RSS Feeds

    Phase 5
    --------
    - GDELT News Index
    - Reuters
    - MarineTraffic
    - AIS
    """

    news_cfg = cfg["news"]

    frames: list[pd.DataFrame] = []

    # -------------------------------------------------
    # Google News
    # -------------------------------------------------

    google = fetch_google_news(news_cfg.get("google_queries", []))

    google = normalize_news(
        google,
        source="Google",
        provider="Google News RSS",
    )

    frames.append(google)

    # -------------------------------------------------
    # RSS
    # -------------------------------------------------

    rss_urls = news_cfg.get("rss_feeds", [])

    if rss_urls:

        rss = fetch_rss(rss_urls)

        rss = normalize_news(
            rss,
            source="RSS",
            provider="RSS Feed",
        )

        frames.append(rss)

    # -------------------------------------------------
    # Merge
    # -------------------------------------------------

    if not frames:
        return pd.DataFrame()

    news = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    news = (
        news.drop_duplicates(subset=["title", "url"])
        .sort_values(
            "published",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return news
