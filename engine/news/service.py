from __future__ import annotations

from urllib.parse import quote

import feedparser
import pandas as pd
import requests

from engine.utils import get_logger

logger = get_logger()


def google_news_count(query: str) -> int:
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        feed = feedparser.parse(url)
        return len(feed.entries)
    except Exception as e:
        logger.warning("Google RSS failed %s: %s", query, e)
        return 0


def gdelt_count(query: str) -> int:
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc?"
        f"query={quote(query)}"
        "&mode=timelinevolraw"
        "&format=json"
        "&timespan=7d"
    )

    try:
        r = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        if r.status_code != 200:
            logger.warning("GDELT status %s for %s", r.status_code, query)
            return 0

        data = r.json()
        timeline = data.get("timeline", [])

        return int(sum(float(x.get("value", 0)) for x in timeline))

    except Exception as e:
        logger.warning("GDELT failed %s: %s", query, e)
        return 0


def rss_count(feed_url: str) -> int:
    try:
        feed = feedparser.parse(feed_url)
        return len(feed.entries)
    except Exception as e:
        logger.warning("RSS failed %s: %s", feed_url, e)
        return 0


def fetch_news(
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch News Layer.

    Returns
    -------
    news : Google News + RSS
    gdelt : GDELT timeline
    """

    today = pd.Timestamp.now("UTC").normalize()
    news_cfg = cfg.get("news", {})

    news_rows: list[dict] = []
    gdelt_rows: list[dict] = []

    # Google News
    for query in news_cfg.get("google_queries", []):
        news_rows.append(
            {
                "date": today,
                "source": "google_news",
                "query": query,
                "count": google_news_count(query),
            }
        )

    # RSS
    for url in news_cfg.get("rss_feeds", []):
        news_rows.append(
            {
                "date": today,
                "source": "rss",
                "query": url,
                "count": rss_count(url),
            }
        )

    # GDELT
    for query in news_cfg.get("gdelt_queries", []):
        gdelt_rows.append(
            {
                "date": today,
                "source": "gdelt",
                "query": query,
                "count": gdelt_count(query),
            }
        )

    news = pd.DataFrame(news_rows)
    gdelt = pd.DataFrame(gdelt_rows)

    return news, gdelt
