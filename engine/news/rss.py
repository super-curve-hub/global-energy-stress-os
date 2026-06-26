from __future__ import annotations

from datetime import datetime, timezone

import feedparser
import pandas as pd


def fetch_rss(
    feeds: list[str],
    max_articles: int = 80,
) -> pd.DataFrame:
    """
    Fetch RSS feeds.

    Parameters
    ----------
    feeds
        RSS URL list.

    Returns
    -------
    pandas.DataFrame
    """

    rows = []

    for url in feeds:

        try:

            feed = feedparser.parse(url)

            for entry in feed.entries[:max_articles]:

                rows.append(
                    {
                        "source": "RSS",
                        "provider": feed.feed.get("title", "RSS"),
                        "query": "",
                        "title": getattr(entry, "title", ""),
                        "published": getattr(
                            entry,
                            "published",
                            None,
                        )
                        or getattr(
                            entry,
                            "updated",
                            None,
                        ),
                        "link": getattr(entry, "link", ""),
                        "summary": getattr(entry, "summary", ""),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

        except Exception as e:

            print(f"[rss] failed {url}: {e}")

    if not rows:

        return pd.DataFrame(
            columns=[
                "source",
                "provider",
                "query",
                "title",
                "published",
                "link",
                "summary",
                "fetched_at",
            ]
        )

    df = pd.DataFrame(rows)

    return df.drop_duplicates(subset=["title", "link"]).reset_index(drop=True)
