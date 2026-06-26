from __future__ import annotations

from datetime import datetime, timezone

import feedparser
import pandas as pd


def fetch_rss_feeds(feeds: dict[str, str], max_articles: int = 80) -> pd.DataFrame:
    rows = []
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_articles]:
                title = getattr(entry, "title", "")
                link = getattr(entry, "link", "")
                summary = getattr(entry, "summary", "")
                published = getattr(entry, "published", None) or getattr(
                    entry, "updated", None
                )
                rows.append(
                    {
                        "source": source,
                        "title": title,
                        "summary": summary,
                        "link": link,
                        "published": published,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
        except Exception as e:
            print(f"[rss] failed {source}: {e}")
    if not rows:
        return pd.DataFrame(
            columns=["source", "title", "summary", "link", "published", "fetched_at"]
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["title", "link"])
    return df.reset_index(drop=True)
