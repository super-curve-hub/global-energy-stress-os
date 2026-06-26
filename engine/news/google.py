from __future__ import annotations

import urllib.parse

import feedparser
import pandas as pd


GOOGLE_RSS = "https://news.google.com/rss/search?q={query}"


def fetch_google_news(
    queries: list[str],
    language: str = "en-US",
    country: str = "US",
) -> pd.DataFrame:
    """
    Fetch Google News RSS.

    Parameters
    ----------
    queries
        Search keywords.
    language
        Google RSS language.
    country
        Google RSS country.

    Returns
    -------
    pandas.DataFrame
    """

    rows = []

    for query in queries:

        url = GOOGLE_RSS.format(
            query=urllib.parse.quote(query)
        )

        url += f"&hl={language}&gl={country}&ceid={country}:{language}"

        feed = feedparser.parse(url)

        for entry in feed.entries:

            rows.append(
                {
                    "source": "Google",
                    "query": query,
                    "title": entry.get("title", ""),
                    "published": entry.get("published", ""),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", ""),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "query",
                "title",
                "published",
                "link",
                "summary",
            ]
        )

    df = pd.DataFrame(rows)

    df["published"] = pd.to_datetime(
        df["published"],
        errors="coerce",
        utc=True,
    )

    df = (
        df.sort_values("published", ascending=False)
        .reset_index(drop=True)
    )

    return df