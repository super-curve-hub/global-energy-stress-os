from __future__ import annotations

import pandas as pd

NEWS_COLUMNS = [
    "source",
    "provider",
    "query",
    "title",
    "published",
    "url",
    "summary",
    "importance",
    "sentiment",
]


def normalize_news(
    df: pd.DataFrame,
    *,
    source: str,
    provider: str,
) -> pd.DataFrame:
    """
    Normalize news dataframe into the standard schema.
    """

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    df = df.copy()

    if "link" in df.columns and "url" not in df.columns:
        df = df.rename(columns={"link": "url"})

    if "published" in df.columns:
        df["published"] = pd.to_datetime(
            df["published"],
            utc=True,
            errors="coerce",
        )

    df["source"] = source
    df["provider"] = provider

    if "query" not in df.columns:
        df["query"] = ""

    if "title" not in df.columns:
        df["title"] = ""

    if "url" not in df.columns:
        df["url"] = ""

    if "summary" not in df.columns:
        df["summary"] = ""

    if "importance" not in df.columns:
        df["importance"] = 0.0

    if "sentiment" not in df.columns:
        df["sentiment"] = 0.0

    return df[NEWS_COLUMNS]
