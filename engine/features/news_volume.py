from __future__ import annotations

import pandas as pd


def build_news_volume(
    news: pd.DataFrame,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Build daily news volume feature from news dataframe.
    """

    if news is None or news.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "news_volume",
            ]
        )

    if "published" not in news.columns:
        raise ValueError("news dataframe must contain 'published' column")

    df = news.copy()

    df["published"] = pd.to_datetime(
        df["published"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(subset=["published"])

    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "news_volume",
            ]
        )

    volume = (
        df.set_index("published")
        .resample(freq)
        .size()
        .rename("news_volume")
        .reset_index()
        .rename(columns={"published": "date"})
    )

    volume["date"] = volume["date"].dt.normalize()

    return volume
