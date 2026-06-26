from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def build_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily sentiment feature from news dataframe.
    """

    if news is None or news.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "sentiment",
            ]
        )

    required = ["published", "title", "summary"]

    for col in required:
        if col not in news.columns:
            raise ValueError(f"Missing required column: {col}")

    analyzer = SentimentIntensityAnalyzer()

    df = news.copy()

    df["published"] = pd.to_datetime(
        df["published"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["published"])

    df["text"] = df["title"].fillna("") + " " + df["summary"].fillna("")

    df["sentiment"] = df["text"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )

    sentiment = (
        df.groupby(df["published"].dt.normalize())["sentiment"]
        .mean()
        .reset_index()
        .rename(columns={"published": "date"})
    )

    return sentiment
