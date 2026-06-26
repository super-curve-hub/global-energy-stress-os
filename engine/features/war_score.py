from __future__ import annotations

import pandas as pd


def build_war_score(features: pd.DataFrame) -> pd.DataFrame:
    """
    Build composite War Score feature.

    Parameters
    ----------
    features : DataFrame

        Required columns

        date

        news_volume

        sentiment

        gdelt_zscore

        gdelt_surprise

    Returns
    -------
    DataFrame

        date

        war_score
    """

    required = [
        "date",
        "news_volume",
        "sentiment",
        "gdelt_zscore",
        "gdelt_surprise",
    ]

    for col in required:
        if col not in features.columns:
            raise ValueError(f"Missing required column: {col}")

    df = features.copy()

    df["date"] = pd.to_datetime(df["date"])

    # ---------- Normalization ----------

    volume = (df["news_volume"] / df["news_volume"].max()).clip(0, 1)

    negative_sentiment = (-df["sentiment"]).clip(0, 1)

    gdelt = (df["gdelt_zscore"] / 3.0).clip(0, 1)

    surprise = df["gdelt_surprise"]

    # ---------- Weighted Score ----------

    df["war_score"] = (
        0.20 * volume + 0.10 * negative_sentiment + 0.30 * gdelt + 0.40 * surprise
    )

    df["war_score"] = df["war_score"].clip(0, 1)

    return df[
        [
            "date",
            "war_score",
        ]
    ]
