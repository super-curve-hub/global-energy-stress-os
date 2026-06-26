from __future__ import annotations

import pandas as pd

from .bundle import FeatureBundle
from .gdelt import build_gdelt_index
from .news_volume import build_news_volume
from .sentiment import build_sentiment
from .war_score import build_war_score


def normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize date column to timezone-naive datetime.
    """

    if df.empty:
        return df

    out = df.copy()

    out["date"] = (
        pd.to_datetime(
            out["date"],
            errors="coerce",
            utc=True,
        )
        .dt.normalize()
        .dt.tz_localize(None)
    )

    return out


def build_features(
    news: pd.DataFrame,
    gdelt: pd.DataFrame,
) -> FeatureBundle:
    """
    Build all Feature Layer outputs.

    Parameters
    ----------
    news : DataFrame
        News dataframe.

    gdelt : DataFrame
        GDELT dataframe.

    Returns
    -------
    FeatureBundle
    """

    news_volume = normalize_date(build_news_volume(news))

    sentiment = normalize_date(build_sentiment(news))

    gdelt_feature = normalize_date(build_gdelt_index(gdelt))

    features = (
        news_volume.merge(
            sentiment,
            on="date",
            how="outer",
        )
        .merge(
            gdelt_feature,
            on="date",
            how="outer",
        )
        .fillna(0)
    )

    war_score = build_war_score(features)

    war_score = normalize_date(war_score)

    dataframe = (
        features.merge(
            war_score,
            on="date",
            how="left",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return FeatureBundle(
        news_volume=news_volume,
        sentiment=sentiment,
        gdelt=gdelt_feature,
        war_score=war_score,
        dataframe=dataframe,
    )
