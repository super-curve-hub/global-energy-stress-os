from __future__ import annotations

import pandas as pd


def build_gdelt_index(
    gdelt: pd.DataFrame,
    span: int = 7,
    surprise_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Build GDELT features.

    Parameters
    ----------
    gdelt : pandas.DataFrame

        Required columns
        ----------------
        date
        GDELT_News

    span : int
        EWMA span.

    surprise_threshold : float
        Z-score threshold.

    Returns
    -------
    pandas.DataFrame

        date

        gdelt_news

        gdelt_ewma

        gdelt_std

        gdelt_zscore

        gdelt_surprise
    """

    if gdelt is None or gdelt.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "gdelt_news",
                "gdelt_ewma",
                "gdelt_std",
                "gdelt_zscore",
                "gdelt_surprise",
            ]
        )

    required = [
        "date",
        "GDELT_News",
    ]

    for col in required:
        if col not in gdelt.columns:
            raise ValueError(f"Missing required column: {col}")

    df = gdelt.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    df["gdelt_news"] = df["GDELT_News"].astype(float)

    df["gdelt_ewma"] = (
        df["gdelt_news"]
        .ewm(
            span=span,
            adjust=False,
        )
        .mean()
    )

    df["gdelt_std"] = (
        df["gdelt_news"]
        .rolling(
            window=span,
            min_periods=2,
        )
        .std()
        .fillna(0.0)
    )

    df["gdelt_zscore"] = (df["gdelt_news"] - df["gdelt_ewma"]) / df[
        "gdelt_std"
    ].replace(
        0,
        1,
    )

    df["gdelt_surprise"] = (df["gdelt_zscore"] >= surprise_threshold).astype(int)

    return df[
        [
            "date",
            "gdelt_news",
            "gdelt_ewma",
            "gdelt_std",
            "gdelt_zscore",
            "gdelt_surprise",
        ]
    ]
