from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    """
    Immutable container for Feature Layer outputs.

    Attributes
    ----------
    news_volume
        Daily news volume feature.

    sentiment
        Daily sentiment feature.

    gdelt
        GDELT-derived features.

    war_score
        Composite war score feature.

    dataframe
        Merged feature dataframe.
    """

    news_volume: pd.DataFrame

    sentiment: pd.DataFrame

    gdelt: pd.DataFrame

    war_score: pd.DataFrame

    dataframe: pd.DataFrame
