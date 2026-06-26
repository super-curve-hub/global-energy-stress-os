import pandas as pd

from engine.features.war_score import build_war_score


def test_build_war_score():

    features = pd.DataFrame(
        {
            "date": [
                "2026-06-20",
                "2026-06-21",
                "2026-06-22",
            ],
            "news_volume": [
                10,
                25,
                15,
            ],
            "sentiment": [
                -0.50,
                -0.80,
                0.20,
            ],
            "gdelt_zscore": [
                0.5,
                2.8,
                0.3,
            ],
            "gdelt_surprise": [
                0,
                1,
                0,
            ],
        }
    )

    result = build_war_score(features)

    assert "date" in result.columns
    assert "war_score" in result.columns

    assert len(result) == 3

    assert result["war_score"].between(0, 1).all()
