import pandas as pd

from engine.features import build_features


def test_build_features():

    news = pd.DataFrame(
        {
            "published": [
                "2026-06-20T00:00:00Z",
                "2026-06-21T00:00:00Z",
            ],
            "title": [
                "Oil prices surge",
                "Markets remain calm",
            ],
            "summary": [
                "Supply concerns support crude.",
                "Trading activity is stable.",
            ],
        }
    )

    gdelt = pd.DataFrame(
        {
            "date": [
                "2026-06-20",
                "2026-06-21",
            ],
            "GDELT_News": [
                12,
                30,
            ],
        }
    )

    bundle = build_features(
        news=news,
        gdelt=gdelt,
    )

    assert bundle.dataframe is not None

    assert "war_score" in bundle.dataframe.columns
