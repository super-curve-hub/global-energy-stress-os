import pandas as pd

from engine.features import build_news_volume


def test_build_news_volume():

    news = pd.DataFrame(
        {
            "published": [
                "2026-06-20T01:00:00Z",
                "2026-06-20T03:00:00Z",
                "2026-06-21T01:00:00Z",
            ],
            "title": [
                "A",
                "B",
                "C",
            ],
        }
    )

    result = build_news_volume(news)

    assert "news_volume" in result.columns
    assert len(result) == 2
    assert result.loc[0, "news_volume"] == 2
    assert result.loc[1, "news_volume"] == 1
