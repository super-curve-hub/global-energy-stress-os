import pandas as pd

from engine.features.sentiment import build_sentiment


def test_build_sentiment():
    news = pd.DataFrame(
        {
            "published": [
                "2026-06-20T00:00:00Z",
                "2026-06-20T01:00:00Z",
                "2026-06-21T00:00:00Z",
            ],
            "title": [
                "Oil prices surge",
                "Middle East tensions rise",
                "Markets remain calm",
            ],
            "summary": [
                "Supply concerns support crude prices.",
                "Investors fear further escalation.",
                "Trading activity is stable.",
            ],
        }
    )

    result = build_sentiment(news)

    assert "date" in result.columns
    assert "sentiment" in result.columns
    assert len(result) == 2
    assert result["sentiment"].between(-1, 1).all()
