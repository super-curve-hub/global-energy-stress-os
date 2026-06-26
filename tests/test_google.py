from engine.news.google import fetch_google_news


def test_google_news():

    df = fetch_google_news(["Hormuz Strait oil"])

    assert len(df) > 0

    assert "title" in df.columns

    assert "published" in df.columns

    assert "source" in df.columns
