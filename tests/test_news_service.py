from __future__ import annotations

from engine.news.service import fetch_news


def test_google_news(monkeypatch):
    def fake_google(query):
        return 5

    monkeypatch.setattr(
        "engine.news.service.google_news_count",
        fake_google,
    )

    cfg = {
        "news": {
            "google_queries": ["Hormuz"],
            "rss_feeds": [],
            "gdelt_queries": [],
        }
    }

    news, gdelt = fetch_news(cfg)

    assert len(news) == 1
    assert gdelt.empty

    row = news.iloc[0]
    assert row["source"] == "google_news"
    assert row["query"] == "Hormuz"
    assert row["count"] == 5


def test_rss(monkeypatch):
    def fake_rss(url):
        return 3

    monkeypatch.setattr(
        "engine.news.service.rss_count",
        fake_rss,
    )

    cfg = {
        "news": {
            "google_queries": [],
            "rss_feeds": [
                "https://example.com/rss",
            ],
            "gdelt_queries": [],
        }
    }

    news, gdelt = fetch_news(cfg)

    assert len(news) == 1
    assert gdelt.empty

    row = news.iloc[0]
    assert row["source"] == "rss"
    assert row["query"] == "https://example.com/rss"
    assert row["count"] == 3


def test_gdelt(monkeypatch):
    def fake_gdelt(query):
        return 42

    monkeypatch.setattr(
        "engine.news.service.gdelt_count",
        fake_gdelt,
    )

    cfg = {
        "news": {
            "google_queries": [],
            "rss_feeds": [],
            "gdelt_queries": ["Hormuz"],
        }
    }

    news, gdelt = fetch_news(cfg)

    assert news.empty
    assert len(gdelt) == 1

    row = gdelt.iloc[0]
    assert row["source"] == "gdelt"
    assert row["query"] == "Hormuz"
    assert row["count"] == 42


def test_empty_config():
    cfg = {"news": {}}

    news, gdelt = fetch_news(cfg)

    assert news.empty
    assert gdelt.empty
