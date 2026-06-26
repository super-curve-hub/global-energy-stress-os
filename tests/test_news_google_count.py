from __future__ import annotations

from types import SimpleNamespace

from engine.news.service import google_news_count


def test_google_news_count(monkeypatch):
    def fake_parse(url):
        return SimpleNamespace(entries=[1, 2, 3])

    monkeypatch.setattr(
        "engine.news.service.feedparser.parse",
        fake_parse,
    )

    assert google_news_count("Hormuz") == 3


def test_google_news_exception(monkeypatch):
    def broken_parse(url):
        raise RuntimeError("network")

    monkeypatch.setattr(
        "engine.news.service.feedparser.parse",
        broken_parse,
    )

    assert google_news_count("Hormuz") == 0
