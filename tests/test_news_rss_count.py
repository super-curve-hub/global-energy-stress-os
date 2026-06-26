from __future__ import annotations

from types import SimpleNamespace

from engine.news.service import rss_count


def test_rss_count(monkeypatch):
    def fake_parse(url):
        return SimpleNamespace(entries=[1, 2])

    monkeypatch.setattr(
        "engine.news.service.feedparser.parse",
        fake_parse,
    )

    assert rss_count("dummy") == 2


def test_rss_exception(monkeypatch):
    def broken(url):
        raise RuntimeError("network")

    monkeypatch.setattr(
        "engine.news.service.feedparser.parse",
        broken,
    )

    assert rss_count("dummy") == 0
