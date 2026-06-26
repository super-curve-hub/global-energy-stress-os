from __future__ import annotations

from engine.news.service import gdelt_count


class DummyResponse:
    status_code = 200

    def json(self):
        return {
            "timeline": [
                {"value": 10},
                {"value": 5},
            ]
        }


class Dummy429:
    status_code = 429

    def json(self):
        return {}


def test_gdelt_ok(monkeypatch):
    monkeypatch.setattr(
        "engine.news.service.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )

    assert gdelt_count("Hormuz") == 15


def test_gdelt_429(monkeypatch):
    monkeypatch.setattr(
        "engine.news.service.requests.get",
        lambda *args, **kwargs: Dummy429(),
    )

    assert gdelt_count("Hormuz") == 0


def test_gdelt_exception(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("network")

    monkeypatch.setattr(
        "engine.news.service.requests.get",
        broken,
    )

    assert gdelt_count("Hormuz") == 0
