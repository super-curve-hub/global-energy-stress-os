from __future__ import annotations

import json
import re
import requests
import pandas as pd

SHIPPING_TERMS = ["tanker", "vlcc", "ship", "shipping", "strait", "hormuz", "red sea", "reroute", "vessel", "port", "ais"]
ENERGY_TERMS = ["oil", "crude", "brent", "wti", "opec", "saudi", "osp", "supply", "export", "refinery", "gas", "lng"]
MILITARY_TERMS = ["iran", "israel", "missile", "drone", "navy", "attack", "strike", "military", "war", "seized", "threat"]
PHYSICAL_TERMS = ["premium", "insurance", "freight", "spread", "backwardation", "contango", "inventory", "loading", "delay", "disruption"]
NEGATIVE_TERMS = ["ceasefire", "resume", "reopen", "de-escalation", "eases", "normal", "agreement", "waiver"]


def _score_text(text: str) -> dict:
    t = text.lower()
    def count(terms):
        return sum(1 for w in terms if w in t)
    shipping = min(10, count(SHIPPING_TERMS) * 2)
    energy = min(10, count(ENERGY_TERMS) * 2)
    military = min(10, count(MILITARY_TERMS) * 2)
    physical = min(10, count(PHYSICAL_TERMS) * 2)
    deesc = count(NEGATIVE_TERMS)
    if deesc > 0:
        shipping = max(0, shipping - deesc * 2)
        energy = max(0, energy - deesc * 2)
        military = max(0, military - deesc * 3)
        physical = max(0, physical - deesc * 2)
    conf = min(0.95, 0.35 + 0.06 * (shipping + energy + military + physical))
    return {
        "shipping": shipping,
        "energy": energy,
        "military": military,
        "physical": physical,
        "confidence": round(conf, 2),
        "summary": text[:220],
        "method": "rule_based",
    }


def classify_with_ollama(text: str, endpoint: str, model: str, timeout_sec: int = 30) -> dict | None:
    prompt = f"""
You are an energy and shipping geopolitical risk classifier.
Return JSON only with keys: shipping, energy, military, physical, confidence, summary.
Scores must be 0-10.
Text: {text[:3000]}
""".strip()
    try:
        r = requests.post(endpoint, json={"model": model, "prompt": prompt, "stream": False}, timeout=timeout_sec)
        if r.status_code != 200:
            return None
        raw = r.json().get("response", "")
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return None
        return json.loads(m.group(0))
    except Exception as e:
        print(f"[ollama] failed: {e}")
        return None


def classify_articles(articles: pd.DataFrame, llm_cfg: dict) -> pd.DataFrame:
    rows = []
    use_ollama = bool(llm_cfg.get("use_ollama", False))
    for _, row in articles.iterrows():
        text = f"{row.get('title','')} {row.get('summary','')}"
        result = None
        if use_ollama:
            result = classify_with_ollama(
                text,
                endpoint=llm_cfg.get("endpoint", "http://localhost:11434/api/generate"),
                model=llm_cfg.get("model", "qwen2.5:7b"),
                timeout_sec=int(llm_cfg.get("timeout_sec", 30)),
            )
        if result is None:
            result = _score_text(text)
        rows.append({**row.to_dict(), **result})
    if not rows:
        return pd.DataFrame(columns=list(articles.columns) + ["shipping", "energy", "military", "physical", "confidence", "summary", "method"])
    return pd.DataFrame(rows)
