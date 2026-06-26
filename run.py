from __future__ import annotations

import math
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote

import feedparser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parent


# ============================================================
# CONFIG / UTILS
# ============================================================

def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict) -> None:
    output_dir = cfg.get("outputs", {}).get("output_dir", "outputs")
    docs_dir = cfg.get("outputs", {}).get("docs_dir", "docs")

    for p in [
        output_dir,
        docs_dir,
        "data/raw",
        "data/processed",
        "data/cache",
    ]:
        (ROOT / p).mkdir(parents=True, exist_ok=True)


def ewma_zscore(x: pd.Series, span: int = 60, clip: float = 3.0) -> pd.Series:
    x = pd.Series(x).astype(float)
    mean = x.ewm(span=span, min_periods=20).mean()
    std = x.ewm(span=span, min_periods=20).std()
    z = (x - mean) / std
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip, clip)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ============================================================
# MARKET LAYER
# ============================================================

def safe_download(ticker: str, start: str) -> pd.Series | None:
    try:
        data = yf.download(
            ticker,
            start=start,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data is None or len(data) == 0:
            print(f"WARN: empty download: {ticker}")
            return None

        close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close.name = ticker
        return close.dropna()

    except Exception as e:
        print(f"WARN: download failed {ticker}: {e}")
        return None


def fetch_market(cfg: dict) -> pd.DataFrame:
    start = cfg["project"].get("start_date", "2021-01-01")
    tickers = cfg["market"]["tickers"]

    series = {}

    for name, ticker in tickers.items():
        s = safe_download(ticker, start)
        if s is not None:
            series[name] = s

    if not series:
        raise RuntimeError("No market data downloaded")

    px = pd.concat(series, axis=1).sort_index().ffill()

    required = [c for c in ["WTI", "BRENT", "DXY", "SPX", "USDJPY"] if c in px.columns]
    px = px.dropna(subset=required)

    return px


# ============================================================
# NEWS LAYER
# ============================================================

def google_news_count(query: str) -> int:
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        feed = feedparser.parse(url)
        return len(feed.entries)

    except Exception as e:
        print(f"WARN: Google RSS failed {query}: {e}")
        return 0


def gdelt_count(query: str) -> int:
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc?"
        f"query={quote(query)}"
        "&mode=timelinevolraw"
        "&format=json"
        "&timespan=7d"
    )

    try:
        r = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        if r.status_code != 200:
            print(f"WARN: GDELT status {r.status_code} for {query}")
            return 0

        data = r.json()
        timeline = data.get("timeline", [])

        return int(sum(float(x.get("value", 0)) for x in timeline))

    except Exception as e:
        print(f"WARN: GDELT failed {query}: {e}")
        return 0


def rss_count(feed_url: str) -> int:
    try:
        feed = feedparser.parse(feed_url)
        return len(feed.entries)

    except Exception as e:
        print(f"WARN: RSS failed {feed_url}: {e}")
        return 0


def fetch_headlines(cfg: dict) -> pd.DataFrame:
    rows = []
    today = pd.Timestamp.now("UTC").normalize()

    news_cfg = cfg.get("news", {})

    for q in news_cfg.get("google_queries", []):
        rows.append(
            {
                "date": today,
                "source": "google_news",
                "query": q,
                "count": google_news_count(q),
            }
        )

    for q in news_cfg.get("gdelt_queries", []):
        rows.append(
            {
                "date": today,
                "source": "gdelt",
                "query": q,
                "count": gdelt_count(q),
            }
        )

    for url in news_cfg.get("rss_feeds", []):
        rows.append(
            {
                "date": today,
                "source": "rss",
                "query": url,
                "count": rss_count(url),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FEATURE ENGINE
# ============================================================

def compute_features(px: pd.DataFrame, news: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    feature_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})

    span = int(feature_cfg.get("rolling_window", model_cfg.get("z_span", 60)))
    clip = float(feature_cfg.get("z_clip", model_cfg.get("z_clip", 3.0)))

    ret = np.log(px / px.shift(1))
    df = pd.DataFrame(index=px.index)

    def has(col: str) -> bool:
        return col in px.columns and px[col].notna().sum() > 30

    zero = pd.Series(0.0, index=df.index)

    # --------------------------------------------------------
    # Core market dislocations
    # --------------------------------------------------------

    if has("BRENT") and has("WTI"):
        df["Brent_WTI"] = px["BRENT"] - px["WTI"]
        df["Brent_WTI_z"] = ewma_zscore(df["Brent_WTI"], span, clip)
    else:
        df["Brent_WTI"] = 0.0
        df["Brent_WTI_z"] = 0.0

    if has("WTI"):
        rv10 = ret["WTI"].rolling(10).std() * np.sqrt(252)
        rv60 = ret["WTI"].rolling(60).std() * np.sqrt(252)
        df["Vol_Regime_z"] = ewma_zscore(rv10 - rv60, span, clip)
    else:
        df["Vol_Regime_z"] = 0.0

    macro = zero.copy()

    if has("DXY"):
        macro += 0.4 * ewma_zscore(ret["DXY"], span, clip)

    if has("SPX"):
        macro += 0.3 * ewma_zscore(ret["SPX"], span, clip)

    if has("USDJPY"):
        macro += 0.3 * ewma_zscore(ret["USDJPY"], span, clip)

    oil = ewma_zscore(ret["WTI"], span, clip) if has("WTI") else zero
    df["Oil_Macro_Residual_z"] = (oil - macro).clip(-clip, clip)

    df["Gold_Oil_z"] = (
        ewma_zscore(px["GOLD"] / px["WTI"], span, clip)
        if has("GOLD") and has("WTI")
        else 0.0
    )

    df["Energy_Relative_z"] = (
        ewma_zscore(ret["XLE"].rolling(20).sum() - ret["SPX"].rolling(20).sum(), span, clip)
        if has("XLE") and has("SPX")
        else 0.0
    )

    df["Dollar_Stress_z"] = (
        ewma_zscore(ret["DXY"].rolling(5).sum(), span, clip)
        if has("DXY")
        else 0.0
    )

    df["Rates_Vol_z"] = (
        ewma_zscore(px["MOVE"], span, clip)
        if has("MOVE")
        else 0.0
    )

    df["Oil_Vol_z"] = (
        ewma_zscore(px["OVX"], span, clip)
        if has("OVX")
        else 0.0
    )

    df["VIX_z"] = ewma_zscore(px["VIX"], span, clip) if has("VIX") else 0.0
    df["VVIX_z"] = ewma_zscore(px["VVIX"], span, clip) if has("VVIX") else 0.0
    df["SKEW_z"] = ewma_zscore(px["SKEW"], span, clip) if has("SKEW") else 0.0

    df["Option_Skew_z"] = (
        df[["Oil_Vol_z", "VIX_z", "VVIX_z", "SKEW_z"]]
        .mean(axis=1)
        .clip(-clip, clip)
    )

    # --------------------------------------------------------
    # News intensity
    # --------------------------------------------------------

    today_count = float(news["count"].sum()) if len(news) else 0.0

    nlp_raw = pd.Series(0.0, index=df.index)

    if len(nlp_raw):
        nlp_raw.iloc[-1] = math.log1p(today_count)

    df["NLP_War_Intensity_z"] = ewma_zscore(nlp_raw, span=20, clip=clip)

    # --------------------------------------------------------
    # Sub indices
    # --------------------------------------------------------

    df["Shipping_Stress_Index"] = (
        0.35 * df["Brent_WTI_z"]
        + 0.25 * df["Oil_Vol_z"]
        + 0.25 * df["NLP_War_Intensity_z"]
        + 0.15 * df["Dollar_Stress_z"]
    ).clip(-clip, clip)

    df["Energy_Supply_Stress_Index"] = (
        0.30 * df["Brent_WTI_z"]
        + 0.25 * df["Oil_Macro_Residual_z"]
        + 0.20 * df["Oil_Vol_z"]
        + 0.15 * df["Energy_Relative_z"]
        + 0.10 * df["Gold_Oil_z"]
    ).clip(-clip, clip)

    df["Military_Escalation_Index"] = (
        0.35 * df["NLP_War_Intensity_z"]
        + 0.20 * df["Gold_Oil_z"]
        + 0.20 * df["Oil_Vol_z"]
        + 0.15 * df["Dollar_Stress_z"]
        + 0.10 * df["Rates_Vol_z"]
    ).clip(-clip, clip)

    df["News_Credibility_Index"] = (
        0.40 * df["NLP_War_Intensity_z"]
        + 0.25 * df["Shipping_Stress_Index"]
        + 0.20 * df["Energy_Supply_Stress_Index"]
        + 0.15 * df["Option_Skew_z"]
    ).clip(-clip, clip)

    # --------------------------------------------------------
    # Final scores
    # --------------------------------------------------------

    df["War_Premium_Score"] = (
        0.22 * df["Shipping_Stress_Index"]
        + 0.22 * df["Energy_Supply_Stress_Index"]
        + 0.18 * df["Military_Escalation_Index"]
        + 0.18 * df["News_Credibility_Index"]
        + 0.08 * df["Option_Skew_z"]
        + 0.06 * df["Rates_Vol_z"]
        + 0.06 * df["Dollar_Stress_z"]
    ).clip(-clip, clip)

    intercept = float(model_cfg.get("hormuz_logit_intercept", -1.35))

    logit = (
        0.30 * df["Shipping_Stress_Index"]
        + 0.25 * df["Energy_Supply_Stress_Index"]
        + 0.20 * df["Military_Escalation_Index"]
        + 0.15 * df["News_Credibility_Index"]
        + 0.10 * df["Oil_Vol_z"]
        + intercept
    )

    df["Hormuz_Closure_Prob"] = sigmoid(logit)

    def regime(x: float) -> str:
        if x < 0:
            return "Calm"
        if x < 0.5:
            return "Headline Repricing"
        if x < 1.5:
            return "War Convexity"
        return "Crisis"

    df["Regime"] = df["War_Premium_Score"].apply(regime)

    return df


# ============================================================
# REPORT ENGINE
# ============================================================

def make_markdown(df: pd.DataFrame, news: pd.DataFrame, out_path: Path) -> None:
    latest = df.iloc[-1]
    now_jst = datetime.now(timezone.utc) + timedelta(hours=9)

    lines = [
        "# Global Energy Stress OS",
        "",
        f"Generated: {now_jst.strftime('%Y-%m-%d %H:%M JST')}",
        "",
        "## Summary",
        "",
        f"- War Premium Score: **{latest['War_Premium_Score']:.2f}**",
        f"- Hormuz Closure Probability: **{latest['Hormuz_Closure_Prob'] * 100:.1f}%**",
        f"- Regime: **{latest['Regime']}**",
        "",
        "## Stress Indices",
        "",
        f"- Shipping Stress: {latest['Shipping_Stress_Index']:.2f}",
        f"- Energy Supply Stress: {latest['Energy_Supply_Stress_Index']:.2f}",
        f"- Military Escalation: {latest['Military_Escalation_Index']:.2f}",
        f"- News Credibility: {latest['News_Credibility_Index']:.2f}",
        "",
        "## News Counts",
        "",
    ]

    if len(news):
        for _, r in news.sort_values("count", ascending=False).iterrows():
            lines.append(f"- {r['source']} / {r['query']}: {int(r['count'])}")
    else:
        lines.append("- No news data")

    lines += [
        "",
        "## Interpretation",
        "",
        "This is an automated free-data prototype using Yahoo Finance, Google News RSS, GDELT, and public RSS feeds.",
        "",
        "Current scoring is rule-based. Ollama-based local LLM classification will be added in a later phase.",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_html(md_path: Path, html_path: Path) -> None:
    md = md_path.read_text(encoding="utf-8")

    html = md
    html = re.sub(r"^# (.*)$", r"<h1>\1</h1>", html, flags=re.M)
    html = re.sub(r"^## (.*)$", r"<h2>\1</h2>", html, flags=re.M)
    html = re.sub(r"^### (.*)$", r"<h3>\1</h3>", html, flags=re.M)
    html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"^- (.*)$", r"<li>\1</li>", html, flags=re.M)
    html = html.replace("\n", "\n<br>")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Global Energy Stress OS</title>
<style>
body {{
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 32px;
    max-width: 960px;
    line-height: 1.6;
}}
img {{
    max-width: 100%;
    border-radius: 12px;
}}
code {{
    background: #f3f3f3;
    padding: 2px 4px;
    border-radius: 4px;
}}
</style>
</head>
<body>
{html}
<p><img src="dashboard.png" alt="dashboard"></p>
</body>
</html>"""

    html_path.write_text(page, encoding="utf-8")


def make_png(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.tail(260)

    fig = plt.figure(figsize=(13, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(plot_df.index, plot_df["War_Premium_Score"], label="War Premium Score")
    ax1.axhline(0.5, linestyle="--", linewidth=1)
    ax1.axhline(1.5, linestyle="--", linewidth=1)
    ax1.set_title("Global Energy Stress OS - War Premium")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(plot_df.index, plot_df["Shipping_Stress_Index"], label="Shipping")
    ax2.plot(plot_df.index, plot_df["Energy_Supply_Stress_Index"], label="Energy")
    ax2.plot(plot_df.index, plot_df["Military_Escalation_Index"], label="Military")
    ax2.plot(plot_df.index, plot_df["News_Credibility_Index"], label="News Credibility")
    ax2.axhline(0.0, linestyle="--", linewidth=1)
    ax2.set_title("Sub Indices")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    output_cfg = cfg.get("outputs", {})

    output_dir = ROOT / output_cfg.get("output_dir", "outputs")
    docs_dir = ROOT / output_cfg.get("docs_dir", "docs")

    output_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Global Energy Stress OS")
    print("=" * 60)

    px = fetch_market(cfg)
    news = fetch_headlines(cfg)
    df = compute_features(px, news, cfg)

    csv_name = output_cfg.get("csv", "dashboard.csv")
    md_name = output_cfg.get("markdown", "dashboard.md")
    html_name = output_cfg.get("html", "dashboard.html")
    png_name = output_cfg.get("png", "dashboard.png")

    df.to_csv(output_dir / csv_name)
    news.to_csv(output_dir / "news_counts.csv", index=False)
    df.to_csv(ROOT / "data" / "processed" / "features.csv")

    make_png(df, output_dir / png_name)
    make_markdown(df, news, output_dir / md_name)
    make_html(output_dir / md_name, output_dir / html_name)

    # GitHub Pages mirror
    (docs_dir / "index.html").write_text(
        (output_dir / html_name).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if (output_dir / png_name).exists():
        (docs_dir / "dashboard.png").write_bytes((output_dir / png_name).read_bytes())

    latest = df.iloc[-1]

    print("Completed.")
    print("Date:", latest.name.date())
    print("War Premium Score:", round(float(latest["War_Premium_Score"]), 3))
    print(
        "Hormuz Closure Probability:",
        round(float(latest["Hormuz_Closure_Prob"] * 100), 2),
        "%",
    )
    print("Regime:", latest["Regime"])


if __name__ == "__main__":
    main()
