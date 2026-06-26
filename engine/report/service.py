from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
        for _, row in news.sort_values("count", ascending=False).iterrows():
            lines.append(f"- {row['source']} / {row['query']}: {int(row['count'])}")
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


def build_report(
    df: pd.DataFrame,
    news: pd.DataFrame,
    output_dir: Path,
    docs_dir: Path,
    csv_name: str = "dashboard.csv",
    md_name: str = "dashboard.md",
    html_name: str = "dashboard.html",
    png_name: str = "dashboard.png",
) -> None:
    df.to_csv(output_dir / csv_name)
    news.to_csv(output_dir / "news_counts.csv", index=False)

    make_png(df, output_dir / png_name)
    make_markdown(df, news, output_dir / md_name)
    make_html(output_dir / md_name, output_dir / html_name)

    (docs_dir / "index.html").write_text(
        (output_dir / html_name).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if (output_dir / png_name).exists():
        (docs_dir / "dashboard.png").write_bytes((output_dir / png_name).read_bytes())
