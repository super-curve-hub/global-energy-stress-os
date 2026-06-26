from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_commentary(latest: pd.Series) -> str:
    wp = latest.get("War_Premium_Score", 0.0)
    hp = latest.get("Hormuz_Closure_Prob", 0.0)
    regime = latest.get("Regime", "Unknown")
    if wp < 0:
        base = "市場は地政学プレミアムをほぼ織り込んでいません。"
    elif wp < 0.5:
        base = "ヘッドラインリスクは残るものの、現物供給ショックは限定的です。"
    elif wp < 1.2:
        base = "戦争コンベクシティが上昇しており、原油・海運・オプション市場の確認が必要です。"
    elif wp < 2.0:
        base = "物理的な供給制約が市場価格に波及し始めている可能性があります。"
    else:
        base = "グローバルなエネルギー危機レジームを警戒する局面です。"
    return f"{base} ホルムズ閉鎖確率は {hp:.1%}、レジームは {regime} です。"


def save_markdown(df: pd.DataFrame, articles: pd.DataFrame, output_path: str = "outputs/dashboard.md") -> None:
    latest = df.iloc[-1]
    lines = []
    lines.append("# Global Energy Stress OS")
    lines.append("")
    lines.append(f"**Date:** {latest.name.date()}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **War Premium Score:** {latest['War_Premium_Score']:.2f}")
    lines.append(f"- **Hormuz Closure Probability:** {latest['Hormuz_Closure_Prob']:.2%}")
    lines.append(f"- **Regime:** {latest['Regime']}")
    lines.append("")
    lines.append("## Stress Indices")
    lines.append("")
    for col in [
        "Shipping_Stress_Index",
        "Energy_Supply_Stress_Index",
        "Military_Escalation_Index",
        "Physical_Supply_Stress_Index",
        "Financial_Market_Stress_Index",
        "News_Credibility_Index",
    ]:
        lines.append(f"- **{col}:** {latest.get(col, 0.0):.2f}")
    lines.append("")
    lines.append("## AI Commentary")
    lines.append("")
    lines.append(build_commentary(latest))
    lines.append("")
    lines.append("## Latest Headlines")
    lines.append("")
    if len(articles) == 0:
        lines.append("No RSS headlines collected.")
    else:
        for _, row in articles.head(10).iterrows():
            title = str(row.get("title", "")).replace("\n", " ")
            source = row.get("source", "")
            link = row.get("link", "")
            lines.append(f"- [{title}]({link}) — {source}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
