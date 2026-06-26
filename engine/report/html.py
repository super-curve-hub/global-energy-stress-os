from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_html(df: pd.DataFrame, output_path: str = "outputs/dashboard.html") -> None:
    latest = df.iloc[-1]
    html = f"""
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Global Energy Stress OS</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; background:#0f172a; color:#e5e7eb; }}
.card {{ background:#111827; border:1px solid #334155; border-radius:16px; padding:18px; margin:14px 0; }}
h1, h2 {{ margin: 0 0 12px; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
.metric {{ font-size: 30px; font-weight: 700; }}
.label {{ color:#94a3b8; font-size: 13px; }}
img {{ max-width:100%; border-radius:12px; background:#fff; }}
</style>
</head>
<body>
<h1>Global Energy Stress OS</h1>
<p>{latest.name.date()}</p>
<div class="grid">
  <div class="card"><div class="label">War Premium Score</div><div class="metric">{latest['War_Premium_Score']:.2f}</div></div>
  <div class="card"><div class="label">Hormuz Closure Probability</div><div class="metric">{latest['Hormuz_Closure_Prob']:.1%}</div></div>
  <div class="card"><div class="label">Regime</div><div class="metric">{latest['Regime']}</div></div>
</div>
<div class="card"><h2>Charts</h2><img src="war_premium_score.png"><br><br><img src="hormuz_probability.png"><br><br><img src="stress_subindices.png"></div>
</body>
</html>
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
