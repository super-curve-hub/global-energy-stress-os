from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_charts(df: pd.DataFrame, output_dir: str = "outputs") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["War_Premium_Score"], label="War Premium Score")
    plt.axhline(0.5, linestyle="--")
    plt.axhline(1.2, linestyle="--")
    plt.axhline(2.0, linestyle="--")
    plt.title("Global Energy Stress OS - War Premium Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "war_premium_score.png", dpi=160)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["Hormuz_Closure_Prob"], label="Hormuz Closure Probability")
    plt.axhline(0.15, linestyle="--")
    plt.axhline(0.30, linestyle="--")
    plt.axhline(0.50, linestyle="--")
    plt.title("Hormuz Closure Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "hormuz_probability.png", dpi=160)
    plt.close()

    sub_cols = [
        "Shipping_Stress_Index",
        "Energy_Supply_Stress_Index",
        "Military_Escalation_Index",
        "Physical_Supply_Stress_Index",
        "Financial_Market_Stress_Index",
        "News_Credibility_Index",
    ]
    plt.figure(figsize=(14, 7))
    for col in sub_cols:
        if col in df.columns:
            plt.plot(df.index, df[col], label=col)
    plt.axhline(0, linestyle="--")
    plt.axhline(1, linestyle="--")
    plt.axhline(2, linestyle="--")
    plt.title("Stress Sub-Indices")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "stress_subindices.png", dpi=160)
    plt.close()
