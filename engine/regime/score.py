from __future__ import annotations

import pandas as pd
from engine.utils import sigmoid


def _weighted(df: pd.DataFrame, weights: dict[str, float], clip: float = 3.0) -> pd.Series:
    cols = [c for c in weights if c in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index)
    wsum = sum(abs(weights[c]) for c in cols) or 1.0
    s = sum(df[c].fillna(0.0) * (weights[c] / wsum) for c in cols)
    return s.clip(-clip, clip)


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Shipping_Stress_Index"] = _weighted(out, {
        "Tanker_Stress_z": 0.30,
        "Brent_WTI_z": 0.15,
        "Oil_Vol_z": 0.15,
        "News_Shipping_Score": 0.25,
        "NLP_War_Intensity_z": 0.15,
    })

    out["Energy_Supply_Stress_Index"] = _weighted(out, {
        "Brent_WTI_z": 0.20,
        "Oil_Macro_Residual_z": 0.20,
        "Oil_Vol_z": 0.20,
        "Energy_Relative_z": 0.15,
        "Gold_Oil_z": 0.10,
        "News_Energy_Score": 0.15,
    })

    out["Military_Escalation_Index"] = _weighted(out, {
        "News_Military_Score": 0.35,
        "NLP_War_Intensity_z": 0.20,
        "Gold_Oil_z": 0.15,
        "Oil_Vol_z": 0.15,
        "Dollar_Stress_z": 0.10,
        "Rates_Vol_z": 0.05,
    })

    out["Physical_Supply_Stress_Index"] = _weighted(out, {
        "News_Physical_Score": 0.30,
        "Brent_WTI_z": 0.20,
        "Oil_Macro_Residual_z": 0.20,
        "Energy_Relative_z": 0.15,
        "Oil_Vol_z": 0.15,
    })

    out["Financial_Market_Stress_Index"] = _weighted(out, {
        "Oil_Vol_z": 0.25,
        "Rates_Vol_z": 0.20,
        "Dollar_Stress_z": 0.20,
        "Gold_Oil_z": 0.15,
        "Option_Skew_z": 0.20,
    })

    out["News_Credibility_Index"] = _weighted(out, {
        "NLP_War_Intensity_z": 0.20,
        "Shipping_Stress_Index": 0.25,
        "Energy_Supply_Stress_Index": 0.25,
        "Physical_Supply_Stress_Index": 0.15,
        "Financial_Market_Stress_Index": 0.15,
    })

    out["War_Premium_Score"] = _weighted(out, {
        "Shipping_Stress_Index": 0.20,
        "Energy_Supply_Stress_Index": 0.20,
        "Military_Escalation_Index": 0.20,
        "Physical_Supply_Stress_Index": 0.15,
        "Financial_Market_Stress_Index": 0.15,
        "News_Credibility_Index": 0.10,
    })

    logit = (
        0.28 * out["Shipping_Stress_Index"]
        + 0.24 * out["Energy_Supply_Stress_Index"]
        + 0.22 * out["Military_Escalation_Index"]
        + 0.16 * out["Physical_Supply_Stress_Index"]
        + 0.10 * out["Oil_Vol_z"]
    )
    out["Hormuz_Closure_Prob"] = sigmoid(logit - 1.35)
    out["Regime"] = out["War_Premium_Score"].apply(regime_label)
    return out


def regime_label(x: float) -> str:
    if x < 0:
        return "Calm"
    if x < 0.5:
        return "Headline Repricing"
    if x < 1.2:
        return "War Convexity"
    if x < 2.0:
        return "Physical Disruption"
    return "Global Energy Crisis"
