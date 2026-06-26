from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from engine.utils import ewma_zscore, bounded_series


def build_market_features(px: pd.DataFrame, z_span: int = 60, z_clip: float = 3.0) -> pd.DataFrame:
    df = pd.DataFrame(index=px.index)
    ret = np.log(px / px.shift(1))

    def col(name):
        return px[name] if name in px.columns else pd.Series(np.nan, index=px.index)

    df["Brent_WTI"] = col("BRENT") - col("WTI")
    df["Brent_WTI_z"] = ewma_zscore(df["Brent_WTI"], z_span, z_clip)

    rv10 = ret.get("WTI", pd.Series(index=px.index, dtype=float)).rolling(10).std() * np.sqrt(252)
    rv60 = ret.get("WTI", pd.Series(index=px.index, dtype=float)).rolling(60).std() * np.sqrt(252)
    df["Vol_Regime_z"] = ewma_zscore(rv10 - rv60, z_span, z_clip)

    macro_cols = [c for c in ["DXY", "SPX", "USDJPY"] if c in ret.columns]
    resid = pd.Series(0.0, index=px.index)
    try:
        reg = pd.concat([ret["WTI"].rename("WTI"), ret[macro_cols]], axis=1).dropna()
        if len(reg) > 80 and len(macro_cols) >= 2:
            model = HuberRegressor().fit(reg[macro_cols].values, reg["WTI"].values)
            resid.loc[reg.index] = reg["WTI"].values - model.predict(reg[macro_cols].values)
    except Exception as e:
        print(f"[features] huber failed: {e}")
    df["Oil_Macro_Residual_z"] = ewma_zscore(resid, z_span, z_clip)

    df["Gold_Oil_z"] = ewma_zscore(col("GOLD") / col("WTI"), z_span, z_clip)
    if "XLE" in ret.columns and "SPX" in ret.columns:
        df["Energy_Relative_z"] = ewma_zscore(ret["XLE"].rolling(20).sum() - ret["SPX"].rolling(20).sum(), z_span, z_clip)
    else:
        df["Energy_Relative_z"] = 0.0
    if "TANKER_PROXY" in ret.columns:
        df["Tanker_Stress_z"] = ewma_zscore(ret["TANKER_PROXY"].rolling(5).sum(), z_span, z_clip)
    else:
        df["Tanker_Stress_z"] = 0.0
    df["Dollar_Stress_z"] = ewma_zscore(ret.get("DXY", pd.Series(index=px.index, dtype=float)).rolling(5).sum(), z_span, z_clip)
    df["Rates_Vol_z"] = ewma_zscore(col("MOVE"), z_span, z_clip)
    df["Oil_Vol_z"] = ewma_zscore(col("OVX"), z_span, z_clip)
    df["VIX_z"] = ewma_zscore(col("VIX"), z_span, z_clip)
    df["VVIX_z"] = ewma_zscore(col("VVIX"), z_span, z_clip)
    df["SKEW_z"] = ewma_zscore(col("SKEW"), z_span, z_clip)
    df["Option_Skew_z"] = df[["Oil_Vol_z", "VVIX_z", "SKEW_z"]].mean(axis=1).fillna(0.0).clip(-z_clip, z_clip)
    return df.fillna(0.0)


def add_news_features(df: pd.DataFrame, gdelt_index: pd.DataFrame, classified_articles: pd.DataFrame, z_span: int = 60, z_clip: float = 3.0) -> pd.DataFrame:
    out = df.copy()
    gd = gdelt_index.reindex(out.index, method="ffill").fillna(0.0)
    out["GDELT_News_z"] = ewma_zscore(gd.iloc[:, 0], z_span, z_clip)

    if len(classified_articles) > 0:
        today = out.index[-1]
        vals = {
            "News_Shipping_Score": classified_articles["shipping"].mean() / 10 * z_clip,
            "News_Energy_Score": classified_articles["energy"].mean() / 10 * z_clip,
            "News_Military_Score": classified_articles["military"].mean() / 10 * z_clip,
            "News_Physical_Score": classified_articles["physical"].mean() / 10 * z_clip,
            "News_Confidence": classified_articles["confidence"].mean(),
        }
        for k, v in vals.items():
            out[k] = 0.0
            out.loc[today, k] = float(v) if pd.notna(v) else 0.0
    else:
        for k in ["News_Shipping_Score", "News_Energy_Score", "News_Military_Score", "News_Physical_Score", "News_Confidence"]:
            out[k] = 0.0

    out["NLP_War_Intensity_z"] = bounded_series(
        0.30 * out["GDELT_News_z"]
        + 0.25 * out["News_Shipping_Score"]
        + 0.25 * out["News_Military_Score"]
        + 0.20 * out["News_Energy_Score"],
        z_clip,
    )
    return out.fillna(0.0)
