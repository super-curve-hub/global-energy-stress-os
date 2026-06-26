from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np
import pandas as pd


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for p in ["data/raw", "data/cache", "data/processed", "outputs", "docs"]:
        Path(p).mkdir(parents=True, exist_ok=True)


def ewma_zscore(x, span: int = 60, clip: float = 3.0) -> pd.Series:
    s = pd.Series(x).astype(float)
    mean = s.ewm(span=span, min_periods=max(10, span // 3)).mean()
    std = s.ewm(span=span, min_periods=max(10, span // 3)).std()
    z = (s - mean) / std
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip, clip)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def bounded_series(x, clip: float = 3.0) -> pd.Series:
    return pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip, clip)
