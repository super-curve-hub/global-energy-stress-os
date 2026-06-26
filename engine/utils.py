"""
Global Energy Stress OS

Common utility functions.

Author: Super Curve
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ==========================================================
# Project Root
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

# ==========================================================
# Config
# ==========================================================


def load_config(path: str | Path = "config.yaml") -> dict:
    """
    Load YAML configuration file.
    """

    config_path = ROOT / path

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==========================================================
# Directories
# ==========================================================


def ensure_dirs() -> None:
    """
    Create required project directories.
    """

    directories = [
        "data/raw",
        "data/cache",
        "data/processed",
        "outputs",
        "docs",
        "logs",
    ]

    for d in directories:
        (ROOT / d).mkdir(
            parents=True,
            exist_ok=True,
        )


# ==========================================================
# Logger
# ==========================================================


def get_logger() -> logging.Logger:
    """
    Global project logger.
    """

    logger = logging.getLogger("global_energy_stress")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    #
    # Console
    #
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    #
    # Ensure logs directory exists
    #
    log_dir = ROOT / "logs"

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # File
    #
    logfile = log_dir / "global_energy_stress.log"

    file_handler = logging.FileHandler(
        logfile,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.info("Logger initialized.")

    return logger


# ==========================================================
# Statistics
# ==========================================================


def ewma_zscore(
    x,
    span: int = 60,
    clip: float = 3.0,
) -> pd.Series:
    """
    EWMA based rolling z-score.
    """

    s = pd.Series(x).astype(float)

    mean = s.ewm(
        span=span,
        min_periods=max(10, span // 3),
    ).mean()

    std = s.ewm(
        span=span,
        min_periods=max(10, span // 3),
    ).std()

    z = (s - mean) / std

    return (
        z.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-clip, clip)
    )


def bounded_series(
    x,
    clip: float = 3.0,
) -> pd.Series:
    """
    Replace NaN/Inf and clip values.
    """

    return (
        pd.Series(x)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-clip, clip)
    )


# ==========================================================
# Math
# ==========================================================


def sigmoid(
    x: float | np.ndarray,
):
    """
    Sigmoid function.
    """

    return 1.0 / (1.0 + np.exp(-x))
