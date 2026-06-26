from __future__ import annotations

from engine.features import build_features
from engine.market import fetch_market
from engine.news import fetch_news
from engine.regime import build_regime
from engine.report import build_report
from engine.utils import (
    ROOT,
    ensure_dirs,
    get_logger,
    load_config,
)

logger = get_logger()


def main() -> None:
    """
    Global Energy Stress OS

    Pipeline

        Config
            ↓
        Market Layer
            ↓
        News Layer
            ↓
        Feature Layer
            ↓
        Regime Layer
            ↓
        Report Layer
    """

    cfg = load_config()
    ensure_dirs()

    output_cfg = cfg.get("outputs", {})

    output_dir = ROOT / output_cfg.get(
        "output_dir",
        "outputs",
    )

    docs_dir = ROOT / output_cfg.get(
        "docs_dir",
        "docs",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    docs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("=" * 60)
    logger.info("Global Energy Stress OS")
    logger.info("=" * 60)

    #
    # Market Layer
    #
    market = fetch_market(cfg)

    #
    # News Layer
    #
    news, gdelt = fetch_news(cfg)

    #
    # Feature Layer
    #
    features = build_features(
        news=news,
        gdelt=gdelt,
    )

    #
    # Regime Layer
    #
    regime = build_regime(
        market=market,
        features=features,
        cfg=cfg,
    )

    #
    # Save processed data
    #
    processed_dir = ROOT / "data" / "processed"

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.dataframe.to_csv(
        processed_dir / "features.csv",
        index=False,
    )

    regime.to_csv(
        processed_dir / "regime.csv",
    )

    #
    # Report Layer
    #
    build_report(
        df=regime,
        news=news,
        output_dir=output_dir,
        docs_dir=docs_dir,
        csv_name=output_cfg.get(
            "csv",
            "dashboard.csv",
        ),
        md_name=output_cfg.get(
            "markdown",
            "dashboard.md",
        ),
        html_name=output_cfg.get(
            "html",
            "dashboard.html",
        ),
        png_name=output_cfg.get(
            "png",
            "dashboard.png",
        ),
    )

    latest = regime.iloc[-1]

    logger.info("Completed.")
    logger.info(
        "Date: %s",
        latest.name.date(),
    )
    logger.info(
        "War Premium Score: %.3f",
        float(latest["War_Premium_Score"]),
    )
    logger.info(
        "Hormuz Closure Probability: %.2f %%",
        float(latest["Hormuz_Closure_Prob"] * 100),
    )
    logger.info(
        "Regime: %s",
        latest["Regime"],
    )


if __name__ == "__main__":
    main()
