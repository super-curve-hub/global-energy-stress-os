import pandas as pd

from engine.features.gdelt import build_gdelt_index


def test_build_gdelt_index():

    gdelt = pd.DataFrame(
        {
            "date": [
                "2026-06-20",
                "2026-06-21",
                "2026-06-22",
                "2026-06-23",
                "2026-06-24",
            ],
            "GDELT_News": [
                10,
                15,
                12,
                40,
                18,
            ],
        }
    )

    result = build_gdelt_index(gdelt)

    assert "date" in result.columns
    assert "gdelt_news" in result.columns
    assert "gdelt_zscore" in result.columns
    assert "gdelt_surprise" in result.columns

    assert len(result) == 5

    assert result["gdelt_surprise"].isin([0, 1]).all()
