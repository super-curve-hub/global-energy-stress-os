import pandas as pd

from engine.utils import ewma_zscore


def test_ewma_zscore():

    s = pd.Series([1, 2, 3, 4, 5])

    z = ewma_zscore(s)

    assert len(z) == 5

    assert z.isna().sum() == 0
