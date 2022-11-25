"""This file contains code that performs several tests for the dfops functions
"""
import numpy as np
import pandas as pd

from sed.core.dfops import apply_jitter

N_PTS = 100
cols = ["posx", "posy", "energy"]
df = pd.DataFrame(np.random.randn(N_PTS, len(cols)), columns=cols)


def test_apply_jitter():
    """This function tests if the apply_jitter function generates the correct
    dataframe column.
    """
    cols_jittered = [col + "_jittered" for col in cols]
    df_jittered = apply_jitter(df, cols=cols, cols_jittered=cols_jittered)
    assert isinstance(df_jittered, pd.DataFrame)
    for col in cols_jittered:
        assert col in df_jittered.columns
