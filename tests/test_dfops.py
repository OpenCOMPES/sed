"""This file contains code that performs several tests for the dfops functions
"""
import numpy as np
import pandas as pd

from sed.mpes.core.dfops import apply_filter
from sed.mpes.core.dfops import apply_jitter
from sed.mpes.core.dfops import drop_column
from sed.mpes.core.dfops import map_columns_2d

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


def test_drop_column():
    """Test function to drop a df column."""
    column_name = "energy"
    df_dropped = drop_column(df, column_name=column_name)
    assert "energy" in df.columns
    assert "energy" not in df_dropped.columns


def test_apply_filter():
    """Test function to filter a df by a column with upper/lower bounds."""
    colname = "posx"
    lower_bound = -0.5
    upper_bound = 0.5
    df_filtered = apply_filter(
        df,
        col=colname,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    assert np.any(df[colname] < lower_bound)
    assert np.any(df[colname] > upper_bound)
    assert np.all(df_filtered[colname] > lower_bound)
    assert np.all(df_filtered[colname] < upper_bound)


def test_map_columns_2d():
    """Test mapping of a 2D-function onto the df."""

    def swap(x, y):
        return (y, x)

    x_column = "posx"
    y_column = "posy"
    new_x_column = "posx"
    new_y_column = "posy"
    df_swapped = map_columns_2d(
        df,
        map_2d=swap,
        x_column=x_column,
        y_column=y_column,
        new_x_column=new_x_column,
        new_y_column=new_y_column,
    )
    assert np.all(df[x_column] == df_swapped[new_x_column])
    assert np.all(df[y_column] == df_swapped[new_y_column])
