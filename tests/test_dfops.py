"""This file contains code that performs several tests for the dfops functions
"""
import numpy as np
import pandas as pd
import dask.dataframe as ddf
import pytest

from sed.core.dfops import apply_filter
from sed.core.dfops import apply_jitter
from sed.core.dfops import drop_column
from sed.core.dfops import map_columns_2d
from sed.core.dfops import forward_fill_lazy


N_PTS = 100
N_PARTITIONS = 10
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


def test_forward_fill_lazy_sparse_nans():
    """ test that a lazy forward fill works as expected with sparse nans"""
    t_df = df.copy()
    t_df['energy'][::2] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    t_dask_df = forward_fill_lazy(t_dask_df, 'energy', before='max')
    t_df = t_df.ffill()
    pd.testing.assert_frame_equal(t_df, t_dask_df.compute())


def test_forward_fill_lazy_full_partition_nans():
    """ test that a lazy forward fill works as expected with a full partition of nans"""
    t_df = df.copy()
    t_df['energy'][5:25] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    t_dask_df = forward_fill_lazy(t_dask_df, 'energy', before='max')
    t_df = t_df.ffill()
    pd.testing.assert_frame_equal(t_df, t_dask_df.compute())


def test_forward_fill_lazy_consecutive_full_partition_nans():
    """ test that a lazy forward fill fails as expected on two consecutive partitions
    full of nans
    """
    t_df = df.copy()
    t_df['energy'][5:35] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    t_dask_df = forward_fill_lazy(t_dask_df, 'energy', before='max')
    t_df = t_df.ffill()
    assert not t_df.equals(t_dask_df.compute())


def test_forward_fill_lazy_wrong_parameters():
    """ test that a lazy forward fill fails as expected on wrong parameters"""
    t_df = df.copy()
    t_df['energy'][5:35] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    with pytest.raises(TypeError):
        t_dask_df = forward_fill_lazy(t_dask_df, 'energy', before='wrong parameter')


def test_forward_fill_lazy_compute():
    """ test that a lazy forward fill works as expected with compute=True"""
    t_df = df.copy()
    t_df['energy'][5:35] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    t_dask_df_comp = forward_fill_lazy(t_dask_df, 'energy', before='max', compute_lengths=True)
    t_dask_df_nocomp = forward_fill_lazy(t_dask_df, 'energy', before='max', compute_lengths=False)
    pd.testing.assert_frame_equal(t_dask_df_comp.compute(), t_dask_df_nocomp.compute())


def test_forward_fill_lazy_keep_head_nans():
    """ test that a lazy forward fill works as expected with missing values at the 
    beginning of the dataframe"""
    t_df = df.copy()
    t_df['energy'][:5] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    t_df = forward_fill_lazy(t_dask_df, 'energy', before='max').compute()
    assert np.all(np.isnan(t_df['energy'][:5]))
    assert np.all(np.isfinite(t_df['energy'][5:]))
