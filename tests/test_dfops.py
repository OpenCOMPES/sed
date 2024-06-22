"""This file contains code that performs several tests for the dfops functions
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import dask.dataframe as ddf
import numpy as np
import pandas as pd
import pytest

from sed.core.dfops import add_time_stamped_data
from sed.core.dfops import apply_filter
from sed.core.dfops import apply_jitter
from sed.core.dfops import backward_fill_lazy
from sed.core.dfops import drop_column
from sed.core.dfops import forward_fill_lazy
from sed.core.dfops import map_columns_2d
from sed.core.dfops import offset_by_other_columns


N_PTS = 100
N_PARTITIONS = 10
cols = ["posx", "posy", "energy"]
df = pd.DataFrame(np.random.randn(N_PTS, len(cols)), columns=cols)


def test_apply_jitter() -> None:
    """This function tests if the apply_jitter function generates the correct
    dataframe column.
    """
    cols_jittered = [col + "_jittered" for col in cols]
    df_jittered = apply_jitter(df, cols=cols, cols_jittered=cols_jittered)
    assert isinstance(df_jittered, pd.DataFrame)
    for col in cols_jittered:
        assert col in df_jittered.columns


def test_drop_column() -> None:
    """Test function to drop a df column."""
    column_name = "energy"
    df_dropped = drop_column(df, column_name=column_name)
    assert "energy" in df.columns
    assert "energy" not in df_dropped.columns


def test_apply_filter() -> None:
    """Test function to filter a df by a column with upper/lower bounds."""
    colname = "posx"
    lower_bound = -0.1
    upper_bound = 0.1
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


def test_apply_filter_index() -> None:
    """Test function to filter a df by a index with upper/lower bounds."""
    colname = "index"
    lower_bound = 3
    upper_bound = 6
    df_filtered = apply_filter(
        df,
        col=colname,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    np.testing.assert_allclose(df_filtered, df.iloc[lower_bound + 1 : upper_bound, :])

    dd: ddf.DataFrame = ddf.from_pandas(df, npartitions=N_PARTITIONS).reset_index(drop=True)
    dd_filtered = apply_filter(
        dd,
        col=colname,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    for i in range(dd.npartitions):
        np.testing.assert_allclose(
            dd_filtered.partitions[i].compute().values,
            dd.partitions[i].compute().values[lower_bound + 1 : upper_bound, :],
        )


def test_add_time_stamped_data() -> None:
    """Test the addition of time-stamped data to the df."""
    df_ts = df
    time_stamp = dt.datetime.now().timestamp()
    df_ts["timeStamps"] = time_stamp + np.linspace(0, 100, N_PTS)
    data = np.linspace(0, 1, 20)
    time_stamps = time_stamp + np.linspace(0, 100, 20)
    with pytest.raises(ValueError):
        add_time_stamped_data(
            df=df_ts,
            time_stamps=time_stamps,
            data=data,
            dest_column="time_stamped_data",
            time_stamp_column="timeStamps",
        )
    dd_ts = ddf.from_pandas(df_ts, npartitions=N_PARTITIONS)
    with pytest.raises(ValueError):
        add_time_stamped_data(
            df=dd_ts,
            time_stamps=time_stamps,
            data=data,
            dest_column="time_stamped_data",
            time_stamp_column="invalidColumn",
        )
    dd_ts = add_time_stamped_data(
        df=dd_ts,
        time_stamps=time_stamps,
        data=data,
        dest_column="time_stamped_data",
        time_stamp_column="timeStamps",
    )
    assert "time_stamped_data" in dd_ts
    res = dd_ts["time_stamped_data"].compute().values
    assert res[0] == 0
    assert res[-1] == 1
    with pytest.raises(ValueError):
        data = np.linspace(0, 1, 19)
        add_time_stamped_data(
            df=dd_ts,
            time_stamps=time_stamps,
            data=data,
            dest_column="time_stamped_data",
            time_stamp_column="timeStamps",
        )


def test_map_columns_2d() -> None:
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


def test_forward_fill_lazy_sparse_nans() -> None:
    """test that a lazy forward fill works as expected with sparse nans"""
    t_df = df.copy()
    t_df["energy"][::2] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    t_dask_df = forward_fill_lazy(t_dask_df, "energy", before="max")
    t_df = t_df.ffill()
    pd.testing.assert_frame_equal(t_df, t_dask_df.compute())


def test_forward_fill_lazy_full_partition_nans() -> None:
    """test that a lazy forward fill works as expected with a full partition of nans"""
    t_df = df.copy()
    t_df["energy"][5:25] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    t_dask_df = forward_fill_lazy(t_dask_df, "energy", before="max")
    t_df = t_df.ffill()
    pd.testing.assert_frame_equal(t_df, t_dask_df.compute())


def test_forward_fill_lazy_consecutive_full_partition_nans() -> None:
    """test that a lazy forward fill fails as expected on two consecutive partitions
    full of nans
    """
    t_df = df.copy()
    t_df["energy"][5:35] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    t_dask_df = forward_fill_lazy(t_dask_df, "energy", before="max")
    t_df = t_df.ffill()
    assert not t_df.equals(t_dask_df.compute())


def test_forward_fill_lazy_wrong_parameters() -> None:
    """test that a lazy forward fill fails as expected on wrong parameters"""
    t_df = df.copy()
    t_df["energy"][5:35] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    with pytest.raises(TypeError):
        t_dask_df = forward_fill_lazy(t_dask_df, "energy", before="wrong parameter")


def test_forward_fill_lazy_compute() -> None:
    """test that a lazy forward fill works as expected with compute=True"""
    t_df = df.copy()
    t_df["energy"][5:35] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    t_dask_df_comp = forward_fill_lazy(t_dask_df, "energy", before="max", compute_lengths=True)
    t_dask_df_nocomp = forward_fill_lazy(t_dask_df, "energy", before="max", compute_lengths=False)
    pd.testing.assert_frame_equal(t_dask_df_comp.compute(), t_dask_df_nocomp.compute())


def test_forward_fill_lazy_keep_head_nans() -> None:
    """test that a lazy forward fill works as expected with missing values at the
    beginning of the dataframe"""
    t_df = df.copy()
    t_df["energy"][:5] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    t_df = forward_fill_lazy(t_dask_df, "energy", before="max").compute()
    assert np.all(np.isnan(t_df["energy"][:5]))
    assert np.all(np.isfinite(t_df["energy"][5:]))


def test_forward_fill_lazy_no_channels() -> None:
    """test that a lazy forward fill raises an error when no channels are specified"""
    t_df = df.copy()
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    with pytest.raises(ValueError):
        t_dask_df = forward_fill_lazy(t_dask_df, [])


def test_forward_fill_lazy_wrong_channels() -> None:
    """test that a lazy forward fill raises an error when the specified channels do not exist"""
    t_df = df.copy()
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    with pytest.raises(KeyError):
        t_dask_df = forward_fill_lazy(t_dask_df, ["nonexistent_channel"])


def test_forward_fill_lazy_multiple_iterations() -> None:
    """test that a lazy forward fill works as expected with multiple iterations"""
    t_df = df.copy()
    t_df["energy"][5:35] = np.nan
    t_dask_df = ddf.from_pandas(t_df, npartitions=N_PARTITIONS)
    t_dask_df = forward_fill_lazy(t_dask_df, "energy", before="max", iterations=5)
    t_df = t_df.ffill()
    pd.testing.assert_frame_equal(t_df, t_dask_df.compute())


def test_backward_fill_lazy() -> None:
    """Test backward fill function"""
    t_df = pd.DataFrame(
        {
            "A": [1, 2, np.nan, np.nan, 5, np.nan],
            "B": [1, np.nan, 3, np.nan, 5, np.nan],
            "C": [np.nan, np.nan, np.nan, np.nan, np.nan, 6],
            "D": [1, 2, 3, 4, 5, 6],
        },
    )
    t_dask_df = ddf.from_pandas(t_df, npartitions=2)
    t_dask_df = backward_fill_lazy(t_dask_df, ["A", "B", "C"], after=2, iterations=2)
    t_df = t_df.bfill().bfill()
    pd.testing.assert_frame_equal(t_df, t_dask_df.compute())


def test_backward_fill_lazy_no_channels() -> None:
    """Test that an error is raised when no channels are specified"""
    t_df = pd.DataFrame(
        {
            "A": [1, 2, np.nan, np.nan, 5, np.nan],
            "B": [1, np.nan, 3, np.nan, 5, np.nan],
            "C": [np.nan, np.nan, np.nan, np.nan, np.nan, 6],
            "D": [1, 2, 3, 4, 5, 6],
        },
    )
    t_dask_df = ddf.from_pandas(t_df, npartitions=2)
    with pytest.raises(ValueError):
        t_dask_df = backward_fill_lazy(t_dask_df, [], after=2, iterations=2)


def test_backward_fill_lazy_wrong_channels() -> None:
    """Test that an error is raised when the specified channels do not exist"""
    t_df = pd.DataFrame(
        {
            "A": [1, 2, np.nan, np.nan, 5, np.nan],
            "B": [1, np.nan, 3, np.nan, 5, np.nan],
            "C": [np.nan, np.nan, np.nan, np.nan, np.nan, 6],
            "D": [1, 2, 3, 4, 5, 6],
        },
    )
    t_dask_df = ddf.from_pandas(t_df, npartitions=2)
    with pytest.raises(KeyError):
        t_dask_df = backward_fill_lazy(t_dask_df, ["nonexistent_channel"], after=2, iterations=2)


def test_backward_fill_lazy_wrong_after() -> None:
    """Test that an error is raised when the 'after' parameter is not an integer or 'max'"""
    t_df = pd.DataFrame(
        {
            "A": [1, 2, np.nan, np.nan, 5, np.nan],
            "B": [1, np.nan, 3, np.nan, 5, np.nan],
            "C": [np.nan, np.nan, np.nan, np.nan, np.nan, 6],
            "D": [1, 2, 3, 4, 5, 6],
        },
    )
    t_dask_df = ddf.from_pandas(t_df, npartitions=2)
    with pytest.raises(TypeError):
        t_dask_df = backward_fill_lazy(
            t_dask_df,
            ["A", "B", "C"],
            after="wrong_parameter",
            iterations=2,
        )


def test_backward_fill_lazy_multiple_iterations() -> None:
    """Test that the function works with multiple iterations"""
    t_df = pd.DataFrame(
        {
            "A": [1, 2, np.nan, np.nan, 5, np.nan],
            "B": [1, np.nan, 3, np.nan, 5, np.nan],
            "C": [np.nan, np.nan, np.nan, np.nan, np.nan, 6],
            "D": [1, 2, 3, 4, 5, 6],
        },
    )
    t_dask_df = ddf.from_pandas(t_df, npartitions=2)
    t_dask_df = backward_fill_lazy(t_dask_df, ["A", "B", "C"], after=2, iterations=2)
    t_df = t_df.bfill().bfill().bfill().bfill()
    pd.testing.assert_frame_equal(t_df, t_dask_df.compute())


def test_offset_by_other_columns_functionality() -> None:
    """test that the offset_by_other_columns function works as expected"""
    pd_df = pd.DataFrame(
        {
            "target": [10, 20, 30, 40, 50, 60],
            "off1": [1, 2, 3, 4, 5, 6],
            "off2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "off3": [9.75, 9.85, 9.95, 10.05, 10.15, 10.25],
        },
    )
    t_df = ddf.from_pandas(pd_df, npartitions=2)
    res = offset_by_other_columns(
        df=t_df.copy(),
        target_column="target",
        offset_columns=["off1"],
        weights=[1],
    )
    expected: list[Any] = [11, 22, 33, 44, 55, 66]
    np.testing.assert_allclose(res["target"].values, expected)

    res = offset_by_other_columns(
        df=t_df.copy(),
        target_column="target",
        offset_columns=["off1", "off2"],
        weights=[1, -1],
    )
    expected = [10.9, 21.8, 32.7, 43.6, 54.5, 65.4]
    np.testing.assert_allclose(res["target"].values, expected)

    res = offset_by_other_columns(
        df=t_df.copy(),
        target_column="target",
        offset_columns=["off3"],
        weights=[1],
        preserve_mean=True,
    )
    expected = [9.75, 19.85, 29.95, 40.05, 50.15, 60.25]
    np.testing.assert_allclose(res["target"].values, expected)

    res = offset_by_other_columns(
        df=t_df.copy(),
        target_column="target",
        offset_columns=["off3"],  # off3 has mean of 10
        weights=[1],
        reductions="mean",
    )
    expected = [20, 30, 40, 50, 60, 70]
    np.testing.assert_allclose(res["target"].values, expected)


def test_offset_by_other_columns_pandas_not_working() -> None:
    """test that the offset_by_other_columns function raises an error when
    used with pandas
    """
    pd_df = pd.DataFrame(
        {
            "target": [10, 20, 30, 40, 50, 60],
            "off1": [1, 2, 3, 4, 5, 6],
            "off2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "off3": [9.75, 9.85, 9.95, 10.05, 10.15, 10.25],
        },
    )
    with pytest.raises(NotImplementedError):
        _ = offset_by_other_columns(
            df=pd_df,
            target_column="target",
            offset_columns=["off1"],
            weights=[1],
        )


def test_offset_by_other_columns_rises() -> None:
    """Test that the offset_by_other_columns function raises an error when
    the specified columns do not exist
    """
    t_df = ddf.from_pandas(
        pd.DataFrame(
            {
                "target": [10, 20, 30, 40, 50, 60],
                "off1": [1, 2, 3, 4, 5, 6],
                "off2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "off3": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
            },
        ),
        npartitions=2,
    )
    pytest.raises(
        KeyError,
        offset_by_other_columns,
        df=t_df.copy(),
        target_column="nonexistent_column",
        offset_columns=["off1"],
        weights=[1],
    )
    pytest.raises(
        KeyError,
        offset_by_other_columns,
        df=t_df.copy(),
        target_column="target",
        offset_columns=["off1", "nonexistent_column"],
        weights=[1, 1],
    )
    pytest.raises(
        NotImplementedError,
        offset_by_other_columns,
        df=t_df.copy(),
        target_column="target",
        offset_columns=["off1"],
        weights=[1],
        reductions="not_mean",
    )
    pytest.raises(
        ValueError,
        offset_by_other_columns,
        df=t_df.copy(),
        target_column="target",
        offset_columns=["off1"],
        weights=[1, 1],
    )
    pytest.raises(
        TypeError,
        offset_by_other_columns,
        df=t_df.copy(),
        target_column="target",
        offset_columns=["off1"],
        weights=[1],
        preserve_mean="asd",
    )
