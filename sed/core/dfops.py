"""This module contains dataframe operations functions for the sed package

"""
# Note: some of the functions presented here were
# inspired by https://github.com/mpes-kit/mpes
from typing import Callable
from typing import Sequence
from typing import Union

import dask.dataframe
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar


def apply_jitter(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    cols: Union[str, Sequence[str]],
    cols_jittered: Union[str, Sequence[str]] = None,
    amps: Union[float, Sequence[float]] = 0.5,
    jitter_type: str = "uniform",
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """Add jittering to one or more dataframe columns.

    Args:
        df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to add
            noise/jittering to.
        cols (Union[str, Sequence[str]]): Names of the columns to add jittering to.
        cols_jittered (Union[str, Sequence[str]], optional): Names of the columns
            with added jitter. Defaults to None.
        amps (Union[float, Sequence[float]], optional): Amplitude scalings for the
            jittering noise. If one number is given, the same is used for all axes.
            For normal noise, the added noise will have sdev [-amp, +amp], for
            uniform noise it will cover the interval [-amp, +amp].
            Defaults to 0.5.
        jitter_type (str, optional): the type of jitter to add. 'uniform' or 'normal'
            distributed noise. Defaults to "uniform".

    Returns:
        Union[pd.DataFrame, dask.dataframe.DataFrame]: dataframe with added columns.
    """
    assert cols is not None, "cols needs to be provided!"
    assert jitter_type in (
        "uniform",
        "normal",
    ), "type needs to be one of 'normal', 'uniform'!"

    if isinstance(cols, str):
        cols = [cols]
    if isinstance(cols_jittered, str):
        cols_jittered = [cols_jittered]
    if cols_jittered is None:
        cols_jittered = [col + "_jittered" for col in cols]
    if isinstance(amps, float):
        amps = list(np.ones(len(cols)) * amps)

    colsize = df[cols[0]].size

    if jitter_type == "uniform":
        # Uniform Jitter distribution
        jitter = np.random.uniform(low=-1, high=1, size=colsize)
    elif jitter_type == "normal":
        # Normal Jitter distribution works better for non-linear transformations and
        # jitter sizes that don't match the original bin sizes
        jitter = np.random.standard_normal(size=colsize)

    for col, col_jittered, amp in zip(cols, cols_jittered, amps):
        df[col_jittered] = df[col] + amp * jitter

    return df


def drop_column(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    column_name: Union[str, Sequence[str]],
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """Delete columns.

    Args:
        df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to use.
        column_name (Union[str, Sequence[str]])): List of column names to be dropped.

    Returns:
        Union[pd.DataFrame, dask.dataframe.DataFrame]: Dataframe with dropped columns.
    """
    out_df = df.drop(column_name, axis=1)

    return out_df


def apply_filter(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    col: str,
    lower_bound: float = -np.inf,
    upper_bound: float = np.inf,
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """Application of bound filters to a specified column (can be used consecutively).

    Args:
        df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to use.
        col (str): Name of the column to filter. Passing "index" for col will
            filter on the index in each dataframe partition.
        lower_bound (float, optional): The lower bound used in the filtering.
            Defaults to -np.inf.
        upper_bound (float, optional): The lower bound used in the filtering.
            Defaults to np.inf.

    Returns:
        Union[pd.DataFrame, dask.dataframe.DataFrame]: The filtered dataframe.
    """
    df = df.copy()
    if col == "index":
        df["index"] = df.index

    out_df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]

    if col == "index":
        out_df = drop_column(out_df, "index")

    return out_df


def add_time_stamped_data(
    df: dask.dataframe.DataFrame,
    time_stamps: np.ndarray,
    data: np.ndarray,
    dest_column: str,
    time_stamp_column: str,
    **kwds,
) -> dask.dataframe.DataFrame:
    """Add data in form of timestamp/value pairs to the dataframe using interpolation to the
    timestamps in the dataframe.

    Args:
        df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to use.
        time_stamps (np.ndarray): Time stamps of the values to add
        data (np.ndarray): Values corresponding at the time stamps in time_stamps
        dest_column (str): destination column name
        time_stamp_column (str): Time stamp column name

    Returns:
        Union[pd.DataFrame, dask.dataframe.DataFrame]: Dataframe with added column
    """
    if time_stamp_column not in df.columns:
        raise ValueError(f"{time_stamp_column} not found in dataframe!")

    if len(time_stamps) != len(data):
        raise ValueError("time_stamps and data have to be of same length!")

    def interpolate_timestamps(
        df: dask.dataframe.DataFrame,
    ) -> dask.dataframe.DataFrame:
        df_timestamps = df[time_stamp_column]
        df[dest_column] = np.interp(df_timestamps, time_stamps, data)
        return df

    if not isinstance(df, dask.dataframe.DataFrame):
        raise ValueError("This function only works for Dask Dataframes!")

    df = df.map_partitions(interpolate_timestamps, **kwds)

    return df


def map_columns_2d(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    map_2d: Callable,
    x_column: str,
    y_column: str,
    **kwds,
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """Apply a 2-dimensional mapping simultaneously to two dimensions.

    Args:
        df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to use.
        map_2d (Callable): 2D mapping function.
        x_column (str): The X column of the dataframe to apply mapping to.
        y_column (str): The Y column of the dataframe to apply mapping to.
        **kwds: Additional arguments for the 2D mapping function.

    Returns:
        Union[pd.DataFrame, dask.dataframe.DataFrame]: Dataframe with mapped columns.
    """
    new_x_column = kwds.pop("new_x_column", x_column)
    new_y_column = kwds.pop("new_y_column", y_column)

    (df[new_x_column], df[new_y_column]) = map_2d(
        df[x_column],
        df[y_column],
        **kwds,
    )

    return df


def forward_fill_lazy(
    df: dask.dataframe.DataFrame,
    columns: Sequence[str] = None,
    before: Union[str, int] = "max",
    compute_lengths: bool = False,
    iterations: int = 2,
) -> dask.dataframe.DataFrame:
    """Forward fill the specified columns multiple times in a dask dataframe.

    Allows forward filling between partitions. This is useful for dataframes
    that have sparse data, such as those with many NaNs.
    Runnin the forward filling multiple times can fix the issue of having
    entire partitions consisting of NaNs. By default we run this twice, which
    is enough to fix the issue for dataframes with no consecutive partitions of NaNs.

    Args:
        df (dask.dataframe.DataFrame): The dataframe to forward fill.
        columns (list): The columns to forward fill. If None, fills all columns
        before (int, str, optional): The number of rows to include before the current partition.
            if 'max' it takes as much as possible from the previous partition, which is
            the size of the smallest partition in the dataframe. Defaults to 'max'.
        compute_lengths (bool, optional): Whether to compute the length of each partition
        iterations (int, optional): The number of times to forward fill the dataframe.

    Returns:
        dask.dataframe.DataFrame: The dataframe with the specified columns forward filled.
    """
    if columns is None:
        columns = df.columns
    elif isinstance(columns, str):
        columns = [columns]
    elif len(columns) == 0:
        raise ValueError("columns must be a non-empty list of strings!")
    for c in columns:
        if c not in df.columns:
            raise KeyError(f"{c} not in dataframe!")

    # Define a custom function to forward fill specified columns
    def forward_fill_partition(df):
        df[columns] = df[columns].ffill()
        return df

    # calculate the number of rows in each partition and choose least
    if before == "max":
        nrows = df.map_partitions(len)
        if compute_lengths:
            with ProgressBar():
                print("Computing dataframe shape...")
                nrows = nrows.compute()
        before = min(nrows)
    elif not isinstance(before, int):
        raise TypeError('before must be an integer or "max"')
    # Use map_overlap to apply forward_fill_partition
    for _ in range(iterations):
        df = df.map_overlap(
            forward_fill_partition,
            before=before,
            after=0,
        )
    return df


def backward_fill_lazy(
    df: dask.dataframe.DataFrame,
    columns: Sequence[str] = None,
    after: Union[str, int] = "max",
    compute_lengths: bool = False,
    iterations: int = 1,
) -> dask.dataframe.DataFrame:
    """Forward fill the specified columns multiple times in a dask dataframe.

    Allows backward filling between partitions. Similar to forward fill, but backwards.
    This helps to fill the initial values of a dataframe, which are often NaNs.
    Use with care as the assumption of the values being the same in the past is often not true.

    Args:
        df (dask.dataframe.DataFrame): The dataframe to forward fill.
        columns (list): The columns to forward fill. If None, fills all columns
        after (int, str, optional): The number of rows to include after the current partition.
            if 'max' it takes as much as possible from the previous partition, which is
            the size of the smallest partition in the dataframe. Defaults to 'max'.
        compute_lengths (bool, optional): Whether to compute the length of each partition
        iterations (int, optional): The number of times to backward fill the dataframe.

    Returns:
        dask.dataframe.DataFrame: The dataframe with the specified columns backward filled.
    """
    if columns is None:
        columns = df.columns
    elif isinstance(columns, str):
        columns = [columns]
    elif len(columns) == 0:
        raise ValueError("columns must be a non-empty list of strings!")
    for c in columns:
        if c not in df.columns:
            raise KeyError(f"{c} not in dataframe!")

    # Define a custom function to forward fill specified columns
    def backward_fill_partition(df):
        df[columns] = df[columns].bfill()
        return df

    # calculate the number of rows in each partition and choose least
    if after == "max":
        nrows = df.map_partitions(len)
        if compute_lengths:
            with ProgressBar():
                print("Computing dataframe shape...")
                nrows = nrows.compute()
        after = min(nrows)
    elif not isinstance(after, int):
        raise TypeError('before must be an integer or "max"')
    # Use map_overlap to apply forward_fill_partition
    for _ in range(iterations):
        df = df.map_overlap(
            backward_fill_partition,
            before=0,
            after=after,
        )
    return df


def offset_by_other_columns(
    df: dask.dataframe.DataFrame,
    target_column: str,
    offset_columns: Union[str, Sequence[str]],
    weights: Union[float, Sequence[float]],
    reductions: Union[str, Sequence[str]] = None,
    preserve_mean: Union[bool, Sequence[bool]] = False,
    inplace: bool = True,
    rename: str = None,
) -> dask.dataframe.DataFrame:
    """Apply an offset to a column based on the values of other columns.

    Args:
        df (dask.dataframe.DataFrame): Dataframe to use. Currently supports only dask dataframes.
        target_column (str): Name of the column to apply the offset to.
        offset_columns (str): Name of the column(s) to use for the offset.
        weights (flot): weights to apply on each column before adding. Used also for changing sign.
        reductions (str, optional): Reduction function to use for the offset. Defaults to "mean".
            Currently, only mean is supported.
        preserve_mean (bool, optional): Whether to subtract the mean of the offset column.
            Defaults to False. If a list is given, it must have the same length as
            offset_columns. Otherwise the value passed is used for all columns.
        inplace (bool, optional): Whether to apply the offset inplace.
            If false, the new column will have the name provided by rename, or has the same name as
            target_column with the suffix _offset if that is None. Defaults to True.
        rename (str, optional): Name of the new column if inplace is False. Defaults to None.
    Returns:
        dask.dataframe.DataFrame: Dataframe with the new column.
    """
    if target_column not in df.columns:
        raise KeyError(f"{target_column} not in dataframe!")

    if isinstance(offset_columns, str):
        offset_columns = [offset_columns]
    elif not isinstance(offset_columns, Sequence):
        raise TypeError(f"Invalid type for columns: {type(offset_columns)}")
    if any(c not in df.columns for c in offset_columns):
        raise KeyError(f"{offset_columns} not in dataframe!")

    if isinstance(weights, (int, float, np.floating, np.integer)):
        weights = [weights]
    elif not isinstance(weights, Sequence):
        raise TypeError(f"Invalid type for signs: {type(weights)}")
    if len(weights) != len(offset_columns):
        raise ValueError("signs and offset_columns must have the same length!")
    signs_dict = dict(zip(offset_columns, weights))

    if isinstance(reductions, str) or reductions is None:
        reductions = [reductions] * len(offset_columns)
    elif not isinstance(reductions, Sequence):
        raise ValueError(f"reductions must be a string or list of strings! not {type(reductions)}")
    if any(r not in ["mean", None] for r in reductions):
        raise NotImplementedError("Only reductions currently supported is 'mean'!")

    if isinstance(preserve_mean, bool):
        preserve_mean = [preserve_mean] * len(offset_columns)
    elif not isinstance(preserve_mean, Sequence):
        raise TypeError(f"Invalid type for preserve_mean: {type(preserve_mean)}")
    elif any(not isinstance(p, bool) for p in preserve_mean):
        raise TypeError(f"Invalid type for preserve_mean: {type(preserve_mean)}")
    if len(preserve_mean) != len(offset_columns):
        raise ValueError("preserve_mean and offset_columns must have the same length!")

    if not inplace:
        if rename is None:
            rename = target_column + "_offset"
        df[rename] = df[target_column]
        target_column = rename

    if isinstance(df, pd.DataFrame):
        raise NotImplementedError(
            "Offsetting by other columns is currently not supported for pandas dataframes! "
            "Please open a request on GitHub if this feature is required.",
        )

    # calculate the mean of the columns to reduce
    means = {
        col: dask.delayed(df[col].mean())
        for col, red, pm in zip(offset_columns, reductions, preserve_mean)
        if red or pm
    }

    # define the functions to apply the offsets
    def shift_by_mean(x, cols, signs, means, flip_signs=False):
        """Shift the target column by the mean of the offset columns."""
        for col in cols:
            s = -signs[col] if flip_signs else signs[col]
            x[target_column] = x[target_column] + s * means[col]
        return x[target_column]

    def shift_by_row(x, cols, signs):
        """Apply the offsets to the target column."""
        for col in cols:
            x[target_column] = x[target_column] + signs[col] * x[col]
        return x[target_column]

    # apply offset from the reduced columns
    df[target_column] = df.map_partitions(
        shift_by_mean,
        cols=[col for col, red in zip(offset_columns, reductions) if red],
        signs=signs_dict,
        means=means,
        meta=df[target_column].dtype,
    )

    # apply offset from the offset columns
    df[target_column] = df.map_partitions(
        shift_by_row,
        cols=[col for col, red in zip(offset_columns, reductions) if not red],
        signs=signs_dict,
        meta=df[target_column].dtype,
    )

    # compensate shift from the preserved mean columns
    if any(preserve_mean):
        df[target_column] = df.map_partitions(
            shift_by_mean,
            cols=[col for col, pmean in zip(offset_columns, preserve_mean) if pmean],
            signs=signs_dict,
            means=means,
            flip_signs=True,
            meta=df[target_column].dtype,
        )

    return df
