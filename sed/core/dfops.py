# Note: some of the functions presented here were
# inspired by https://github.com/mpes-kit/mpes
from typing import Sequence
from typing import Union

import dask.dataframe
import numpy as np
import pandas as pd


def apply_jitter(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    cols: Union[str, Sequence[str]],
    cols_jittered: Union[str, Sequence[str]] = None,
    amps: Union[float, Sequence[float]] = 0.5,
    type: str = "uniform",
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """Add jittering to one or more dataframe columns.

    :Parameters:
        df : dataframe
            Dataframe to add noise/jittering to.
        amps : numer or list of numbers
            Amplitude scalings for the jittering noise. If one number is given,
            the same is used for all axes
        cols : str list
            Names of the columns to add jittering to.
        cols_jittered : str list
            Names of the columns with added jitter.
        type: the type of jitter to add. 'uniform' or 'normal' distributed noise.
    :Return:
        dataframe with added columns
    """
    assert cols is not None, "cols needs to be provided!"
    assert type in (
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
        amps = np.ones(len(cols)) * amps

    colsize = df[cols[0]].size

    if type == "uniform":
        # Uniform Jitter distribution
        jitter = np.random.uniform(low=-1, high=1, size=colsize)
    elif type == "normal":
        # Normal Jitter distribution works better for non-linear transformations and
        # jitter sizes that don't match the original bin sizes
        jitter = np.random.standard_normal(size=colsize)

    for (col, col_jittered, amp) in zip(cols, cols_jittered, amps):
        df[col_jittered] = df[col] + amp * jitter

    return df


def drop_column(df, column_name):
    """Delete columns.

    **Parameters**\n
    colnames: str/list/tuple
        List of column names to be dropped.
    """

    out_df = df.drop(column_name, axis=1)

    return out_df


def apply_filter(df, col, lower_bound=-np.inf, upper_bound=np.inf):
    """Application of bound filters to a specified column (can be used consecutively).

    **Parameters**\n
    colname: str
        Name of the column to filter.
    lb, ub: numeric, numeric | -infinity, infinity
        The lower and upper bounds used in the filtering.
    update: str | 'replace'
        Update option for the filtered dataframe.
    ret: bool | False
        Return option for the filtered dataframe.
    """
    out_df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]

    return out_df


def map_columns_2d(df, map2D, x_column, y_column, **kwds):
    """Apply a 2-dimensional mapping simultaneously to two dimensions.

    **Parameters**\n
    map2D: function
        2D mapping function.
    X, Y: series, series
        The two columns of the dataframe to apply mapping to.
    **kwds: keyword arguments
        Additional arguments for the 2D mapping function.
    """

    newX = kwds.pop("newX", x_column)
    newY = kwds.pop("newY", y_column)

    (df[newX], df[newY]) = map2D(df[x_column], df[y_column], **kwds)

    return df
