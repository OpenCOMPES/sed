import dask.array as dda
import dask.dataframe as ddf
import numpy as np
import pandas as pd
import pytest

from sed.dataframe.dataframe import DataFrame

colnames = ["energy", "dldPosX", "dldPosY", "other_parameter"]
alias_dict = {
    "ENERGY": "energy",
    "DETECTOR_POSITION_X": "dldPosX",
    "DETECTOR_POSITION_Y": "dldPosY",
}
name = "test_df"
nrows, ncols = 100, 4
chunksize = 50
arr_np = np.random.rand(nrows, ncols)
arr_dask = dda.from_array(arr_np, chunks=(chunksize, ncols), name=name)
df_pandas = pd.DataFrame(
    arr_np,
    columns=colnames,
)
df_dask_dataframe = ddf.from_pandas(df_pandas, chunksize=chunksize)


def test_dataframe_from_pandas():
    df = DataFrame()
    df.from_pandas(df_pandas)
    df.compute(inplace=True)
    assert np.allclose(df.values, arr_np)


def test_dataframe_from_pandas_lazy():
    df = DataFrame()
    df.from_pandas(df_pandas, chunksize)
    df.compute(inplace=True)
    assert np.allclose(df.values, arr_np)


def test_dataframe_from_dask_dataframe():
    df = DataFrame()
    df.from_dask_dataframe(df_dask_dataframe)
    df.compute(inplace=True)
    assert np.allclose(df.values, arr_np)


def test_dataframe_from_array():
    df = DataFrame()
    df.from_array(arr_np, columns=colnames)
    assert np.allclose(df.values, arr_np)


def test_dataframe_from_array_lazy():
    df = DataFrame()
    df.from_array(arr_np, columns=colnames, chunksize=chunksize)
    df.compute(inplace=True)
    assert np.allclose(df.values, arr_np)


def test_dataframe_from_dask_array():
    df = DataFrame()
    df.from_dask_array(arr_dask, columns=colnames)
    df.compute(inplace=True)
    assert np.allclose(df.values, arr_np)


def test_shapes():
    df = DataFrame(df_dask_dataframe)
    assert df.ncols == ncols
    assert len(df) == nrows


@pytest.mark.parametrize(
    "input",
    [arr_np, arr_dask, df_pandas, df_dask_dataframe],
    ids=lambda x: f"input type:{type(x)}",
)
def test_class_creation_input(input):
    df = DataFrame(input, columns=colnames, chunksize=chunksize)
    df.compute(inplace=True)
    assert df.ncols == ncols
    assert len(df) == nrows
    assert np.allclose(df.values, arr_np)


def test_column_by_alias():
    df = DataFrame(df_pandas, alias_dict=alias_dict)
    for k, v in alias_dict.items():
        pd.testing.assert_frame_equal(df[k]._df, df[v]._df)


def test_copy():
    df = DataFrame(df_pandas, alias_dict=alias_dict)
    dfc = df.copy()
    assert dfc == df


def test_deepcopy():
    df = DataFrame(df_pandas, alias_dict=alias_dict)
    dfc = df.deepcopy()
    assert dfc == df
