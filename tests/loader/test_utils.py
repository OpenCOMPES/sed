import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from sed.loader.utils import split_channel_bitwise

test_df = pd.DataFrame(
    {
        "a": [0, 1, 2, 3, 4, 5, 6, 7],
    },
)


@pytest.fixture
def input_df():
    return pd.DataFrame(
        {
            "input_column": [0, 1, 2, 3, 4, 5, 6, 7],
        },
    )


def test_split_channel_bitwise(input_df):
    output_columns = ["output1", "output2"]
    bit_mask = 2
    expected_output = pd.DataFrame(
        {
            "input_column": [0, 1, 2, 3, 4, 5, 6, 7],
            "output1": np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int8),
            "output2": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        },
    )
    df = dd.from_pandas(input_df, npartitions=2)
    result = split_channel_bitwise(df, "input_column", output_columns, bit_mask)
    pd.testing.assert_frame_equal(result.compute(), expected_output)


def test_split_channel_bitwise_raises():
    pytest.raises(
        KeyError,
        split_channel_bitwise,
        test_df,
        "wrong",
        ["b", "c"],
        3,
        [np.int8, np.int16],
    )
    pytest.raises(
        ValueError,
        split_channel_bitwise,
        test_df,
        "a",
        ["b", "c", "wrong"],
        3,
        [np.int8, np.int16],
    )
    pytest.raises(
        ValueError,
        split_channel_bitwise,
        test_df,
        "a",
        ["b", "c"],
        -1,
        [np.int8, np.int16],
    )
    pytest.raises(
        ValueError,
        split_channel_bitwise,
        test_df,
        "a",
        ["b", "c"],
        3,
        [np.int8, np.int16, np.int32],
    )
    pytest.raises(ValueError, split_channel_bitwise, test_df, "a", ["b", "c"], 3, [np.int8])
    pytest.raises(
        ValueError,
        split_channel_bitwise,
        test_df,
        "a",
        ["b", "c"],
        3,
        ["wrong", np.int16],
    )
    other_df = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5, 6, 7],
            "b": [0, 1, 2, 3, 4, 5, 6, 7],
        },
    )
    pytest.raises(KeyError, split_channel_bitwise, other_df, "a", ["b", "c"], 3, None)
