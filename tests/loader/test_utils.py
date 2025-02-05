"""Module tests.loader.test_utils, tests for the sed.load.utils file
"""
from __future__ import annotations

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


def test_split_channel_bitwise() -> None:
    """Test split_channel_bitwise function"""
    output_columns = ["b", "c"]
    bit_mask = 2
    expected_output = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5, 6, 7],
            "b": np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int8),
            "c": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        },
    )
    df = dd.from_pandas(test_df, npartitions=2)
    result = split_channel_bitwise(df, "a", output_columns, bit_mask)
    pd.testing.assert_frame_equal(result.compute(), expected_output)


def test_split_channel_bitwise_raises() -> None:
    """Test split_channel_bitwise function raises"""
    pytest.raises(
        KeyError,
        split_channel_bitwise,
        test_df,
        "wrong",
        ["b", "c"],
        3,
        False,
        [np.int8, np.int16],
    )
    pytest.raises(
        ValueError,
        split_channel_bitwise,
        test_df,
        "a",
        ["b", "c", "wrong"],
        3,
        False,
        [np.int8, np.int16],
    )
    pytest.raises(
        ValueError,
        split_channel_bitwise,
        test_df,
        "a",
        ["b", "c"],
        -1,
        False,
        [np.int8, np.int16],
    )
    pytest.raises(
        ValueError,
        split_channel_bitwise,
        test_df,
        "a",
        ["b", "c"],
        3,
        False,
        [np.int8, np.int16, np.int32],
    )
    pytest.raises(ValueError, split_channel_bitwise, test_df, "a", ["b", "c"], 3, False, [np.int8])
    pytest.raises(
        ValueError,
        split_channel_bitwise,
        test_df,
        "a",
        ["b", "c"],
        3,
        False,
        ["wrong", np.int16],
    )
    other_df = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5, 6, 7],
            "b": [0, 1, 2, 3, 4, 5, 6, 7],
        },
    )
    pytest.raises(KeyError, split_channel_bitwise, other_df, "a", ["b", "c"], 3, False, None)
