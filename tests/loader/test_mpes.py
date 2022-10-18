"""Module tests.loader.mpes, tests for the sed.load.mpes file
"""
import itertools
import os
from importlib.util import find_spec

import dask.dataframe as ddf
import pytest

from sed.loader.mpes import gather_files
from sed.loader.mpes import MpesLoader

# import numpy as np

package_dir = os.path.dirname(find_spec("sed").origin)
source_folder = package_dir + "/../tests/data/loader"

file_types = ["h5", "parquet", "csv", "json"]
read_types = ["folder", "files"]


@pytest.mark.parametrize(
    "file_type, read_type",
    itertools.product(file_types, read_types),
)
def test_mpes_loader(file_type: str, read_type: str):
    """Function to test if the mpes loader returns correct dataframes

    Args:
        file_type (str): File type to test. "h5", "parquet", "csv" or "json"
        read_type (str): read method to use. "files" or "folder".
    """
    ml = MpesLoader()  # pylint: disable=invalid-name
    if read_type == "folder":
        df = ml.read_dataframe(folder=source_folder, ftype=file_type)
    else:
        files = gather_files(folder=source_folder, extension=file_type)
        assert len(files) == 2
        df = ml.read_dataframe(files=files, ftype=file_type)
    assert isinstance(df, ddf.DataFrame)
    assert df.npartitions == 2