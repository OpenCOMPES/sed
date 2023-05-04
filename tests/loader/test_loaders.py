"""Test cases for loaders used to load dataframes"""
import os
from importlib.util import find_spec
from typing import List

import dask.dataframe as ddf
import pytest
from _pytest.mark.structures import ParameterSet

from sed.loader.base.loader import BaseLoader
from sed.loader.loader_interface import get_loader
from sed.loader.loader_interface import get_names_of_all_loaders
from sed.loader.utils import gather_files

package_dir = os.path.dirname(find_spec("sed").origin)

test_data_dir = os.path.join(package_dir, "..", "tests", "data")

read_types = ["folder", "files"]


def get_loader_name_from_loader_object(loader: BaseLoader) -> str:
    """Helper function to find the name of a loader given the loader object.

    Args:
        loader (BaseLoader): Loader object from which to get the name.

    Returns:
        str: extracted name.
    """
    for loader_name in get_names_of_all_loaders():
        gotten_loader = get_loader(loader_name)
        if loader.__name__ is gotten_loader.__name__:
            return loader_name
    return ""


def get_all_loaders() -> List[ParameterSet]:
    """Scans through the loader list and returns them for pytest parametrization"""
    loaders = []

    for loader in [
        get_loader(x) for x in get_names_of_all_loaders() if x != "flash"
    ]:
        loaders.append(pytest.param(loader))

    return loaders


@pytest.mark.parametrize("loader", get_all_loaders())
def test_if_loaders_are_children_of_base_loader(loader):
    """Test to verify that all loaders are children of BaseLoader"""
    if loader.__name__ != "BaseLoader":
        assert isinstance(loader, BaseLoader)


@pytest.mark.parametrize("loader", get_all_loaders())
def test_has_correct_read_dataframe_func(loader):
    """Test if all loaders have a valid read function implemented"""
    assert callable(loader.read_dataframe)
    if loader.__name__ != "BaseLoader":
        assert hasattr(loader, "files")
        assert hasattr(loader, "supported_file_types")

        loader_name = get_loader_name_from_loader_object(loader)

        for read_type in read_types:
            input_folder = os.path.join(test_data_dir, "loader", loader_name)
            for supported_file_type in loader.supported_file_types:
                input_files = gather_files(
                    folder=input_folder,
                    extension=supported_file_type,
                )
                if read_type == "folder":
                    loaded_dataframe, loaded_metadata = loader.read_dataframe(
                        folder=input_folder,
                        ftype=supported_file_type,
                        collect_metadata=False,
                    )
                else:
                    loaded_dataframe, loaded_metadata = loader.read_dataframe(
                        files=list(input_files),
                        ftype=supported_file_type,
                        collect_metadata=False,
                    )

                assert isinstance(loaded_dataframe, ddf.DataFrame)
                assert loaded_dataframe.npartitions == len(input_files)
                assert isinstance(loaded_metadata, dict)


def test_mpes_timestamps():
    """Function to test if the timestamps are loaded correctly"""
    loader_name = "mpes"
    loader = get_loader(loader_name)
    input_folder = os.path.join(test_data_dir, "loader", loader_name)
    df, _ = loader.read_dataframe(
        folder=input_folder,
        collect_metadata=False,
        time_stamps=True,
    )
    assert "timeStamps" in df.columns
