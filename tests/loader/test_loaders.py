"""Test cases for loaders used to load dataframes
"""
import os
from importlib.util import find_spec
from pathlib import Path
from typing import cast
from typing import List

import dask.dataframe as ddf
import pytest
from _pytest.mark.structures import ParameterSet

from sed.core.config import parse_config
from sed.loader.base.loader import BaseLoader
from sed.loader.flash.loader import FlashLoader
from sed.loader.loader_interface import get_loader
from sed.loader.loader_interface import get_names_of_all_loaders
from sed.loader.utils import gather_files

package_dir = os.path.dirname(find_spec("sed").origin)

test_data_dir = os.path.join(package_dir, "..", "tests", "data")

read_types = ["one_file", "files", "one_folder", "folders", "one_run", "runs"]
runs = {"generic": None, "mpes": None, "flash": ["43878", "43878"]}


def get_loader_name_from_loader_object(loader: BaseLoader) -> str:
    """Helper function to find the name of a loader given the loader object.

    Args:
        loader (BaseLoader): Loader object from which to get the name.

    Returns:
        str: extracted name.
    """
    for loader_name in get_names_of_all_loaders():
        gotten_loader = get_loader(
            loader_name,
            config=parse_config(
                os.path.join(
                    test_data_dir,
                    "loader",
                    loader_name,
                    "config.yaml",
                ),
            ),
        )
        if loader.__name__ is gotten_loader.__name__:
            return loader_name
    return ""


def get_all_loaders() -> List[ParameterSet]:
    """Scans through the loader list and returns them for pytest parametrization"""
    loaders = []

    for loader in [
        get_loader(
            loader_name,
            config=parse_config(
                os.path.join(
                    test_data_dir,
                    "loader",
                    loader_name,
                    "config.yaml",
                ),
            ),
        )
        for loader_name in get_names_of_all_loaders()
    ]:
        loaders.append(pytest.param(loader))

    return loaders


@pytest.mark.parametrize("loader", get_all_loaders())
def test_if_loaders_are_children_of_base_loader(loader: BaseLoader):
    """Test to verify that all loaders are children of BaseLoader"""
    if loader.__name__ != "BaseLoader":
        assert isinstance(loader, BaseLoader)


@pytest.mark.parametrize("loader", get_all_loaders())
@pytest.mark.parametrize("read_type", read_types)
def test_has_correct_read_dataframe_func(loader: BaseLoader, read_type: str):
    """Test if all loaders have a valid read function implemented"""
    assert callable(loader.read_dataframe)
    if loader.__name__ != "BaseLoader":
        assert hasattr(loader, "files")
        assert hasattr(loader, "supported_file_types")

        loader_name = get_loader_name_from_loader_object(loader)

        input_folder = os.path.join(test_data_dir, "loader", loader_name)
        for supported_file_type in loader.supported_file_types:
            input_files = gather_files(
                folder=input_folder,
                extension=supported_file_type,
            )
            if read_type == "one_file":
                loaded_dataframe, loaded_metadata = loader.read_dataframe(
                    files=input_files[0],
                    ftype=supported_file_type,
                    collect_metadata=False,
                )
                expected_size = 1
            elif read_type == "files":
                loaded_dataframe, loaded_metadata = loader.read_dataframe(
                    files=list(input_files),
                    ftype=supported_file_type,
                    collect_metadata=False,
                )
                expected_size = len(input_files)
            elif read_type == "one_folder":
                loaded_dataframe, loaded_metadata = loader.read_dataframe(
                    folders=input_folder,
                    ftype=supported_file_type,
                    collect_metadata=False,
                )
                expected_size = len(input_files)
            elif read_type == "folders":
                loaded_dataframe, loaded_metadata = loader.read_dataframe(
                    folders=[input_folder],
                    ftype=supported_file_type,
                    collect_metadata=False,
                )
                expected_size = len(input_files)
            elif read_type == "one_run":
                if runs[get_loader_name_from_loader_object(loader)] is None:
                    pytest.skip("Not implemented")
                loaded_dataframe, loaded_metadata = loader.read_dataframe(
                    runs=runs[get_loader_name_from_loader_object(loader)][0],
                    ftype=supported_file_type,
                    collect_metadata=False,
                )
                expected_size = 1
            elif read_type == "runs":
                if runs[get_loader_name_from_loader_object(loader)] is None:
                    pytest.skip("Not implemented")
                loaded_dataframe, loaded_metadata = loader.read_dataframe(
                    runs=runs[get_loader_name_from_loader_object(loader)],
                    ftype=supported_file_type,
                    collect_metadata=False,
                )
                expected_size = len(
                    runs[get_loader_name_from_loader_object(loader)],
                )

            assert isinstance(loaded_dataframe, ddf.DataFrame)
            assert loaded_dataframe.npartitions == expected_size
            assert isinstance(loaded_metadata, dict)

    if loader.__name__ == "flash":
        loader = cast(FlashLoader, loader)
        _, parquet_data_dir = loader.initialize_paths()
        for file in os.listdir(Path(parquet_data_dir, "per_file")):
            os.remove(Path(parquet_data_dir, "per_file", file))


@pytest.mark.parametrize("loader", get_all_loaders())
def test_get_count_rate(loader: BaseLoader):
    """Test the get_count_rate function

    Args:
        loader (BaseLoader): the loader object to test
    """
    if loader.__name__ != "BaseLoader":
        loader_name = get_loader_name_from_loader_object(loader)
        input_folder = os.path.join(test_data_dir, "loader", loader_name)
        for supported_file_type in loader.supported_file_types:
            loader.read_dataframe(
                folders=input_folder,
                ftype=supported_file_type,
                collect_metadata=False,
            )
            loaded_time, loaded_countrate = loader.get_count_rate()
            if loaded_time is None and loaded_countrate is None:
                pytest.skip("Not implemented")
            assert len(loaded_time) == len(loaded_countrate)
            loaded_time2, loaded_countrate2 = loader.get_count_rate(fids=[0])
            assert len(loaded_time2) == len(loaded_countrate2)
            assert len(loaded_time2) < len(loaded_time)


@pytest.mark.parametrize("loader", get_all_loaders())
def test_get_elapsed_time(loader: BaseLoader):
    """Test the get_elapsed_time function

    Args:
        loader (BaseLoader): the loader object to test
    """
    if loader.__name__ != "BaseLoader":
        loader_name = get_loader_name_from_loader_object(loader)
        input_folder = os.path.join(test_data_dir, "loader", loader_name)
        for supported_file_type in loader.supported_file_types:
            loader.read_dataframe(
                folders=input_folder,
                ftype=supported_file_type,
                collect_metadata=False,
            )
            elapsed_time = loader.get_elapsed_time()
            if elapsed_time is None:
                pytest.skip("Not implemented")
            assert elapsed_time > 0
            elapsed_time2 = loader.get_elapsed_time(fids=[0])
            assert elapsed_time2 > 0
            assert elapsed_time > elapsed_time2


def test_mpes_timestamps():
    """Function to test if the timestamps are loaded correctly"""
    loader_name = "mpes"
    loader = get_loader(loader_name)
    input_folder = os.path.join(test_data_dir, "loader", loader_name)
    df, _ = loader.read_dataframe(
        folders=input_folder,
        collect_metadata=False,
        time_stamps=True,
    )
    assert "timeStamps" in df.columns
