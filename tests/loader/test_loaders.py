"""Test cases for loaders used to load dataframes
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import cast

import dask.dataframe as ddf
import pytest
from _pytest.mark.structures import ParameterSet

from sed.core.config import parse_config
from sed.loader.base.loader import BaseLoader
from sed.loader.flash.loader import FlashLoader
from sed.loader.loader_interface import get_loader
from sed.loader.loader_interface import get_names_of_all_loaders
from sed.loader.utils import gather_files

test_dir = os.path.join(os.path.dirname(__file__), "..")
test_data_dir = os.path.join(test_dir, "data")

read_types = ["one_file", "files", "one_folder", "folders", "one_run", "runs"]
runs = {"generic": None, "mpes": ["30", "50"], "flash": ["43878", "43878"], "sxp": ["0016", "0016"]}


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
                folder_config={},
                user_config={},
                system_config={},
            ),
        )
        if loader.__name__ is gotten_loader.__name__:
            return loader_name
    return ""


def get_all_loaders() -> list[ParameterSet]:
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
                folder_config={},
                user_config={},
                system_config={},
            ),
        )
        for loader_name in get_names_of_all_loaders()
    ]:
        loaders.append(pytest.param(loader))

    return loaders


@pytest.mark.parametrize("loader", get_all_loaders())
def test_if_loaders_are_children_of_base_loader(loader: BaseLoader) -> None:
    """Test to verify that all loaders are children of BaseLoader"""
    if loader.__name__ != "BaseLoader":
        assert isinstance(loader, BaseLoader)


@pytest.mark.parametrize("loader", get_all_loaders())
@pytest.mark.parametrize("read_type", read_types)
def test_has_correct_read_dataframe_func(loader: BaseLoader, read_type: str) -> None:
    """Test if all loaders have a valid read function implemented"""
    assert callable(loader.read_dataframe)

    # Fix for race condition during parallel testing
    if loader.__name__ in {"flash", "sxp"}:
        config = deepcopy(loader._config)  # pylint: disable=protected-access
        config["core"]["paths"]["processed"] = Path(
            config["core"]["paths"]["processed"],
            f"_{read_type}",
        )
        loader = get_loader(loader_name=loader.__name__, config=config)

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
                loaded_dataframe, _, loaded_metadata = loader.read_dataframe(
                    files=input_files[0],
                    ftype=supported_file_type,
                    collect_metadata=False,
                )
                expected_size = 1
            elif read_type == "files":
                loaded_dataframe, _, loaded_metadata = loader.read_dataframe(
                    files=list(input_files),
                    ftype=supported_file_type,
                    collect_metadata=False,
                )
                expected_size = len(input_files)
            elif read_type == "one_folder":
                loaded_dataframe, _, loaded_metadata = loader.read_dataframe(
                    folders=input_folder,
                    ftype=supported_file_type,
                    collect_metadata=False,
                )
                expected_size = len(input_files)
            elif read_type == "folders":
                loaded_dataframe, _, loaded_metadata = loader.read_dataframe(
                    folders=[input_folder],
                    ftype=supported_file_type,
                    collect_metadata=False,
                )
                expected_size = len(input_files)
            elif read_type == "one_run":
                if runs[get_loader_name_from_loader_object(loader)] is None:
                    pytest.skip("Not implemented")
                loaded_dataframe, _, loaded_metadata = loader.read_dataframe(
                    runs=runs[get_loader_name_from_loader_object(loader)][0],
                    ftype=supported_file_type,
                    collect_metadata=False,
                )
                expected_size = 1
            elif read_type == "runs":
                if runs[get_loader_name_from_loader_object(loader)] is None:
                    pytest.skip("Not implemented")
                loaded_dataframe, _, loaded_metadata = loader.read_dataframe(
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

    if loader.__name__ in {"flash", "sxp"}:
        loader = cast(FlashLoader, loader)
        loader._initialize_dirs()
        for file in os.listdir(Path(loader.processed_dir, "buffer")):
            os.remove(Path(loader.processed_dir, "buffer", file))


@pytest.mark.parametrize("loader", get_all_loaders())
def test_timed_dataframe(loader: BaseLoader) -> None:
    """Test if the loaders return a correct timed dataframe

    Args:
        loader (BaseLoader): the loader object to test
    """

    # Fix for race condition during parallel testing
    if loader.__name__ in {"flash", "sxp"}:
        config = deepcopy(loader._config)  # pylint: disable=protected-access
        config["core"]["paths"]["processed"] = Path(
            config["core"]["paths"]["processed"],
            "_timed_dataframe",
        )
        loader = get_loader(loader_name=loader.__name__, config=config)

    if loader.__name__ != "BaseLoader":
        loader_name = get_loader_name_from_loader_object(loader)
        input_folder = os.path.join(test_data_dir, "loader", loader_name)
        for supported_file_type in loader.supported_file_types:
            loaded_dataframe, loaded_timed_dataframe, _ = loader.read_dataframe(
                folders=input_folder,
                ftype=supported_file_type,
                collect_metadata=False,
            )
            if loaded_timed_dataframe is None:
                if loader.__name__ in {"flash", "sxp"}:
                    loader = cast(FlashLoader, loader)
                    loader._initialize_dirs()
                    for file in os.listdir(Path(loader.processed_dir, "buffer")):
                        os.remove(Path(loader.processed_dir, "buffer", file))
                pytest.skip("Not implemented")
            assert isinstance(loaded_timed_dataframe, ddf.DataFrame)
            assert set(loaded_timed_dataframe.columns).issubset(set(loaded_dataframe.columns))
            assert loaded_timed_dataframe.npartitions == loaded_dataframe.npartitions

    if loader.__name__ in {"flash", "sxp"}:
        loader = cast(FlashLoader, loader)
        loader._initialize_dirs()
        for file in os.listdir(Path(loader.processed_dir, "buffer")):
            os.remove(Path(loader.processed_dir, "buffer", file))


@pytest.mark.parametrize("loader", get_all_loaders())
def test_get_count_rate(loader: BaseLoader) -> None:
    """Test the get_count_rate function

    Args:
        loader (BaseLoader): the loader object to test
    """

    # Fix for race condition during parallel testing
    if loader.__name__ in {"flash", "sxp"}:
        config = deepcopy(loader._config)  # pylint: disable=protected-access
        config["core"]["paths"]["processed"] = Path(
            config["core"]["paths"]["processed"],
            "_count_rate",
        )
        loader = get_loader(loader_name=loader.__name__, config=config)

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
                if loader.__name__ in {"flash", "sxp"}:
                    loader = cast(FlashLoader, loader)
                    loader._initialize_dirs()
                    for file in os.listdir(Path(loader.processed_dir, "buffer")):
                        os.remove(Path(loader.processed_dir, "buffer", file))
                pytest.skip("Not implemented")
            assert len(loaded_time) == len(loaded_countrate)
            loaded_time2, loaded_countrate2 = loader.get_count_rate(fids=[0])
            assert len(loaded_time2) == len(loaded_countrate2)
            assert len(loaded_time2) < len(loaded_time)

            # illegal keywords
            with pytest.raises(TypeError):
                loader.get_count_rate(illegal_kwd=True)

    if loader.__name__ in {"flash", "sxp"}:
        loader = cast(FlashLoader, loader)
        loader._initialize_dirs()
        for file in os.listdir(Path(loader.processed_dir, "buffer")):
            os.remove(Path(loader.processed_dir, "buffer", file))


@pytest.mark.parametrize("loader", get_all_loaders())
def test_get_elapsed_time(loader: BaseLoader) -> None:
    """Test the get_elapsed_time function

    Args:
        loader (BaseLoader): the loader object to test
    """

    # Fix for race condition during parallel testing
    if loader.__name__ in {"flash", "sxp"}:
        config = deepcopy(loader._config)  # pylint: disable=protected-access
        config["core"]["paths"]["processed"] = Path(
            config["core"]["paths"]["processed"],
            "_elapsed_time",
        )
        loader = get_loader(loader_name=loader.__name__, config=config)

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
                if loader.__name__ in {"sxp"}:
                    loader = cast(FlashLoader, loader)
                    loader._initialize_dirs()
                    for file in os.listdir(Path(loader.processed_dir, "buffer")):
                        os.remove(Path(loader.processed_dir, "buffer", file))
                pytest.skip("Not implemented")
            assert elapsed_time > 0
            elapsed_time2 = loader.get_elapsed_time(fids=[0])
            assert elapsed_time2 > 0
            assert elapsed_time > elapsed_time2

            # illegal keywords
            with pytest.raises(TypeError):
                loader.get_elapsed_time(illegal_kwd=True)

    if loader.__name__ in {"flash", "sxp"}:
        loader = cast(FlashLoader, loader)
        loader._initialize_dirs()
        for file in os.listdir(Path(loader.processed_dir, "buffer")):
            os.remove(Path(loader.processed_dir, "buffer", file))


def test_mpes_timestamps() -> None:
    """Function to test if the timestamps are loaded correctly"""
    loader_name = "mpes"
    loader = get_loader(loader_name)
    input_folder = os.path.join(test_data_dir, "loader", loader_name)
    df, _, _ = loader.read_dataframe(
        folders=input_folder,
        collect_metadata=False,
        time_stamps=True,
    )
    assert "timeStamps" in df.columns
