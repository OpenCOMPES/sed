""" This module contains fixtures for the CFEL module tests.
"""
import os
import shutil
from pathlib import Path

import h5py
import pytest

from sed.core.config import parse_config

test_dir = os.path.join(os.path.dirname(__file__), "../..")
# Use CFEL config instead of FLASH config
config_path = os.path.join(test_dir, "data/loader/cfel/config2.yaml")
# Use CFEL test data paths
H5_PATH = "20250411_12h34m03s185_000123.h5"
H5_PATHS = [H5_PATH]  


@pytest.fixture
def config():
    config_dict = parse_config(
        config=config_path,
        user_config=None,
        system_config=None,
    )
    

    return config_dict


@pytest.fixture(name="config_dataframe")
def fixture_config_file_dataframe() -> dict:
    """Fixture providing a configuration file for CFELLoader tests.

    Returns:
        dict: The parsed configuration file.
    """
    return parse_config(config_path, folder_config={}, user_config={}, system_config={})[
        "dataframe"
    ]


@pytest.fixture(name="h5_file")
def fixture_h5_file() -> h5py.File:
    """Fixture providing an open h5 file.

    Returns:
        h5py.File: The open h5 file.
    """
    return h5py.File(os.path.join(test_dir, f"data/loader/cfel/{H5_PATH}"), "r")


@pytest.fixture(name="h5_file_copy")
def fixture_h5_file_copy(tmp_path: Path) -> h5py.File:
    """Fixture providing a copy of an open h5 file.

    Returns:
        h5py.File: The open h5 file copy.
    """
    # Create a copy of the h5 file in a temporary directory
    original_file_path = os.path.join(test_dir, f"data/loader/cfel/{H5_PATH}")
    copy_file_path = tmp_path / "copy.h5"
    shutil.copyfile(original_file_path, copy_file_path)

    return h5py.File(copy_file_path, "r+")


@pytest.fixture(name="h5_file2_copy")
def fixture_h5_file2_copy(tmp_path: Path) -> h5py.File:
    """Fixture providing a copy of an open h5 file.

    Returns:
        h5py.File: The open h5 file copy.
    """
    # Create a copy of the h5 file in a temporary directory
    original_file_path = os.path.join(test_dir, f"data/loader/cfel/{H5_PATHS[0] if len(H5_PATHS) > 1 else H5_PATH}")  # Use first file if multiple, else single file
    copy_file_path = tmp_path / "copy2.h5"
    shutil.copyfile(original_file_path, copy_file_path)

    # Open the copy in 'read-write' mode and return it
    return h5py.File(copy_file_path, "r+")


@pytest.fixture(name="h5_paths")
def fixture_h5_paths() -> list[Path]:
    """Fixture providing a list of h5 file paths.

    Returns:
        list: A list of h5 file paths.
    """
    return [Path(os.path.join(test_dir, f"data/loader/cfel/{path}")) for path in H5_PATHS]
