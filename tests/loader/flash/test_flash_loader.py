"""Tests for FlashLoader functionality"""
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import pytest

from sed.core.config import parse_config
from sed.loader.flash.loader import FlashLoader

package_dir = os.path.dirname(find_spec("sed").origin)
config_path = os.path.join(package_dir, "../tests/data/loader/flash/config.yaml")
H5_PATH = "FLASH1_USER3_stream_2_run43878_file1_20230130T153807.1.h5"


@pytest.fixture(name="config_file")
def fixture_config_file() -> dict:
    """Fixture providing a configuration file for FlashLoader tests.

    Returns:
        dict: The parsed configuration file.
    """
    return parse_config(config_path)


@pytest.mark.parametrize(
    "sub_dir",
    ["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"],
)
def test_initialize_dirs(
    config_file: dict,
    fs,
    sub_dir: Literal["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"],
) -> None:
    """
    Test the initialization of paths based on the configuration and directory structures.

    Args:
    fs: A fixture for a fake file system.
    sub_dir (Literal["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"]): Sub-directory.
    """
    config = config_file
    del config["core"]["paths"]
    config["core"]["beamtime_id"] = "12345678"
    config["core"]["year"] = "2000"

    # Find base path of beamline from config. Here, we use pg2
    base_path = config["dataframe"]["beamtime_dir"]["pg2"]
    expected_path = (
        Path(base_path) / config["core"]["year"] / "data" / config["core"]["beamtime_id"]
    )
    # Create expected paths
    expected_raw_path = expected_path / "raw" / "hdf" / sub_dir
    expected_processed_path = expected_path / "processed" / "parquet"

    # Create a fake file system for testing
    fs.create_dir(expected_raw_path)
    fs.create_dir(expected_processed_path)

    # Instance of class with correct config and call initialize_dirs
    fl = FlashLoader(config=config)
    fl._initialize_dirs()
    assert str(expected_raw_path) == fl.raw_dir
    assert str(expected_processed_path) == fl.parquet_dir

    # remove breamtimeid, year and daq from config to raise error
    del config["core"]["beamtime_id"]
    with pytest.raises(ValueError) as e:
        fl._initialize_dirs()
    print(e.value)
    assert "The beamtime_id and year are required." in str(e.value)


def test_initialize_dirs_filenotfound(config_file: dict) -> None:
    """
    Test FileNotFoundError during the initialization of paths.
    """
    # Test the FileNotFoundError
    config = config_file
    del config["core"]["paths"]
    config["core"]["beamtime_id"] = "11111111"
    config["core"]["year"] = "2000"

    # Instance of class with correct config and call initialize_dirs
    with pytest.raises(FileNotFoundError):
        fl = FlashLoader(config=config)
        fl._initialize_dirs()


def test_save_read_parquet_flash(config):
    """Test ParquetHandler save and read parquet"""
    config_ = config
    config_["core"]["paths"]["data_parquet_dir"] = (
        config_["core"]["paths"]["data_parquet_dir"] + "_flash_save_read/"
    )
    fl = FlashLoader(config=config_)
    df1, _, _ = fl.read_dataframe(runs=[43878, 43879])

    df2, _, _ = fl.read_dataframe(runs=[43878, 43879])


def test_get_elapsed_time_fid(config_file):
    """Test get_elapsed_time method of FlashLoader class"""
    # Create an instance of FlashLoader
    fl = FlashLoader(config=config_file)

    # Mock the file_statistics and files
    fl.metadata = {
        "file_statistics": {
            0: {"time_stamps": [10, 20]},
            1: {"time_stamps": [20, 30]},
            2: {"time_stamps": [30, 40]},
        },
    }
    fl.files = ["file0", "file1", "file2"]

    # Test get_elapsed_time with fids
    assert fl.get_elapsed_time(fids=[0, 1]) == 20

    # # Test get_elapsed_time with runs
    # # Assuming get_files_from_run_id(43878) returns ["file0", "file1"]
    # assert fl.get_elapsed_time(runs=[43878]) == 20

    # Test get_elapsed_time with aggregate=False
    assert fl.get_elapsed_time(fids=[0, 1], aggregate=False) == [10, 10]

    # Test KeyError when file_statistics is missing
    fl.metadata = {"something": "else"}
    with pytest.raises(KeyError) as e:
        fl.get_elapsed_time(fids=[0, 1])

    assert "File statistics missing. Use 'read_dataframe' first." in str(e.value)
    # Test KeyError when time_stamps is missing
    fl.metadata = {
        "file_statistics": {
            0: {},
            1: {"time_stamps": [20, 30]},
        },
    }
    with pytest.raises(KeyError) as e:
        fl.get_elapsed_time(fids=[0, 1])

    assert "Timestamp metadata missing in file 0" in str(e.value)


def test_get_elapsed_time_run(config_file):
    config = config_file
    config["core"]["paths"] = {
        "data_raw_dir": "tests/data/loader/flash/",
        "data_parquet_dir": "tests/data/loader/flash/parquet/get_elapsed_time_run",
    }
    """Test get_elapsed_time method of FlashLoader class"""
    # Create an instance of FlashLoader
    fl = FlashLoader(config=config_file)

    fl.read_dataframe(runs=[43878, 43879])
    start, end = fl.metadata["file_statistics"][0]["time_stamps"]
    expected_elapsed_time_0 = end - start
    start, end = fl.metadata["file_statistics"][1]["time_stamps"]
    expected_elapsed_time_1 = end - start

    elapsed_time = fl.get_elapsed_time(runs=[43878])
    assert elapsed_time == expected_elapsed_time_0

    elapsed_time = fl.get_elapsed_time(runs=[43878, 43879], aggregate=False)
    assert elapsed_time == [expected_elapsed_time_0, expected_elapsed_time_1]

    elapsed_time = fl.get_elapsed_time(runs=[43878, 43879])
    start, end = fl.metadata["file_statistics"][1]["time_stamps"]
    assert elapsed_time == expected_elapsed_time_0 + expected_elapsed_time_1
