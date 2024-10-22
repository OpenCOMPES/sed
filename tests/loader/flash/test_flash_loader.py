"""Tests for FlashLoader functionality"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pytest

from .test_buffer_handler import create_parquet_dir
from sed.loader.flash.loader import FlashLoader


@pytest.mark.parametrize(
    "sub_dir",
    ["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"],
)
def test_initialize_dirs(
    config: dict,
    fs,
    sub_dir: Literal["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"],
) -> None:
    """
    Test the initialization of paths based on the configuration and directory structures.

    Args:
    fs: A fixture for a fake file system.
    sub_dir (Literal["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"]): Sub-directory.
    """
    config_ = config.copy()
    del config_["core"]["paths"]
    config_["core"]["beamtime_id"] = "12345678"
    config_["core"]["year"] = "2000"

    # Find base path of beamline from config. Here, we use pg2
    base_path = config_["core"]["beamtime_dir"]["pg2"]
    expected_path = (
        Path(base_path) / config_["core"]["year"] / "data" / config_["core"]["beamtime_id"]
    )
    # Create expected paths
    expected_raw_path = expected_path / "raw" / "hdf" / sub_dir
    expected_processed_path = expected_path / "processed"

    # Create a fake file system for testing
    fs.create_dir(expected_raw_path)
    fs.create_dir(expected_processed_path)

    # Instance of class with correct config and call initialize_dirs
    fl = FlashLoader(config=config_)
    fl._initialize_dirs()
    assert str(expected_raw_path) == fl.raw_dir
    assert str(expected_processed_path) == fl.processed_dir

    # remove beamtime_id, year and daq from config to raise error
    del config_["core"]["beamtime_id"]
    with pytest.raises(ValueError) as e:
        fl._initialize_dirs()
    assert "The beamtime_id and year are required." in str(e.value)


def test_initialize_dirs_filenotfound(config: dict) -> None:
    """
    Test FileNotFoundError during the initialization of paths.
    """
    # Test the FileNotFoundError
    config_ = config.copy()
    del config_["core"]["paths"]
    config_["core"]["beamtime_id"] = "11111111"
    config_["core"]["year"] = "2000"

    # Instance of class with correct config and call initialize_dirs
    with pytest.raises(FileNotFoundError):
        fl = FlashLoader(config=config_)
        fl._initialize_dirs()


def test_save_read_parquet_flash(config: dict) -> None:
    """
    Test the functionality of saving and reading parquet files with FlashLoader.

    This test performs three main actions:
    1. First call to create and read parquet files. Verifies new files are created.
    2. Second call with the same parameters to check that it only reads from
    the existing parquet files without creating new ones. It asserts that the files' modification
    times remain unchanged, indicating no new files were created or existing files overwritten.
    3. Third call with `force_recreate=True` to force the recreation of parquet files.
    It verifies that the files were indeed overwritten by checking that their modification
    times have changed.
    """
    config_ = config.copy()
    data_parquet_dir = create_parquet_dir(config_, "flash_save_read")
    config_["core"]["paths"]["processed"] = data_parquet_dir
    fl = FlashLoader(config=config_)

    # First call: should create and read the parquet file
    df1, _, _ = fl.read_dataframe(runs=[43878, 43879])
    # Check if new files were created
    data_parquet_dir = data_parquet_dir.joinpath("buffer")
    new_files = {
        file: os.path.getmtime(data_parquet_dir.joinpath(file))
        for file in os.listdir(data_parquet_dir)
    }
    assert new_files

    # Second call: should only read the parquet file, not create new ones
    df2, _, _ = fl.read_dataframe(runs=[43878, 43879])

    # Verify no new files were created after the second call
    final_files = {
        file: os.path.getmtime(data_parquet_dir.joinpath(file))
        for file in os.listdir(data_parquet_dir)
    }
    assert (
        new_files == final_files
    ), "Files were overwritten or new files were created after the second call."

    # Third call: We force_recreate the parquet files
    df3, _, _ = fl.read_dataframe(runs=[43878, 43879], force_recreate=True)

    # Verify files were overwritten
    new_files = {
        file: os.path.getmtime(data_parquet_dir.joinpath(file))
        for file in os.listdir(data_parquet_dir)
    }
    assert new_files != final_files, "Files were not overwritten after the third call."

    # remove the parquet files
    for file in new_files:
        data_parquet_dir.joinpath(file).unlink()


def test_get_elapsed_time_fid(config: dict) -> None:
    """Test get_elapsed_time method of FlashLoader class"""
    # Create an instance of FlashLoader
    fl = FlashLoader(config=config)

    # Mock the file_statistics and files
    fl.metadata = {
        "file_statistics": {
            "timed": {
                "0": {"columns": {"timeStamp": {"min": 10, "max": 20}}},
                "1": {"columns": {"timeStamp": {"min": 20, "max": 30}}},
                "2": {"columns": {"timeStamp": {"min": 30, "max": 40}}},
            },
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
            "timed": {
                "0": {},
                "1": {"columns": {"timeStamp": {"min": 20, "max": 30}}},
            },
        },
    }
    with pytest.raises(KeyError) as e:
        fl.get_elapsed_time(fids=[0, 1])

    assert "Timestamp metadata missing in file 0" in str(e.value)


def test_get_elapsed_time_run(config: dict) -> None:
    """Test get_elapsed_time method of FlashLoader class"""
    config_ = config.copy()
    config_["core"]["paths"] = {
        "raw": "tests/data/loader/flash/",
        "processed": "tests/data/loader/flash/parquet/get_elapsed_time_run",
    }
    config_ = config.copy()
    data_parquet_dir = create_parquet_dir(config_, "get_elapsed_time_run")
    config_["core"]["paths"]["processed"] = data_parquet_dir
    # Create an instance of FlashLoader
    fl = FlashLoader(config=config_)

    fl.read_dataframe(runs=[43878, 43879])
    min_max = fl.metadata["file_statistics"]["electron"]["0"]["columns"]["timeStamp"]
    expected_elapsed_time_0 = min_max["max"] - min_max["min"]
    min_max = fl.metadata["file_statistics"]["electron"]["1"]["columns"]["timeStamp"]
    expected_elapsed_time_1 = min_max["max"] - min_max["min"]

    elapsed_time = fl.get_elapsed_time(runs=[43878])
    assert elapsed_time == expected_elapsed_time_0

    elapsed_time = fl.get_elapsed_time(runs=[43878, 43879], aggregate=False)
    assert elapsed_time == [expected_elapsed_time_0, expected_elapsed_time_1]

    elapsed_time = fl.get_elapsed_time(runs=[43878, 43879])
    assert elapsed_time == expected_elapsed_time_0 + expected_elapsed_time_1

    # remove the parquet files
    for file in os.listdir(Path(fl.processed_dir, "buffer")):
        Path(fl.processed_dir, "buffer").joinpath(file).unlink()


def test_available_runs(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    """Test available_runs property of FlashLoader class"""
    # Create an instance of FlashLoader
    fl = FlashLoader(config=config)

    # Mock the raw_dir and files
    fl.raw_dir = "/path/to/raw_dir"
    files = [
        "run1_file1.h5",
        "run3_file1.h5",
        "run2_file1.h5",
        "run1_file2.h5",
    ]

    # Mock the glob method to return the mock files
    def mock_glob(*args, **kwargs):  # noqa: ARG001
        return [Path(fl.raw_dir, file) for file in files]

    monkeypatch.setattr(Path, "glob", mock_glob)

    # Test available_runs
    assert fl.available_runs == [1, 2, 3]
