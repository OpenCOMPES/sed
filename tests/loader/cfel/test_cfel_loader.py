"""Tests for CFEL Loader functionality"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pytest

from .test_buffer_handler import create_parquet_dir
from sed.loader.cfel.loader import CFELLoader


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

    # Find base path of beamline from config. Here, we use cfel for CFEL loader
    base_path = config_["core"]["beamtime_dir"]["cfel"]
    expected_path = (
        Path(base_path) / config_["core"]["year"] / "data" / config_["core"]["beamtime_id"]
    )
    # Create expected paths
    expected_raw_path = expected_path / "raw" / sub_dir
    expected_processed_path = expected_path / "processed"

    # Create a fake file system for testing
    fs.create_dir(expected_raw_path)
    fs.create_dir(expected_processed_path)

    # Instance of class with correct config and call initialize_dirs
    fl = CFELLoader(config=config_)
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
        fl = CFELLoader(config=config_)
        fl._initialize_dirs()


def test_save_read_parquet_cfel(config: dict) -> None:
    """
    Test the functionality of saving and reading parquet files with CFELLoader.

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
    data_parquet_dir = create_parquet_dir(config_, "cfel_save_read")
    config_["core"]["paths"]["processed"] = data_parquet_dir
    # Update the raw path to point to the CFEL test data directory
    config_["core"]["paths"]["raw"] = "tests/data/loader/cfel/"
    fl = CFELLoader(config=config_)

    # First call: should create and read the parquet file
    df1, _, _ = fl.read_dataframe(runs=[123], force_recreate=True)  # was runs = [179]
    # Check if new files were created
    data_parquet_dir = data_parquet_dir.joinpath("buffer")
    new_files = {
        file: os.path.getmtime(data_parquet_dir.joinpath(file))
        for file in os.listdir(data_parquet_dir)
    }
    assert new_files

    # Second call: should only read the parquet file, not create new ones
    df2, _, _ = fl.read_dataframe(runs=[123])

    # Verify no new files were created after the second call
    final_files = {
        file: os.path.getmtime(data_parquet_dir.joinpath(file))
        for file in os.listdir(data_parquet_dir)
    }
    assert (
        new_files == final_files
    ), "Files were overwritten or new files were created after the second call."

    # Third call: We force_recreate the parquet files
    df3, _, _ = fl.read_dataframe(runs=[123], force_recreate=True)

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
    """Test get_elapsed_time method of CFELLoader class"""
    # Create an instance of CFELLoader
    fl = CFELLoader(config=config)

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

    # -------------------------
    # Aggregate=True → sum differences
    # -------------------------
    elapsed_total = fl.get_elapsed_time(fids=[0, 1], aggregate=True)
    expected_total = (20 - 10) + (30 - 20)  # 20
    assert elapsed_total == expected_total

    # -------------------------
    # Aggregate=False → list of per-file differences
    # -------------------------
    elapsed_list = fl.get_elapsed_time(fids=[0, 1], aggregate=False)
    expected_list = [(20 - 10), (30 - 20)]  # [10, 10]
    assert elapsed_list == expected_list

    # -------------------------
    # Test KeyError when file_statistics is missing
    # -------------------------
    fl.metadata = {"something": "else"}
    with pytest.raises(KeyError) as e:
        fl.get_elapsed_time(fids=[0, 1])
    assert "File statistics missing. Use 'read_dataframe' first." in str(e.value)

    # -------------------------
    # Test KeyError when timeStamp metadata is missing for a file
    # -------------------------
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
    assert "Timestamp metadata missing in file file0 (fid=0)" in str(e.value)


def test_get_elapsed_time_run(config: dict) -> None:
    """Test get_elapsed_time method for runs with multiple files"""
    config_ = config.copy()
    data_parquet_dir = create_parquet_dir(config_, "get_elapsed_time_run")
    config_["core"]["paths"]["processed"] = data_parquet_dir
    config_["core"]["paths"]["raw"] = "tests/data/loader/cfel/"

    # Create an instance of CFELLoader
    fl = CFELLoader(config=config_)

    # Read dataframe for run 123
    fl.read_dataframe(runs=[123])

    # Extract expected elapsed times per file from metadata
    file_stats = fl.metadata["file_statistics"]["electron"]
    expected_elapsed_list = [
        file_stats[str(fid)]["columns"]["timeStamp"]["max"]
        - file_stats[str(fid)]["columns"]["timeStamp"]["min"]
        for fid in range(len(fl.files))
    ]

    # -------------------------
    # Aggregate=False → list of per-file elapsed times
    # -------------------------
    elapsed_list = fl.get_elapsed_time(runs=[123], aggregate=False)
    assert elapsed_list == expected_elapsed_list

    # -------------------------
    # Aggregate=True → sum of per-file elapsed times
    # -------------------------
    elapsed_total = fl.get_elapsed_time(runs=[123], aggregate=True)
    expected_total = sum(expected_elapsed_list)
    assert elapsed_total == expected_total

    # -------------------------
    # Remove the parquet files created during test
    # -------------------------
    buffer_dir = Path(fl.processed_dir, "buffer")
    if buffer_dir.exists():
        for file in buffer_dir.iterdir():
            file.unlink()


def test_available_runs(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    """Test available_runs property of CFELLoader class"""
    # Create an instance of CFELLoader
    fl = CFELLoader(config=config)

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
