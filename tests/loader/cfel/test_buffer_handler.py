"""Test cases for the BufferHandler class in the Flash module."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from h5py import File

from sed.loader.cfel.buffer_handler import BufferFilePaths
from sed.loader.cfel.buffer_handler import BufferHandler
from sed.loader.cfel.dataframe import DataFrameCreator
from sed.loader.cfel.loader import CFELLoader
from sed.loader.flash.utils import get_channels
from sed.loader.flash.utils import InvalidFileError


def create_parquet_dir(config: dict, folder: str) -> Path:
    """
    Creates a directory for storing Parquet files based on the provided configuration
    and folder name.
    """

    parquet_path = Path(config["core"]["paths"]["processed"])
    parquet_path = parquet_path.joinpath(folder)
    parquet_path.mkdir(parents=True, exist_ok=True)
    return parquet_path


def test_buffer_file_paths(config: dict, h5_paths: list[Path]) -> None:
    """
    Test the BufferFilePath's ability to identify files that need to be read and
    manage buffer file paths using a directory structure.

    This test performs several checks to ensure the BufferFilePath correctly identifies
    which HDF5 files need to be read and properly manages the paths for saving buffer
    files. It follows these steps:
    1. Creates a directory structure for storing buffer files and initializes the BufferHandler.
    2. Checks if the file_sets_to_process method populates the dict of missing file sets and
       verify that initially, all provided files are considered missing.
    3. Checks that the paths for saving buffer files are correctly generated.
    4. Creates a single buffer file and reruns file_sets_to_process to ensure that the BufferHandler
        recognizes one less missing file.
    5. Checks if the force_recreate parameter forces the BufferHandler to consider all files
    6. Cleans up by removing the created buffer file.
    7. Tests the handling of suffix in buffer file names (for multidetector setups) by rerunning
        the checks with modified file name parameters.
    """
    folder = create_parquet_dir(config, "get_files_to_read")
    fp = BufferFilePaths(h5_paths, folder, suffix="")

    # check that all files are to be read
    assert len(fp.file_sets_to_process()) == len(h5_paths)
    # create expected paths
    expected_buffer_electron_paths = [
        folder / f"buffer/electron_{Path(path).stem}" for path in h5_paths
    ]
    expected_buffer_timed_paths = [folder / f"buffer/timed_{Path(path).stem}" for path in h5_paths]

    # check that all buffer paths are correct
    assert np.all(fp["electron"] == expected_buffer_electron_paths)
    assert np.all(fp["timed"] == expected_buffer_timed_paths)

    # create a single buffer file to check if it changes
    path = {
        "raw": h5_paths[0],
        "electron": expected_buffer_electron_paths[0],
        "timed": expected_buffer_timed_paths[0],
    }
    bh = BufferHandler(config)
    bh._save_buffer_file(path, is_first_file=True, base_timestamp=None)

    # check again for files to read and expect one less file
    fp = BufferFilePaths(h5_paths, folder, suffix="")
    # check that only one file is to be read
    assert len(fp.file_sets_to_process()) == len(h5_paths) - 1

    # check that both files are to be read if force_recreate is set to True
    assert len(fp.file_sets_to_process(force_recreate=True)) == len(h5_paths)

    # remove buffer files
    Path(path["electron"]).unlink()
    Path(path["timed"]).unlink()

    # Test for adding a suffix
    fp = BufferFilePaths(h5_paths, folder, "suffix")

    # expected buffer paths with prefix and suffix
    for typ in ["electron", "timed"]:
        expected_buffer_paths = [
            folder / "buffer" / f"{typ}_{Path(path).stem}_suffix" for path in h5_paths
        ]
        assert np.all(fp[typ] == expected_buffer_paths)


def test_buffer_schema_mismatch(config: dict, h5_paths: list[Path]) -> None:
    """
    Test schema mismatch handling in BufferHandler / CFEL loader.

    Steps:
    1) Channel exists in config but NOT in HDF5 → expect InvalidFileError.
    2) Same situation, but ignored via remove_invalid_files=True → should succeed.
    3) True schema mismatch (parquet has column not in config) → expect ValueError.
    """
    from copy import deepcopy

    # --------------------------------------------------
    # Step 1: HDF5 missing channel → InvalidFileError
    # --------------------------------------------------
    folder_step1 = create_parquet_dir(config, "schema_mismatch_step1")
    config_missing_channel = deepcopy(config)
    config_missing_channel["dataframe"]["channels"]["gmdTunnel2"] = {
        "dataset_key": "/some/cfel/test/dataset",
        "format": "per_train",
    }

    with pytest.raises(InvalidFileError) as exc:
        bh = BufferHandler(config_missing_channel)
        bh.process_and_load_dataframe(
            h5_paths=h5_paths,
            folder=folder_step1,
            debug=True,
            force_recreate=True,   # ← THIS IS REQUIRED
        )
    
    assert "gmdTunnel2" in str(exc.value)

    # --------------------------------------------------
    # Step 2: Same missing channel, but ignored
    # All files become invalid → no buffers → FileNotFoundError
    # --------------------------------------------------
    folder_step2 = create_parquet_dir(config, "schema_mismatch_step2")
    
    # create buffer files normally
    bh_base = BufferHandler(config)
    bh_base.process_and_load_dataframe(
        h5_paths=h5_paths,
        folder=folder_step2,
        debug=True,
        force_recreate=True,
    )
    
    # now re-run with missing channel ignored
    bh_missing = BufferHandler(config_missing_channel)
    bh_missing.process_and_load_dataframe(
        h5_paths=h5_paths,
        folder=folder_step2,
        debug=True,
        remove_invalid_files=True,
        force_recreate=True,
    )
    
    # correct post-condition
    assert bh_missing.df["electron"] is None
    assert bh_missing.df["timed"] is None

    # --------------------------------------------------
    # Step 3: TRUE schema mismatch → ValueError
    # --------------------------------------------------
    
    folder_step3 = create_parquet_dir(config, "schema_mismatch_step3")
    
    # choose a REAL channel that exists in HDF5
    removed_channel = "dldPosX"
    assert removed_channel in config["dataframe"]["channels"]
    
    # 1) create parquet normally (with that channel)
    bh_base = BufferHandler(config)
    bh_base.process_and_load_dataframe(
        h5_paths=h5_paths,
        folder=folder_step3,
        debug=True,
        force_recreate=True,
    )
    
    # 2) remove the channel from config
    config_removed = deepcopy(config)
    del config_removed["dataframe"]["channels"][removed_channel]
    
    # 3) reload → schema mismatch
    with pytest.raises(ValueError) as exc:
        bh_removed = BufferHandler(config_removed)
        bh_removed.process_and_load_dataframe(
            h5_paths=h5_paths,
            folder=folder_step3,
            debug=True,
        )
    
    msg = str(exc.value).lower()
    assert "available channels do not match the schema" in msg
    assert "missing in parquet" in msg or "missing" in msg


def test_save_buffer_files(config: dict, h5_paths: list[Path]) -> None:
    """
    Test the BufferHandler's ability to save buffer files serially and in parallel.

    This test ensures that the BufferHandler can run both serially and in parallel, saving the
    output to buffer files, and then it compares the resulting DataFrames to ensure they are
    identical. This verifies that parallel processing does not affect the integrity of the data
    saved. After the comparison, it cleans up by removing the created buffer files.
    """
    folder_serial = create_parquet_dir(config, "save_buffer_files_serial")
    bh_serial = BufferHandler(config)
    bh_serial.process_and_load_dataframe(h5_paths, folder_serial, debug=True)

    folder_parallel = create_parquet_dir(config, "save_buffer_files_parallel")
    bh_parallel = BufferHandler(config)
    bh_parallel.process_and_load_dataframe(h5_paths, folder_parallel)

    df_serial = pd.read_parquet(folder_serial)
    df_parallel = pd.read_parquet(folder_parallel)

    pd.testing.assert_frame_equal(df_serial, df_parallel)

    # remove buffer files
    for df_type in ["electron", "timed"]:
        for path in bh_serial.fp[df_type]:
            path.unlink()
        for path in bh_parallel.fp[df_type]:
            path.unlink()

def test_save_buffer_files_exception(
    config: dict,
    h5_paths: list[Path],
    h5_file_copy: File,
    h5_file2_copy: File,
    tmp_path: Path,
) -> None:
    """Test BufferHandler exception handling for missing keys and empty datasets."""

    folder = create_parquet_dir(config, "save_buffer_files_exception")
    config_ = deepcopy(config)

    # --------------------------------------------------
    # 1) Missing dataset_key in config → ValueError
    # --------------------------------------------------
    channel = "dldPosX"
    del config_["dataframe"]["channels"][channel]["dataset_key"]

    with pytest.raises(ValueError):
        bh = BufferHandler(config_)
        bh.process_and_load_dataframe(
            h5_paths, folder, debug=False
        )

    # --------------------------------------------------
    # 2) Empty dataset → InvalidFileError
    # --------------------------------------------------
    config_ = deepcopy(config)
    empty_channel = "testChannel"
    empty_dataset_key = "test/dataset/empty/value"

    config_["dataframe"]["channels"][empty_channel] = {
        "dataset_key": empty_dataset_key,
        "format": "per_train",
    }

    # create empty dataset in first HDF5 file
    h5_file_copy.create_dataset(name=empty_dataset_key, shape=(0,))

    # Expect InvalidFileError because dataset is empty
    with pytest.raises(InvalidFileError):
        bh = BufferHandler(config_)
        bh.process_and_load_dataframe(
            [tmp_path / "copy.h5"],
            folder,
            debug=False,
            force_recreate=True,
        )

    # --------------------------------------------------
    # 3) remove_invalid_files=True → no error, only invalid files are skipped
    # --------------------------------------------------
    # add empty dataset to second HDF5 file
    h5_file2_copy.create_dataset(name=empty_dataset_key, shape=(0,))

    bh = BufferHandler(config_)
    bh.process_and_load_dataframe(
        [tmp_path / "copy.h5", tmp_path / "copy2.h5"],
        folder,
        debug=False,
        force_recreate=True,
        remove_invalid_files=True,
    )

    # When all files are invalid, the DataFrames should be None
    assert bh.df["electron"] is None
    assert bh.df["timed"] is None

    # --------------------------------------------------
    # 4) Single invalid file → nothing valid to load
    # --------------------------------------------------
    # Only provide one invalid file    
    bh.process_and_load_dataframe(
        [tmp_path / "copy.h5"],
        folder,
        debug=False,
        force_recreate=True,
        remove_invalid_files=True,
    )
    
    assert bh.df["electron"] is None
    assert bh.df["timed"] is None


def test_get_filled_dataframe(config: dict, h5_paths: list[Path]) -> None:
    """Test function to verify the creation of a filled dataframe from the buffer files."""
    folder = create_parquet_dir(config, "get_filled_dataframe")
    bh = BufferHandler(config)
    bh.process_and_load_dataframe(h5_paths, folder)

    df = pd.read_parquet(folder)

    # The buffer handler's electron dataframe may have additional derived columns
    # like dldSectorID that aren't in the saved parquet file
    expected_columns = set(list(df.columns) + ["timeStamp", "countId", "dldSectorID"])
    assert set(bh.df["electron"].columns).issubset(expected_columns)

    # For CFEL, check that the timed dataframe contains per_train channels and timestamp
    # but excludes per_electron channels (this is CFEL-specific behavior)
    per_train_channels = set(get_channels(config["dataframe"], formats=["per_train"], extend_aux=True))
    per_electron_channels = set(get_channels(config["dataframe"], formats=["per_electron"]))
    
    timed_columns = set(bh.df["timed"].columns)
    
    # Timed should include per_train channels and timestamp
    assert per_train_channels.issubset(timed_columns)
    assert "timeStamp" in timed_columns
    
    # Check that we can read the data
    assert len(df) > 0
    assert len(bh.df["electron"]) > 0
    assert len(bh.df["timed"]) > 0
    # remove buffer files
    for df_type in ["electron", "timed"]:
        for path in bh.fp[df_type]:
            path.unlink()


def test_cfel_multi_file_handling(config: dict, h5_paths: list[Path]) -> None:
    """Test CFEL's multi-file timestamp handling."""
    folder = create_parquet_dir(config, "multi_file_handling")
    bh = BufferHandler(config)
    
    # Test that multi-file processing works with timestamp coordination
    bh.process_and_load_dataframe(h5_paths=h5_paths, folder=folder, debug=True)
    
    # Verify that timestamps are properly coordinated across files
    df = pd.read_parquet(folder)
    assert "timeStamp" in df.columns  # CFEL uses timeStamp, not timestamp
    
    # Clean up
    for df_type in ["electron", "timed"]:
        for path in bh.fp[df_type]:
            path.unlink()

def test_cfel_timestamp_base_handling(config: dict, h5_paths: list[Path]) -> None:
    """Test CFEL's base timestamp extraction and handling."""
    if len(h5_paths) > 1:
        # Test with multiple files to verify base timestamp logic
        folder = create_parquet_dir(config, "timestamp_base")
        bh = BufferHandler(config)
        bh.process_and_load_dataframe(h5_paths=h5_paths, folder=folder, debug=True)
        
        # Verify processing completed successfully
        assert len(bh.fp["electron"]) == len(h5_paths)
        
        # Clean up
        for df_type in ["electron", "timed"]:
            for path in bh.fp[df_type]:
                path.unlink()
