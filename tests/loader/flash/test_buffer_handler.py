"""Test cases for the BufferHandler class in the Flash module."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from h5py import File

from sed.loader.flash.buffer_handler import BufferFilePaths
from sed.loader.flash.buffer_handler import BufferHandler
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
    fp = BufferFilePaths(config, h5_paths, folder, suffix="", remove_invalid_files=False)

    # check that all files are to be read
    assert len(fp.file_sets_to_process()) == len(h5_paths)
    print(folder)
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
    bh._save_buffer_file(path)

    # check again for files to read and expect one less file
    fp = BufferFilePaths(config, h5_paths, folder, suffix="", remove_invalid_files=False)
    # check that only one file is to be read
    assert len(fp.file_sets_to_process()) == len(h5_paths) - 1

    # check that both files are to be read if force_recreate is set to True
    assert len(fp.file_sets_to_process(force_recreate=True)) == len(h5_paths)

    # remove buffer files
    Path(path["electron"]).unlink()
    Path(path["timed"]).unlink()

    # Test for adding a suffix
    fp = BufferFilePaths(config, h5_paths, folder, "suffix", remove_invalid_files=False)

    # expected buffer paths with prefix and suffix
    for typ in ["electron", "timed"]:
        expected_buffer_paths = [
            folder / "buffer" / f"{typ}_{Path(path).stem}_suffix" for path in h5_paths
        ]
        assert np.all(fp[typ] == expected_buffer_paths)


def test_buffer_schema_mismatch(config: dict, h5_paths: list[Path]) -> None:
    """
    Test function to verify schema mismatch handling in the FlashLoader's 'read_dataframe' method.

    The test validates the error handling mechanism when the available channels do not match the
    schema of the existing parquet files.

    Test Steps:
    - Attempt to read a dataframe after adding a new channel 'gmdTunnel2' to the configuration.
    - Check for an expected error related to the mismatch between available channels and schema.
    - Force recreation of dataframe with the added channel, ensuring successful dataframe
      creation.
    - Simulate a missing channel scenario by removing 'gmdTunnel2' from the configuration.
    - Check for an error indicating a missing channel in the configuration.
    - Clean up created buffer files after the test.
    """
    folder = create_parquet_dir(config, "schema_mismatch")
    bh = BufferHandler(config)
    bh.process_and_load_dataframe(h5_paths=h5_paths, folder=folder, debug=True)

    # Manipulate the configuration to introduce a new channel 'gmdTunnel2'
    config_dict = config
    config_dict["dataframe"]["channels"]["gmdTunnel2"] = {
        "index_key": "/FL1/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/index",
        "dataset_key": "/FL1/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/value",
        "format": "per_pulse",
        "slice": 0,
    }

    # Reread the dataframe with the modified configuration, expecting a schema mismatch error
    with pytest.raises(ValueError) as e:
        bh = BufferHandler(config)
        bh.process_and_load_dataframe(h5_paths=h5_paths, folder=folder, debug=True)
    expected_error = e.value.args[0]

    # Validate the specific error messages for schema mismatch
    assert "The available channels do not match the schema of file" in expected_error
    assert "Missing in parquet: {'gmdTunnel2'}" in expected_error
    assert "Please check the configuration file or set force_recreate to True." in expected_error

    # Force recreation of the dataframe, including the added channel 'gmdTunnel2'
    bh = BufferHandler(config)
    bh.process_and_load_dataframe(h5_paths=h5_paths, folder=folder, force_recreate=True, debug=True)

    # Remove 'gmdTunnel2' from the configuration to simulate a missing channel scenario
    del config["dataframe"]["channels"]["gmdTunnel2"]
    # also results in error but different from before
    with pytest.raises(ValueError) as e:
        # Attempt to read the dataframe again to check for the missing channel error
        bh = BufferHandler(config)
        bh.process_and_load_dataframe(h5_paths=h5_paths, folder=folder, debug=True)

    expected_error = e.value.args[0]
    # Check for the specific error message indicating a missing channel in the configuration
    assert "Missing in config: {'gmdTunnel2'}" in expected_error

    # Clean up created buffer files after the test
    for path in bh.fp["electron"]:
        path.unlink()
    for path in bh.fp["timed"]:
        path.unlink()


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
    """Test function to verify exception handling in the BufferHandler's
    'process_and_load_dataframe' method. The test checks for exceptions raised due to missing
    channels in the configuration and empty datasets.
    Test Steps:
    - Create a directory structure for storing buffer files and initialize the BufferHandler.
    - Check for an exception when a channel is missing in the configuration.
    - Create an empty dataset in the HDF5 file to simulate an invalid file scenario.
    - Check for an expected error related to the missing index dataset that invalidates the file.
    - Check for an error when 'remove_invalid_files' is set to True and the file is invalid.
    - Create an empty dataset in the second HDF5 file to simulate an invalid file scenario.
    - Check for an error when 'remove_invalid_files' is set to True and the file is invalid.
    - Check for an error when only a single file is provided, and the file is not buffered.
    """
    folder_parallel = create_parquet_dir(config, "save_buffer_files_exception")
    config_ = deepcopy(config)

    # check exception in case of missing channel in config
    channel = "dldPosX"
    del config_["dataframe"]["channels"][channel]["index_key"]

    # testing exception in parallel execution
    with pytest.raises(ValueError):
        bh = BufferHandler(config_)
        bh.process_and_load_dataframe(h5_paths, folder_parallel, debug=False)

    # check exception message with empty dataset
    config_ = deepcopy(config)
    channel = "testChannel"
    channel_index_key = "test/dataset/empty/index"
    empty_dataset_key = "test/dataset/empty/value"
    config_["dataframe"]["channels"][channel] = {
        "index_key": channel_index_key,
        "dataset_key": empty_dataset_key,
        "format": "per_train",
    }

    # create an empty dataset
    h5_file_copy.create_dataset(
        name=empty_dataset_key,
        shape=0,
    )

    # expect invalid file error because of missing index dataset that invalidates entire file
    with pytest.raises(InvalidFileError):
        bh = BufferHandler(config_)
        bh.process_and_load_dataframe(
            [tmp_path / "copy.h5"],
            folder_parallel,
            debug=False,
            force_recreate=True,
        )

    # create an empty dataset
    h5_file2_copy.create_dataset(
        name=channel_index_key,
        shape=0,
    )
    h5_file2_copy.create_dataset(
        name=empty_dataset_key,
        shape=0,
    )

    # if remove_invalid_files is True, the file should be removed and no error should be raised
    bh = BufferHandler(config_)
    try:
        bh.process_and_load_dataframe(
            [tmp_path / "copy.h5", tmp_path / "copy2.h5"],
            folder_parallel,
            debug=False,
            force_recreate=True,
            remove_invalid_files=True,
        )
    except InvalidFileError:
        assert (
            False
        ), "InvalidFileError should not be raised when remove_invalid_files is set to True"

    # with only a single file, the file will not be buffered so a FileNotFoundError should be raised
    with pytest.raises(FileNotFoundError):
        bh.process_and_load_dataframe(
            [tmp_path / "copy.h5"],
            folder_parallel,
            debug=False,
            force_recreate=True,
            remove_invalid_files=True,
        )


def test_get_filled_dataframe(config: dict, h5_paths: list[Path]) -> None:
    """Test function to verify the creation of a filled dataframe from the buffer files."""
    folder = create_parquet_dir(config, "get_filled_dataframe")
    bh = BufferHandler(config)
    bh.process_and_load_dataframe(h5_paths, folder)

    df = pd.read_parquet(folder)

    assert np.all(list(bh.df["electron"].columns) == list(df.columns) + ["dldSectorID"])

    channel_pulse = set(
        get_channels(
            config["dataframe"],
            formats=["per_pulse", "per_train"],
            index=True,
            extend_aux=True,
        ),
    ) - {"electronId"}
    assert np.all(set(bh.df["timed"].columns) == channel_pulse)
    # remove buffer files
    for df_type in ["electron", "timed"]:
        for path in bh.fp[df_type]:
            path.unlink()
