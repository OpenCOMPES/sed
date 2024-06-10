from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sed.loader.flash.buffer_handler import BufferHandler
from sed.loader.flash.utils import get_channels


def create_parquet_dir(config, folder):
    parquet_path = Path(config["core"]["paths"]["data_parquet_dir"])
    parquet_path = parquet_path.joinpath(folder)
    parquet_path.mkdir(parents=True, exist_ok=True)
    return parquet_path


def test_get_files_to_read(config, h5_paths):
    folder = create_parquet_dir(config, "get_files_to_read")
    subfolder = folder.joinpath("buffer")
    # set to false to avoid creating buffer files unnecessarily
    bh = BufferHandler(config)
    bh._get_files_to_read(h5_paths, folder, "", "", False)

    assert len(bh.missing_h5_files) == len(h5_paths)
    assert len(bh.save_paths) == len(h5_paths)

    assert np.all(bh.missing_h5_files == h5_paths)

    # create expected paths
    expected_buffer_paths = [Path(subfolder, f"{Path(path).stem}") for path in h5_paths]

    assert np.all(bh.save_paths == expected_buffer_paths)

    # create only one buffer file
    bh._save_buffer_file(h5_paths[0], expected_buffer_paths[0])
    # check again for files to read
    bh._get_files_to_read(h5_paths, folder, "", "", False)
    # check that only one file is to be read
    assert len(bh.missing_h5_files) == len(h5_paths) - 1
    Path(expected_buffer_paths[0]).unlink()  # remove buffer file

    # add prefix and suffix
    bh._get_files_to_read(h5_paths, folder, "prefix", "suffix", False)

    # expected buffer paths with prefix and suffix
    expected_buffer_paths = [
        Path(subfolder, f"prefix_{Path(path).stem}_suffix") for path in h5_paths
    ]
    assert np.all(bh.save_paths == expected_buffer_paths)


def test_buffer_schema_mismatch(config, h5_paths):
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
    bh.run(h5_paths=h5_paths, folder=folder, debug=True)

    # Manipulate the configuration to introduce a new channel 'gmdTunnel2'
    config_dict = config
    config_dict["dataframe"]["channels"]["gmdTunnel2"] = {
        "group_name": "/FL1/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/",
        "format": "per_pulse",
        "slice": 0,
    }

    # Reread the dataframe with the modified configuration, expecting a schema mismatch error
    with pytest.raises(ValueError) as e:
        bh = BufferHandler(config)
        bh.run(h5_paths=h5_paths, folder=folder, debug=True)
    expected_error = e.value.args[0]

    # Validate the specific error messages for schema mismatch
    assert "The available channels do not match the schema of file" in expected_error
    assert "Missing in parquet: {'gmdTunnel2'}" in expected_error
    assert "Please check the configuration file or set force_recreate to True." in expected_error

    # Force recreation of the dataframe, including the added channel 'gmdTunnel2'
    bh = BufferHandler(config)
    bh.run(h5_paths=h5_paths, folder=folder, force_recreate=True, debug=True)

    # Remove 'gmdTunnel2' from the configuration to simulate a missing channel scenario
    del config["dataframe"]["channels"]["gmdTunnel2"]
    # also results in error but different from before
    with pytest.raises(ValueError) as e:
        # Attempt to read the dataframe again to check for the missing channel error
        bh = BufferHandler(config)
        bh.run(h5_paths=h5_paths, folder=folder, debug=True)

    expected_error = e.value.args[0]
    # Check for the specific error message indicating a missing channel in the configuration
    assert "Missing in config: {'gmdTunnel2'}" in expected_error

    # Clean up created buffer files after the test
    [path.unlink() for path in bh.buffer_paths]


def test_save_buffer_files(config, h5_paths):
    folder_serial = create_parquet_dir(config, "save_buffer_files_serial")
    bh_serial = BufferHandler(config)
    bh_serial.run(h5_paths, folder_serial, debug=True)

    folder_parallel = create_parquet_dir(config, "save_buffer_files_parallel")
    bh_parallel = BufferHandler(config)
    bh_parallel.run(h5_paths, folder_parallel)

    df_serial = pd.read_parquet(folder_serial)
    df_parallel = pd.read_parquet(folder_parallel)

    pd.testing.assert_frame_equal(df_serial, df_parallel)

    # remove buffer files
    [path.unlink() for path in bh_serial.buffer_paths]
    [path.unlink() for path in bh_parallel.buffer_paths]


def test_get_filled_dataframe(config, h5_paths):
    """Test function to verify the creation of a filled dataframe from the buffer files."""
    folder = create_parquet_dir(config, "get_filled_dataframe")
    bh = BufferHandler(config)
    bh.run(h5_paths, folder)

    df = pd.read_parquet(folder)

    assert np.all(list(bh.df_electron.columns) == list(df.columns) + ["dldSectorID"])

    channel_pulse = get_channels(
        config["dataframe"]["channels"],
        formats=["per_pulse", "per_train"],
        index=True,
        extend_aux=True,
    )
    assert np.all(list(bh.df_pulse.columns) == channel_pulse)
    # remove buffer files
    [path.unlink() for path in bh.buffer_paths]
