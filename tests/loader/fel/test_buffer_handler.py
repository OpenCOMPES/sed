from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sed.loader.fel import BufferHandler
from sed.loader.fel.utils import get_channels
from sed.loader.flash.dataframe import FlashDataFrameCreator


def create_parquet_dir(config, folder):
    parquet_path = Path(config.core.paths.data_parquet_dir)
    parquet_path = parquet_path.joinpath(folder)
    parquet_path.mkdir(parents=True, exist_ok=True)
    return parquet_path


def test_get_files_to_read(config, h5_paths):
    folder = create_parquet_dir(config, "get_files_to_read")
    subfolder = folder.joinpath("buffer")
    # set to false to avoid creating buffer files unnecessarily
    bh = BufferHandler(FlashDataFrameCreator, config.dataframe, h5_paths, folder, auto=False)
    bh.get_files_to_read(h5_paths, folder, "", "", False)

    assert bh.num_files == len(h5_paths)
    assert len(bh.buffer_to_create) == len(h5_paths)

    assert np.all(bh.h5_to_create == h5_paths)

    # create expected paths
    expected_buffer_paths = [Path(subfolder, f"{Path(path).stem}") for path in h5_paths]

    assert np.all(bh.buffer_to_create == expected_buffer_paths)

    # create only one buffer file
    bh._create_buffer_file(h5_paths[0], expected_buffer_paths[0])
    # check again for files to read
    bh.get_files_to_read(h5_paths, folder, "", "", False)
    # check that only one file is to be read
    assert bh.num_files == len(h5_paths) - 1
    Path(expected_buffer_paths[0]).unlink()  # remove buffer file

    # add prefix and suffix
    bh.get_files_to_read(h5_paths, folder, "prefix_", "_suffix", False)

    # expected buffer paths with prefix and suffix
    expected_buffer_paths = [
        Path(subfolder, f"prefix_{Path(path).stem}_suffix") for path in h5_paths
    ]
    assert np.all(bh.buffer_to_create == expected_buffer_paths)


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
    bh = BufferHandler(
        FlashDataFrameCreator,
        config.dataframe,
        h5_paths,
        folder,
        auto=True,
        debug=True,
    )

    # Manipulate the configuration to introduce a new channel 'gmdTunnel2'
    config_alt = config
    gmdTunnel2 = config_alt.dataframe.channels["gmdTunnel"]
    gmdTunnel2.group_name = "/FL1/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/"
    gmdTunnel2.format = "per_pulse"
    gmdTunnel2.slice = 0
    config_alt.dataframe.channels["gmdTunnel2"] = gmdTunnel2

    # Reread the dataframe with the modified configuration, expecting a schema mismatch error
    with pytest.raises(ValueError) as e:
        bh = BufferHandler(
            FlashDataFrameCreator,
            config.dataframe,
            h5_paths,
            folder,
            auto=True,
            debug=True,
        )
    expected_error = e.value.args

    # Validate the specific error messages for schema mismatch
    assert "The available channels do not match the schema of file" in expected_error[0]
    assert expected_error[2] == "Missing in parquet: {'gmdTunnel2'}"
    assert expected_error[4] == "Please check the configuration file or set force_recreate to True."

    # Force recreation of the dataframe, including the added channel 'gmdTunnel2'
    bh = BufferHandler(
        FlashDataFrameCreator,
        config.dataframe,
        h5_paths,
        folder,
        auto=True,
        force_recreate=True,
        debug=True,
    )

    # Remove 'gmdTunnel2' from the configuration to simulate a missing channel scenario
    del config.dataframe.channels["gmdTunnel2"]
    # also results in error but different from before
    with pytest.raises(ValueError) as e:
        # Attempt to read the dataframe again to check for the missing channel error
        bh = BufferHandler(
            FlashDataFrameCreator,
            config.dataframe,
            h5_paths,
            folder,
            auto=True,
            debug=True,
        )

    expected_error = e.value.args
    # Check for the specific error message indicating a missing channel in the configuration
    assert expected_error[3] == "Missing in config: {'gmdTunnel2'}"

    # Clean up created buffer files after the test
    [path.unlink() for path in bh.buffer_paths]


def test_create_buffer_files(config, h5_paths):
    folder_serial = create_parquet_dir(config, "create_buffer_files_serial")
    bh_serial = BufferHandler(
        FlashDataFrameCreator,
        config.dataframe,
        h5_paths,
        folder_serial,
        debug=True,
    )

    folder_parallel = create_parquet_dir(config, "create_buffer_files_parallel")
    bh_parallel = BufferHandler(FlashDataFrameCreator, config.dataframe, h5_paths, folder_parallel)

    df_serial = pd.read_parquet(folder_serial)
    df_parallel = pd.read_parquet(folder_parallel)

    pd.testing.assert_frame_equal(df_serial, df_parallel)

    # remove buffer files
    [path.unlink() for path in bh_serial.buffer_paths]
    [path.unlink() for path in bh_parallel.buffer_paths]


def test_get_filled_dataframe(config, h5_paths):
    """Test function to verify the creation of a filled dataframe from the buffer files."""
    folder = create_parquet_dir(config, "get_filled_dataframe")
    bh = BufferHandler(FlashDataFrameCreator, config.dataframe, h5_paths, folder)

    df = pd.read_parquet(folder)

    assert np.all(list(bh.dataframe_electron.columns) == list(df.columns) + ["dldSectorID"])

    channel_pulse = get_channels(
        config.dataframe.channels,
        formats=["per_pulse", "per_train"],
        index=True,
        extend_aux=True,
    )
    assert np.all(list(bh.dataframe_pulse.columns) == channel_pulse)
    # remove buffer files
    [path.unlink() for path in bh.buffer_paths]
