"""Tests for SXPLoader functionality"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from sed.core.config import parse_config
from sed.loader.sxp.loader import SXPLoader

test_dir = os.path.join(os.path.dirname(__file__), "../..")
config_path = os.path.join(test_dir, "data/loader/sxp/config.yaml")
H5_PATH = "RAW-R0016-DA03-S00000.h5"


@pytest.fixture(name="config_file")
def fixture_config_file() -> dict:
    """Fixture providing a configuration file for SXPLoader tests.

    Returns:
        dict: The parsed configuration file.
    """
    return parse_config(config=config_path, folder_config={}, user_config={}, system_config={})


def test_get_channels_by_format(config_file: dict) -> None:
    """
    Test function to verify the 'get_channels' method in SXPLoader class for
    retrieving channels based on formats and index inclusion.
    """
    # Initialize the SXPLoader instance with the given config_file.
    sl = SXPLoader(config_file)

    # Define expected channels for each format.
    electron_channels = ["dldPosX", "dldPosY", "dldTimeSteps"]
    pulse_channels: list[str] = []
    train_channels = ["timeStamp", "delayStage"]
    index_channels = ["trainId", "pulseId", "electronId"]

    # Call get_channels method with different format options.

    # Request channels for 'per_electron' format using a list.
    format_electron = sl.get_channels(["per_electron"])

    # Request channels for 'per_pulse' format using a string.
    format_pulse = sl.get_channels("per_pulse")

    # Request channels for 'per_train' format using a list.
    format_train = sl.get_channels(["per_train"])

    # Request channels for 'all' formats using a list.
    format_all = sl.get_channels(["all"])

    # Request index channels only.
    format_index = sl.get_channels(index=True)

    # Request 'per_electron' format and include index channels.
    format_index_electron = sl.get_channels(["per_electron"], index=True)

    # Request 'all' formats and include index channels.
    format_all_index = sl.get_channels(["all"], index=True)

    # Assert that the obtained channels match the expected channels.
    assert set(electron_channels) == set(format_electron)
    assert set(pulse_channels) == set(format_pulse)
    assert set(train_channels) == set(format_train)
    assert set(electron_channels + pulse_channels + train_channels) == set(format_all)
    assert set(index_channels) == set(format_index)
    assert set(index_channels + electron_channels) == set(format_index_electron)
    assert set(index_channels + electron_channels + pulse_channels + train_channels) == set(
        format_all_index,
    )


def test_initialize_dirs(config_file: dict, fs) -> None:
    """
    Test the initialization of paths based on the configuration and directory structures.

    Args:
    fs: A fixture for a fake file system.
    """
    config = config_file
    del config["core"]["paths"]
    config["core"]["beamtime_id"] = "12345678"
    config["core"]["year"] = "2000"

    # Find base path of beamline from config.
    base_path = config["core"]["beamtime_dir"]["sxp"]
    expected_path = Path(base_path) / config["core"]["year"] / config["core"]["beamtime_id"]
    # Create expected paths
    expected_raw_path = expected_path / "raw"
    expected_processed_path = expected_path / "processed" / "parquet"

    # Create a fake file system for testing
    fs.create_dir(expected_raw_path)
    fs.create_dir(expected_processed_path)

    # Instance of class with correct config and call initialize_dirs
    sl = SXPLoader(config=config)
    sl._initialize_dirs()

    assert expected_raw_path == sl.raw_dir[0]
    assert expected_processed_path == sl.processed_dir


def test_initialize_dirs_filenotfound(config_file: dict):
    """
    Test FileNotFoundError during the initialization of paths.
    """
    # Test the FileNotFoundError
    config = config_file
    del config["core"]["paths"]
    config["core"]["beamtime_id"] = "11111111"
    config["core"]["year"] = "2000"

    # Instance of class with correct config and call initialize_dirs
    sl = SXPLoader(config=config)
    with pytest.raises(FileNotFoundError):
        sl._initialize_dirs()


def test_invalid_channel_format(config_file: dict):
    """
    Test ValueError for an invalid channel format.
    """
    config = config_file
    config["dataframe"]["channels"]["dldPosX"]["format"] = "foo"

    sl = SXPLoader(config=config)

    with pytest.raises(ValueError):
        sl.read_dataframe()


@pytest.mark.parametrize(
    "key_type",
    ["dataset_key", "index_key"],
)
def test_data_keys_not_in_h5(config_file: dict, key_type: str):
    """Test ValueError when the dataset_key or index_key for a channel does not exist in the H5
    file.

    Args:
        key_type (str): Key type to check
    """
    config = config_file
    config["dataframe"]["channels"]["dldPosX"][key_type] = "foo"
    sl = SXPLoader(config=config)

    with pytest.raises(ValueError) as e:
        sl.create_dataframe_per_file(Path(config["core"]["paths"]["raw"], H5_PATH))

    assert str(e.value.args[0]) == f"The {key_type} for channel dldPosX does not exist."


def test_buffer_schema_mismatch(config_file: dict):
    """
    Test function to verify schema mismatch handling in the SXPLoader's 'read_dataframe' method.

    The test validates the error handling mechanism when the available channels do not match the
    schema of the existing parquet files.

    Test Steps:
    - Attempt to read a dataframe after adding a new channel 'delayStage2' to the configuration.
    - Check for an expected error related to the mismatch between available channels and schema.
    - Force recreation of dataframe with the added channel, ensuring successful dataframe creation.
    - Simulate a missing channel scenario by removing 'delayStage2' from the configuration.
    - Check for an error indicating a missing channel in the configuration.
    - Clean up created buffer files after the test.
    """
    sl = SXPLoader(config=config_file)

    # Read a dataframe for a specific run
    sl.read_dataframe(runs=["0016"])

    # Manipulate the configuration to introduce a new channel 'delayStage2'
    config = config_file
    config["dataframe"]["channels"]["delayStage2"] = {
        "format": "per_train",
        "dataset_key": "/CONTROL/SCS_ILH_LAS/MDL/OPTICALDELAY_PP800/actualPosition/value",
        "index_key": "/INDEX/trainId",
    }

    # Reread the dataframe with the modified configuration, expecting a schema mismatch error
    sl = SXPLoader(config=config)
    with pytest.raises(ValueError) as e:
        sl.read_dataframe(runs=["0016"])
    expected_error = e.value.args

    # Validate the specific error messages for schema mismatch
    assert "The available channels do not match the schema of file" in expected_error[0]
    assert expected_error[2] == "Missing in parquet: {'delayStage2'}"
    assert expected_error[4] == "Please check the configuration file or set force_recreate to True."

    # Force recreation of the dataframe, including the added channel 'delayStage2'
    sl.read_dataframe(runs=["0016"], force_recreate=True)

    # Remove 'delayStage2' from the configuration to simulate a missing channel scenario
    del config["dataframe"]["channels"]["delayStage2"]
    sl = SXPLoader(config=config)
    with pytest.raises(ValueError) as e:
        # Attempt to read the dataframe again to check for the missing channel error
        sl.read_dataframe(runs=["0016"])

    expected_error = e.value.args
    # Check for the specific error message indicating a missing channel in the configuration
    assert expected_error[3] == "Missing in config: {'delayStage2'}"

    # Clean up created buffer files after the test
    sl._initialize_dirs()
    for file in os.listdir(Path(sl.processed_dir, "buffer")):
        os.remove(Path(sl.processed_dir, "buffer", file))
