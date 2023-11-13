import os
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import pytest

from sed.core.config import parse_config
from sed.loader.flash.loader import FlashLoader

package_dir = os.path.dirname(find_spec("sed").origin)
config_path = os.path.join(package_dir, "../tests/data/loader/flash/config.yaml")


@pytest.fixture
def config_file():
    return parse_config(config_path)


def test_get_channels_by_format(config_file: dict):
    """
    Test function to verify the 'get_channels' method in FlashLoader class for
    retrieving channels based on formats and index inclusion.

    Args:
    config_file (dict): Configuration file or settings required for initializing the FlashLoader
    instance.

    Returns:
    None: This function performs assertions to validate the 'get_channels' method's functionality.
    """
    # Initialize the FlashLoader instance with the given config_file.
    fl = FlashLoader(config_file)

    # Define expected channels for each format.
    electron_channels = ["dldPosX", "dldPosY", "dldTimeSteps"]
    pulse_channels = [
        "sampleBias",
        "tofVoltage",
        "extractorVoltage",
        "extractorCurrent",
        "cryoTemperature",
        "sampleTemperature",
        "dldTimeBinSize",
        "gmdTunnel",
    ]
    train_channels = ["timeStamp", "delayStage"]
    index_channels = ["trainId", "pulseId", "electronId"]

    # Call get_channels method with different format options.

    # Request channels for 'per_electron' format using a list.
    format_electron = fl.get_channels(["per_electron"])

    # Request channels for 'per_pulse' format using a string.
    format_pulse = fl.get_channels("per_pulse")

    # Request channels for 'per_train' format using a list.
    format_train = fl.get_channels(["per_train"])

    # Request channels for 'all' formats using a list.
    format_all = fl.get_channels(["all"])

    # Request index channels only.
    format_index = fl.get_channels(index=True)

    # Request 'per_electron' format and include index channels.
    format_index_electron = fl.get_channels(["per_electron"], index=True)

    # Request 'all' formats and include index channels.
    format_all_index = fl.get_channels(["all"], index=True)

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


@pytest.mark.parametrize(
    "sub_dir",
    ["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"],
)
def test_initialize_paths(
    config_file: dict,
    fs,
    sub_dir: Literal["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"],
):
    """
    Test the initialization of paths based on the configuration and directory structures.

    Args:
    config_file (dict): The configuration file.
    fs: A fixture for a fake file system.
    sub_dir (Literal["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"]): Sub-directory.

    Returns:
    None
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

    # Instance of class with correct config and call initialize_paths
    fl = FlashLoader(config=config)
    data_raw_dir, data_parquet_dir = fl.initialize_paths()

    assert expected_raw_path == data_raw_dir[0]
    assert expected_processed_path == data_parquet_dir


def test_initialize_paths_filenotfound(config_file: dict):
    """
    Test FileNotFoundError during the initialization of paths.

    Args:
    config_file (dict): The configuration file.

    Returns:
    None
    """
    # Test the FileNotFoundError
    config = config_file
    del config["core"]["paths"]
    config["core"]["beamtime_id"] = "11111111"
    config["core"]["year"] = "2000"

    # Instance of class with correct config and call initialize_paths
    fl = FlashLoader(config=config)
    with pytest.raises(FileNotFoundError):
        _, _ = fl.initialize_paths()


def test_invalid_channel_format(config_file: dict):
    """
    Test ValueError for an invalid channel format.

    Args:
    config_file (dict): The configuration file.

    Returns:
    None
    """
    config = config_file
    config["dataframe"]["channels"]["dldPosX"]["format"] = "foo"

    fl = FlashLoader(config=config)

    with pytest.raises(ValueError):
        fl.read_dataframe()


def test_group_name_not_in_h5(config_file: dict):
    """
    Test ValueError when the group_name for a channel does not exist in the H5 file.

    Args:
    config_file (dict): The configuration file.

    Returns:
    None
    """
    config = config_file
    config["dataframe"]["channels"]["dldPosX"]["group_name"] = "foo"
    h5_path = "FLASH1_USER3_stream_2_run43878_file1_20230130T153807.1.h5"
    fl = FlashLoader(config=config)

    with pytest.raises(ValueError) as e:
        fl.create_dataframe_per_file(config["core"]["paths"]["data_raw_dir"] + h5_path)

    assert str(e.value.args[0]) == "The group_name for channel dldPosX does not exist."


def test_buffer_schema_mismatch(config_file: dict):
    """
    Test function to verify schema mismatch handling in the FlashLoader's 'read_dataframe' method.

    The test validates the error handling mechanism when the available channels do not match the
    schema of the existing parquet files.

    Args:
    config_file (dict): Configuration file required for initializing the FlashLoader instance.

    Returns:
    None: The function performs assertions to validate error handling for schema mismatch scenarios.

    Test Steps:
    - Attempt to read a dataframe after adding a new channel 'gmdTunnel2' to the configuration.
    - Check for an expected error related to the mismatch between available channels and schema.
    - Force recreation of dataframe with the added channel, ensuring successful dataframe creation.
    - Simulate a missing channel scenario by removing 'gmdTunnel2' from the configuration.
    - Check for an error indicating a missing channel in the configuration.
    - Clean up created buffer files after the test.
    """
    fl = FlashLoader(config=config_file)

    # Read a dataframe for a specific run
    fl.read_dataframe(runs=["43878"])

    # Manipulate the configuration to introduce a new channel 'gmdTunnel2'
    config = config_file
    config["dataframe"]["channels"]["gmdTunnel2"] = {
        "group_name": "/FL1/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/",
        "format": "per_pulse",
    }

    # Reread the dataframe with the modified configuration, expecting a schema mismatch error
    fl = FlashLoader(config=config)
    with pytest.raises(ValueError) as e:
        fl.read_dataframe(runs=["43878"])
    expected_error = e.value.args

    # Validate the specific error messages for schema mismatch
    assert "The available channels do not match the schema of file" in expected_error[0]
    assert expected_error[2] == "Missing in parquet: {'gmdTunnel2'}"
    assert expected_error[4] == "Please check the configuration file or set force_recreate to True."

    # Force recreation of the dataframe, including the added channel 'gmdTunnel2'
    fl.read_dataframe(runs=["43878"], force_recreate=True)

    # Remove 'gmdTunnel2' from the configuration to simulate a missing channel scenario
    del config["dataframe"]["channels"]["gmdTunnel2"]
    fl = FlashLoader(config=config)
    with pytest.raises(ValueError) as e:
        # Attempt to read the dataframe again to check for the missing channel error
        fl.read_dataframe(runs=["43878"])

    expected_error = e.value.args
    # Check for the specific error message indicating a missing channel in the configuration
    assert expected_error[3] == "Missing in config: {'gmdTunnel2'}"

    # Clean up created buffer files after the test
    _, parquet_data_dir = fl.initialize_paths()
    for file in os.listdir(Path(parquet_data_dir, "buffer")):
        os.remove(Path(parquet_data_dir, "buffer", file))
