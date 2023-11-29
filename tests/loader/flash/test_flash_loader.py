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
def test_initialize_paths(
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

    # Instance of class with correct config and call initialize_paths
    fl = FlashLoader(config=config)
    data_raw_dir, data_parquet_dir = fl.initialize_paths()

    assert expected_raw_path == data_raw_dir[0]
    assert expected_processed_path == data_parquet_dir


def test_initialize_paths_filenotfound(config_file: dict) -> None:
    """
    Test FileNotFoundError during the initialization of paths.
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


# def test_buffer_schema_mismatch(config_file):
#     """
#     Test function to verify schema mismatch handling in the FlashLoader's 'read_dataframe' method.

#     The test validates the error handling mechanism when the available channels do not match the
#     schema of the existing parquet files.

#     Test Steps:
#     - Attempt to read a dataframe after adding a new channel 'gmdTunnel2' to the configuration.
#     - Check for an expected error related to the mismatch between available channels and schema.
#     - Force recreation of dataframe with the added channel, ensuring successful dataframe
#       creation.
#     - Simulate a missing channel scenario by removing 'gmdTunnel2' from the configuration.
#     - Check for an error indicating a missing channel in the configuration.
#     - Clean up created buffer files after the test.
#     """
#     fl = FlashLoader(config=config_file)

#     # Read a dataframe for a specific run
#     fl.read_dataframe(runs=["43878"])

#     # Manipulate the configuration to introduce a new channel 'gmdTunnel2'
#     config = config_file
#     config["dataframe"]["channels"]["gmdTunnel2"] = {
#         "group_name": "/FL1/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/",
#         "format": "per_pulse",
#     }

#     # Reread the dataframe with the modified configuration, expecting a schema mismatch error
#     fl = FlashLoader(config=config)
#     with pytest.raises(ValueError) as e:
#         fl.read_dataframe(runs=["43878"])
#     expected_error = e.value.args

#     # Validate the specific error messages for schema mismatch
#     assert "The available channels do not match the schema of file" in expected_error[0]
#     assert expected_error[2] == "Missing in parquet: {'gmdTunnel2'}"
#     assert expected_error[4] == "Please check the configuration file or set force_recreate to
#       True."

#     # Force recreation of the dataframe, including the added channel 'gmdTunnel2'
#     fl.read_dataframe(runs=["43878"], force_recreate=True)

#     # Remove 'gmdTunnel2' from the configuration to simulate a missing channel scenario
#     del config["dataframe"]["channels"]["gmdTunnel2"]
#     fl = FlashLoader(config=config)
#     with pytest.raises(ValueError) as e:
#         # Attempt to read the dataframe again to check for the missing channel error
#         fl.read_dataframe(runs=["43878"])

#     expected_error = e.value.args
#     # Check for the specific error message indicating a missing channel in the configuration
#     assert expected_error[3] == "Missing in config: {'gmdTunnel2'}"

#     # Clean up created buffer files after the test
#     _, parquet_data_dir = fl.initialize_paths()
#     for file in os.listdir(Path(parquet_data_dir, "buffer")):
#         os.remove(Path(parquet_data_dir, "buffer", file))
