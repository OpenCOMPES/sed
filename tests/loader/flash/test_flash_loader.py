"""Tests for FlashLoader functionality"""
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import pandas as pd
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

    # remove breamtimeid, year and daq from config to raise error
    del config["core"]["beamtime_id"]
    fl = FlashLoader(config=config)
    with pytest.raises(ValueError) as e:
        _, _ = fl.initialize_paths()

    assert "The beamtime_id, year and daq are required." in str(e.value)


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


def test_save_read_parquet_flash(config):
    """Test ParquetHandler save and read parquet"""
    fl = FlashLoader(config=config)
    df1, df_timed1, _ = fl.read_dataframe(runs=[43878, 43879], save_parquet=True)

    df2, df_timed2, _ = fl.read_dataframe(runs=[43878, 43879], load_parquet=True)

    # check if parquet read is same as parquet saved read correctly
    pd.testing.assert_frame_equal(df1.compute().reset_index(drop=True), df2.compute())
