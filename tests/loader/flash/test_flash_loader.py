"""Tests for FlashLoader functionality"""
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest
from pydantic_core import ValidationError

from sed.loader.flash.loader import FlashLoader

package_dir = os.path.dirname(find_spec("sed").origin)
config_path = os.path.join(package_dir, "../tests/data/loader/flash/config.yaml")
H5_PATH = "FLASH1_USER3_stream_2_run43878_file1_20230130T153807.1.h5"


@pytest.mark.parametrize(
    "sub_dir",
    ["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"],
)
def test_initialize_paths(
    config_raw,
    config,
    fs,
    sub_dir: Literal["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"],
) -> None:
    """
    Test the initialization of paths based on the configuration and directory structures.

    Args:
    fs: A fixture for a fake file system.
    sub_dir (Literal["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"]): Sub-directory.
    """
    config_dict = config_raw
    del config_dict["core"]["paths"]
    config_dict["core"]["beamtime_id"] = "12345678"
    config_dict["core"]["year"] = "2000"

    # Find base path of beamline from config. Here, we use pg2
    base_path = config_dict["dataframe"]["beamtime_dir"]["pg2"]
    expected_path = (
        Path(base_path) / config_dict["core"]["year"] / "data" / config_dict["core"]["beamtime_id"]
    )
    # Create expected paths
    expected_raw_path = expected_path / "raw" / "hdf" / sub_dir
    expected_processed_path = expected_path / "processed" / "parquet"

    # Create a fake file system for testing
    fs.create_dir(expected_raw_path)
    fs.create_dir(expected_processed_path)

    # Instance of class with correct config and call initialize_paths
    # config_alt = FlashLoaderConfig(**config_dict)
    fl = FlashLoader(config=config_dict)
    data_raw_dir = fl._config.core.paths.data_raw_dir
    data_parquet_dir = fl._config.core.paths.data_parquet_dir

    assert expected_raw_path == data_raw_dir
    assert expected_processed_path == data_parquet_dir

    # remove breamtimeid, year and daq from config to raise error
    del config_dict["core"]["beamtime_id"]
    with pytest.raises(ValidationError) as e:
        fl = FlashLoader(config=config_dict)

    error_messages = [error["msg"] for error in e.value.errors()]
    assert (
        "Value error, Either 'paths' or 'beamtime_id' and 'year' must be provided."
        in error_messages
    )


# def test_initialize_paths_filenotfound(config) -> None:
#     """
#     Test FileNotFoundError during the initialization of paths.
#     """
#     config_alt = config
#     # Test the FileNotFoundError
#     del config_alt.core.paths
#     config_alt.core.beamtime_id = "11111111"
#     config_alt.core.year = "2000"

#     # Instance of class with correct config and call initialize_paths
#     with pytest.raises(FileNotFoundError):
#         fl = FlashLoader(config=config_alt)


def test_save_read_parquet_flash(config):
    """Test ParquetHandler save and read parquet"""
    config_alt = config
    config_alt.core.paths.data_parquet_dir = config_alt.core.paths.data_parquet_dir.joinpath(
        "_flash_save_read/",
    )
    fl = FlashLoader(config=config_alt)
    df1, _, _ = fl.read_dataframe(runs=[43878, 43879], save_parquet=True)

    df2, _, _ = fl.read_dataframe(runs=[43878, 43879], load_parquet=True)

    # check if parquet read is same as parquet saved read correctly
    pd.testing.assert_frame_equal(df1.compute().reset_index(drop=True), df2.compute())
