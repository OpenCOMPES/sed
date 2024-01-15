# pylint: disable=duplicate-code
"""Tests for SXPLoader functionality"""
from pathlib import Path

from sed.loader.sxp.loader import SXPLoader


def test_initialize_paths(config_raw: dict, fs) -> None:
    """
    Test the initialization of paths based on the configuration and directory structures.

    Args:
    fs: A fixture for a fake file system.
    """
    config = config_raw
    del config["core"]["paths"]
    config["core"]["beamtime_id"] = "12345678"
    config["core"]["year"] = "2000"

    # Find base path of beamline from config.
    base_path = config["dataframe"]["beamtime_dir"]["sxp"]
    expected_path = Path(base_path) / config["core"]["year"] / config["core"]["beamtime_id"]
    # Create expected paths
    expected_raw_path = expected_path / "raw"
    expected_processed_path = expected_path / "processed" / "parquet"

    # Create a fake file system for testing
    fs.create_dir(expected_raw_path)
    fs.create_dir(expected_processed_path)

    # Instance of class with correct config and call initialize_paths
    sl = SXPLoader(config=config)
    data_raw_dir = sl._config.core.paths.data_raw_dir
    data_parquet_dir = sl._config.core.paths.data_parquet_dir

    assert expected_raw_path == data_raw_dir
    assert expected_processed_path == data_parquet_dir
