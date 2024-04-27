from pathlib import Path

import pytest

from sed.loader.flash.utils import initialize_parquet_paths


def create_parquet_dir(config, folder):
    parquet_path = Path(config["core"]["paths"]["data_parquet_dir"])
    parquet_path = parquet_path.joinpath(folder)
    parquet_path.mkdir(parents=True, exist_ok=True)
    return parquet_path


def test_parquet_init_error():
    """Test ParquetHandler initialization error"""
    with pytest.raises(ValueError) as e:
        _ = initialize_parquet_paths(parquet_names="test")

    assert "Please provide folder or parquet_paths." in str(e.value)

    with pytest.raises(ValueError) as e:
        _ = initialize_parquet_paths(folder="test")

    assert "With folder, please provide parquet_names." in str(e.value)


def test_initialize_paths(config):
    """Test ParquetHandler initialization"""
    folder = create_parquet_dir(config, "parquet_init")

    ph = initialize_parquet_paths("test", folder, extension="xyz")
    assert ph[0].suffix == ".xyz"
    assert ph[0].name == "test.xyz"

    # test prefix and suffix
    ph = initialize_parquet_paths("test", folder, prefix="prefix_", suffix="_suffix")
    assert ph[0].name == "prefix_test_suffix.parquet"

    # test with list of parquet_names and subfolder
    ph = initialize_parquet_paths(["test1", "test2"], folder, subfolder="subfolder")
    assert ph[0].parent.name == "subfolder"
    assert ph[0].name == "test1.parquet"
    assert ph[1].name == "test2.parquet"
