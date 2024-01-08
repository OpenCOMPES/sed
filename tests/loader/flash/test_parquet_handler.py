from pathlib import Path

import dask.dataframe as ddf
import pandas as pd
import pytest

from sed.loader.fel import BufferHandler
from sed.loader.fel import ParquetHandler


def create_parquet_dir(config, folder):
    parquet_path = Path(config["core"]["paths"]["data_parquet_dir"])
    parquet_path = parquet_path.joinpath(folder)
    parquet_path.mkdir(parents=True, exist_ok=True)
    return parquet_path


def test_parquet_init_error():
    """Test ParquetHandler initialization error"""
    with pytest.raises(ValueError) as e:
        ParquetHandler(parquet_names="test")

    assert "Please provide folder or parquet_paths." in str(e.value)

    with pytest.raises(ValueError) as e:
        ParquetHandler(folder="test")

    assert "With folder, please provide parquet_names." in str(e.value)


def test_initialize_paths(config):
    """Test ParquetHandler initialization"""
    folder = create_parquet_dir(config, "parquet_init")

    ph = ParquetHandler("test", folder, extension="xyz")
    assert ph.parquet_paths[0].suffix == ".xyz"
    assert ph.parquet_paths[0].name == "test.xyz"

    # test prefix and suffix
    ph = ParquetHandler("test", folder, prefix="prefix_", suffix="_suffix")
    assert ph.parquet_paths[0].name == "prefix_test_suffix.parquet"

    # test with list of parquet_names and subfolder
    ph = ParquetHandler(["test1", "test2"], folder, subfolder="subfolder")
    assert ph.parquet_paths[0].parent.name == "subfolder"
    assert ph.parquet_paths[0].name == "test1.parquet"
    assert ph.parquet_paths[1].name == "test2.parquet"


def test_sav_read_parquet(config, h5_paths):
    """Test ParquetHandler save and read parquet"""
    # provide instead parquet_paths
    folder = create_parquet_dir(config, "parquet_save")
    parquet_path = folder.joinpath("test.parquet")

    ph = ParquetHandler(parquet_paths=parquet_path)
    print(ph.parquet_paths)
    bh = BufferHandler(config["dataframe"], h5_paths, folder)
    ph.save_parquet(bh.dataframe_electron, drop_index=True)
    ph.save_parquet(bh.dataframe_electron, drop_index=False)

    df = ph.read_parquet()
    # check if parquet is read correctly
    assert isinstance(df, ddf.DataFrame)
    [path.unlink() for path in bh.buffer_paths]
    parquet_path.unlink()
    # Test file not found
    with pytest.raises(FileNotFoundError) as e:
        ph.read_parquet()
