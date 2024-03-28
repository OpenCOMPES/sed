from pathlib import Path

import numpy as np
import pytest

from sed.loader.flash.loader import BufferHandler
from sed.loader.flash.loader import ParquetHandler


def create_parquet_dir(config, folder):
    parquet_path = Path(config["core"]["paths"]["data_parquet_dir"])
    parquet_path = parquet_path.joinpath(folder)
    parquet_path.mkdir(parents=True, exist_ok=True)
    return parquet_path


def test_parquet_init_error():
    """Test ParquetHandler initialization error"""
    with pytest.raises(ValueError) as e:
        ph = ParquetHandler(parquet_names="test")
        ph.initialize_paths()

    assert "Please provide folder or parquet_paths." in str(e.value)

    with pytest.raises(ValueError) as e:
        ph = ParquetHandler(folder="test")
        ph.initialize_paths()

    assert "With folder, please provide parquet_names." in str(e.value)


def test_initialize_paths(config):
    """Test ParquetHandler initialization"""
    folder = create_parquet_dir(config, "parquet_init")

    ph = ParquetHandler("test", folder, extension="xyz")
    ph.initialize_paths()
    assert ph.parquet_paths[0].suffix == ".xyz"
    assert ph.parquet_paths[0].name == "test.xyz"

    # test prefix and suffix
    ph = ParquetHandler("test", folder, prefix="prefix_", suffix="_suffix")
    ph.initialize_paths()
    assert ph.parquet_paths[0].name == "prefix_test_suffix.parquet"

    # test with list of parquet_names and subfolder
    ph = ParquetHandler(["test1", "test2"], folder, subfolder="subfolder")
    ph.initialize_paths()
    assert ph.parquet_paths[0].parent.name == "subfolder"
    assert ph.parquet_paths[0].name == "test1.parquet"
    assert ph.parquet_paths[1].name == "test2.parquet"


def test_save_read_parquet(config, h5_paths):
    """Test ParquetHandler save and read parquet"""
    # provide instead parquet_paths
    folder = create_parquet_dir(config, "parquet_save")
    parquet_path = folder.joinpath("test.parquet")

    # create some parquet files for testing save and read
    ph = ParquetHandler()
    bh = BufferHandler(config["dataframe"])
    bh.run(h5_paths=h5_paths, folder=folder)

    ph.save_parquet([bh.dataframe_electron], parquet_paths=[parquet_path], drop_index=False)
    parquet_path.unlink()  # remove parquet file
    ph.save_parquet([bh.dataframe_electron], parquet_paths=[parquet_path], drop_index=True)

    # Provide different number of DataFrames and paths
    with pytest.raises(ValueError) as e:
        ph.save_parquet(
            [bh.dataframe_electron],
            parquet_paths=[parquet_path, parquet_path],
            drop_index=False,
        )

    assert "Number of DataFrames provided does not match the number of paths." in str(e.value)

    df = ph.read_parquet()
    df_loaded = df[0].compute()
    df_saved = bh.dataframe_electron.compute().reset_index(drop=True)

    # compare the saved and loaded dataframes
    assert np.all(df[0].columns == bh.dataframe_electron.columns)
    assert df_loaded.equals(df_saved)

    [path.unlink() for path in bh.buffer_paths]  # remove buffer files
    parquet_path.unlink()  # remove parquet file
    # Test file not found
    with pytest.raises(FileNotFoundError) as e:
        ph.read_parquet()
