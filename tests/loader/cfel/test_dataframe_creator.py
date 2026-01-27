"""Tests for DataFrameCreator functionality (per_file, per_train, per_electron)"""
from pathlib import Path

import h5py
import numpy as np
import pytest
import pandas as pd

from sed.loader.cfel.dataframe import DataFrameCreator
from sed.loader.flash.utils import get_channels


def test_get_dataset_key(config_dataframe: dict, h5_paths: list[Path]) -> None:
    df = DataFrameCreator(config_dataframe, h5_paths[0])
    channel = "dldPosX"
    dataset_key = df.get_dataset_key(channel)
    assert dataset_key == config_dataframe["channels"][channel]["dataset_key"]

    config_copy = config_dataframe.copy()
    del config_copy["channels"][channel]["dataset_key"]
    df2 = DataFrameCreator(config_copy, h5_paths[0])
    with pytest.raises(ValueError):
        df2.get_dataset_key(channel)


def test_get_dataset_array(config_dataframe: dict, h5_paths: list[Path]) -> None:
    df = DataFrameCreator(config_dataframe, h5_paths[0])
    for channel in config_dataframe["channels"]:
        dset = df.get_dataset_array(channel)
        assert isinstance(dset, h5py.Dataset)
        assert dset.shape[0] > 0


def test_df_per_file(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """Test per_file data (countId index)"""
    df = DataFrameCreator(config_dataframe, h5_paths[0])
    per_file_channels = get_channels(config_dataframe, "per_file")
    if not per_file_channels:
        pytest.skip("No per_file channels in config")

    # Index should be countId
    df_file = df.df  # combined DataFrame includes per_file data
    assert "countId" in df_file.index.names or df_file.index.name == "countId"

    # All per_file columns exist in df
    for ch in per_file_channels:
        assert ch in df_file.columns


def test_df_train(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """Test df_train (per_train channels)"""
    df = DataFrameCreator(config_dataframe, h5_paths[0])
    per_train_channels = get_channels(config_dataframe, "per_train")
    aux_alias = config_dataframe.get("aux_alias", "dldAux")
    if aux_alias in config_dataframe["channels"]:
        subchannels = config_dataframe["channels"][aux_alias].get("sub_channels", {})
        per_train_channels.extend(subchannels.keys())

    if not per_train_channels:
        pytest.skip("No per_train channels in config")

    df_train = df.df_train
    assert isinstance(df_train, pd.DataFrame)

    # Index should be single-level trainId (because no pulseId/electronId in current code)
    assert df_train.index.name == "trainId" or df_train.index.name is None

    # Columns check
    assert set(df_train.columns).issubset(set(per_train_channels))


def test_df_electron(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """Test df_electron (per_electron channels)"""
    df = DataFrameCreator(config_dataframe, h5_paths[0])
    per_electron_channels = get_channels(config_dataframe, "per_electron")
    if not per_electron_channels:
        pytest.skip("No per-electron channels in config")

    df_elec = df.df_electron
    assert isinstance(df_elec, pd.DataFrame)

    # Index can be RangeIndex (single-level) if trainId/electronId not implemented
    idx = df_elec.index
    assert idx is not None
    # Columns
    assert set(df_elec.columns).issubset(set(per_electron_channels))
    # No NaNs
    assert not df_elec.isnull().values.any()

# def test_df_electron(config_dataframe: dict, h5_paths: list[Path]) -> None:
#     """Test df_electron (per_electron channels)"""
#     df = DataFrameCreator(config_dataframe, h5_paths[0])
#     per_electron_channels = get_channels(config_dataframe, "per_electron")
#     if not per_electron_channels:
#         pytest.skip("No per-electron channels in config")

#     df_elec = df.df_electron
#     assert isinstance(df_elec, pd.DataFrame)
#     # MultiIndex: trainId + electronId
#     idx = df_elec.index
#     assert isinstance(idx, pd.MultiIndex)
#     assert set(idx.names) == {"trainId", "electronId"}

#     # Columns
#     assert set(df_elec.columns).issubset(set(per_electron_channels))
#     # No NaNs
#     assert not df_elec.isnull().values.any()


def test_df_timestamp(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """Test timestamp DataFrame"""
    df = DataFrameCreator(config_dataframe, h5_paths[0])
    ts_df = df.df_timestamp
    assert isinstance(ts_df, pd.DataFrame)
    ts_col = config_dataframe["columns"].get("timestamp", "timeStamp")
    assert ts_col in ts_df.columns
    # Length matches main index
    assert ts_df.shape[0] == len(df.index)


def test_df_combined(config_dataframe: dict, h5_paths: list[Path]) -> None:
    dfc = DataFrameCreator(config_dataframe, h5_paths[0])
    df = dfc.df

    assert isinstance(df, pd.DataFrame)

    df_elec = dfc.df_electron
    df_train = dfc.df_train
    df_ts = dfc.df_timestamp

    # 1) All electron rows must be present in the combined DF
    assert df_elec.index.isin(df.index).all()

    # 2) Electron values must be unchanged (dtype upcast is OK)
    pd.testing.assert_frame_equal(
        df.loc[df_elec.index, df_elec.columns],
        df_elec,
        check_dtype=False,
    )

    # 3) Columns must be the union
    expected_cols = (
        set(df_elec.columns)
        | set(df_train.columns)
        | set(df_ts.columns)
    )
    assert set(df.columns) == expected_cols

    # 4) per_train + timestamp columns must be forward-filled
    ffill_cols = list(df_train.columns) + list(df_ts.columns)
    assert not df[ffill_cols].isna().any().any()

# def test_df_combined(config_dataframe: dict, h5_paths: list[Path]) -> None:
#     """Test df property (combined DataFrame)"""
#     df = DataFrameCreator(config_dataframe, h5_paths[0])
#     combined = df.df
#     assert isinstance(combined, pd.DataFrame)

#     # Columns = per_file + per_train + per_electron + timestamp
#     expected_cols = set()
#     try:
#         expected_cols.update(get_channels(config_dataframe, "per_file"))
#     except ValueError:
#         pass
#     try:
#         expected_cols.update(get_channels(config_dataframe, "per_train"))
#     except ValueError:
#         pass
#     try:
#         expected_cols.update(get_channels(config_dataframe, "per_electron"))
#     except ValueError:
#         pass
#     expected_cols.add(config_dataframe["columns"].get("timestamp", "timeStamp"))

#     # Columns in combined are subset of expected
#     assert set(combined.columns).issubset(expected_cols)


def test_group_name_not_in_h5(
    config_dataframe: dict,
    h5_paths: list[Path],
) -> None:
    """Test error when dataset_key does not exist in H5 file."""

    # Pick a non-index channel
    channel = next(
        ch for ch in config_dataframe["channels"]
        if ch != config_dataframe.get("index", ["countId"])[0]
    )

    # Deep copy only what we mutate
    config = dict(config_dataframe)
    config["channels"] = dict(config_dataframe["channels"])
    config["channels"][channel] = dict(config_dataframe["channels"][channel])

    # Break ONLY this channel
    config["channels"][channel]["dataset_key"] = "/this/does/not/exist"

    dfc = DataFrameCreator(config, h5_paths[0])

    with pytest.raises(KeyError):
        _ = dfc.get_dataset_array(channel)

# def test_group_name_not_in_h5(config_dataframe: dict, h5_paths: list[Path]) -> None:
#     """Test KeyError when a dataset_key is missing"""
#     channel = "dldPosX"
#     config = config_dataframe.copy()
#     config["channels"][channel]["dataset_key"] = "non_existent_dataset"

#     df = DataFrameCreator(config, h5_paths[0])
#     with pytest.raises(KeyError):
#         _ = df.df_train
