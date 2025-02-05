"""Tests for DataFrameCreator functionality"""
from pathlib import Path

import h5py
import numpy as np
import pytest
from pandas import DataFrame
from pandas import Index
from pandas import MultiIndex

from sed.loader.flash.dataframe import DataFrameCreator
from sed.loader.flash.utils import get_channels


def test_get_index_dataset_key(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """Test the creation of the index and dataset keys for a given channel."""
    config = config_dataframe
    channel = "dldPosX"
    df = DataFrameCreator(config, h5_paths[0])
    index_key, dataset_key = df.get_index_dataset_key(channel)
    assert index_key == config["channels"][channel]["index_key"]
    assert dataset_key == config["channels"][channel]["dataset_key"]

    # remove index_key
    del config["channels"][channel]["index_key"]
    with pytest.raises(ValueError):
        df.get_index_dataset_key(channel)


def test_get_dataset_array(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """Test the creation of a h5py dataset for a given channel."""

    df = DataFrameCreator(config_dataframe, h5_paths[0])
    channel = "dldPosX"

    train_id, dset = df.get_dataset_array(channel, slice_=False)
    # Check that the train_id and np_array have the correct shapes and types
    assert isinstance(train_id, Index)
    assert isinstance(dset, h5py.Dataset)
    assert train_id.name == "trainId"
    assert train_id.shape[0] == dset.shape[0]
    assert dset.shape[1] == 5
    assert dset.shape[2] == 321

    train_id, dset = df.get_dataset_array(channel, slice_=True)
    assert train_id.shape[0] == dset.shape[0]
    assert dset.shape[1] == 321

    channel = "gmdTunnel"
    train_id, dset = df.get_dataset_array(channel, True)
    assert train_id.shape[0] == dset.shape[0]
    assert dset.shape[1] == 500


def test_empty_get_dataset_array(
    config_dataframe: dict,
    h5_paths: list[Path],
    h5_file_copy: h5py.File,
) -> None:
    """Test the method when given an empty dataset."""

    channel = "gmdTunnel"
    df = DataFrameCreator(config_dataframe, h5_paths[0])
    train_id, dset = df.get_dataset_array(channel, slice_=False)

    channel_index_key = "/FL1/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/index"
    # channel_dataset_key = config_dataframe["channels"][channel]["group_name"] + "value"
    empty_dataset_key = "/FL1/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/empty"
    config_dataframe["channels"][channel]["index_key"] = channel_index_key
    config_dataframe["channels"][channel]["dataset_key"] = empty_dataset_key

    # create an empty dataset
    h5_file_copy.create_dataset(
        name=empty_dataset_key,
        shape=(train_id.shape[0], 0),
    )

    df = DataFrameCreator(config_dataframe, h5_paths[0])
    df.h5_file = h5_file_copy
    train_id, dset_empty = df.get_dataset_array(channel, slice_=False)

    assert dset_empty.shape[0] == train_id.shape[0]
    assert dset.shape[1] == 8
    assert dset_empty.shape[1] == 0


def test_pulse_index(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """Test the creation of the pulse index for electron resolved data"""

    df = DataFrameCreator(config_dataframe, h5_paths[0])
    pulse_index, pulse_array = df.get_dataset_array("pulseId", slice_=True)
    index, indexer = df.pulse_index(config_dataframe["ubid_offset"])
    # Check if the index_per_electron is a MultiIndex and has the correct levels
    assert isinstance(index, MultiIndex)
    assert set(index.names) == {"trainId", "pulseId", "electronId"}

    # Check if the pulse_index has the correct number of elements
    # This should be the pulses without nan values
    pulse_rav = pulse_array.ravel()
    pulse_no_nan = pulse_rav[~np.isnan(pulse_rav)]
    assert len(index) == len(pulse_no_nan)

    # Check if all pulseIds are correctly mapped to the index
    assert np.all(
        index.get_level_values("pulseId").values
        == (pulse_no_nan - config_dataframe["ubid_offset"])[indexer],
    )

    assert np.all(
        index.get_level_values("electronId").values[:5] == [0, 1, 0, 1, 0],
    )

    assert np.all(
        index.get_level_values("electronId").values[-5:] == [1, 0, 1, 0, 1],
    )

    # check if all indexes are unique and monotonic increasing
    assert index.is_unique
    assert index.is_monotonic_increasing


def test_df_electron(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """Test the creation of a pandas DataFrame for a channel of type [per electron]."""
    df = DataFrameCreator(config_dataframe, h5_paths[0])

    result_df = df.df_electron

    # check index levels
    assert set(result_df.index.names) == {"trainId", "pulseId", "electronId"}

    # check that there are no nan values in the dataframe
    assert ~result_df.isnull().values.any()

    # Check if first 5 values are as expected
    # e.g. that the values are dropped for pulseId index below 0 (ubid_offset)
    # however in this data the lowest value is 9 and offset was 5 so no values are dropped
    assert np.all(
        result_df.values[:5]
        == np.array(
            [
                [556.0, 731.0, 42888.0],
                [549.0, 737.0, 42881.0],
                [671.0, 577.0, 39181.0],
                [671.0, 579.0, 39196.0],
                [714.0, 859.0, 37530.0],
            ],
            dtype=np.float32,
        ),
    )
    assert np.all(result_df.index.get_level_values("pulseId") >= 0)
    assert isinstance(result_df, DataFrame)

    assert result_df.index.is_unique

    # check that dataframe contains all subchannels
    assert np.all(
        set(result_df.columns) == set(get_channels(config_dataframe, ["per_electron"])),
    )


def test_create_dataframe_per_pulse(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """Test the creation of a pandas DataFrame for a channel of type [per pulse]."""
    df = DataFrameCreator(config_dataframe, h5_paths[0])
    result_df = df.df_pulse
    # Check that the result_df is a DataFrame and has the correct shape
    assert isinstance(result_df, DataFrame)

    _, data = df.get_dataset_array("gmdTunnel", slice_=True)
    assert result_df.shape[0] == data.shape[0] * data.shape[1]

    # check index levels
    assert set(result_df.index.names) == {"trainId", "pulseId", "electronId"}

    # all electronIds should be 0
    assert np.all(result_df.index.get_level_values("electronId") == 0)

    # pulse ids should span 0-499 on each train
    for train_id in result_df.index.get_level_values("trainId"):
        assert np.all(
            result_df.loc[train_id].index.get_level_values("pulseId").values == np.arange(500),
        )
    # assert index uniqueness
    assert result_df.index.is_unique

    # assert that dataframe contains all channels
    assert np.all(
        set(result_df.columns) == set(get_channels(config_dataframe, ["per_pulse"])),
    )


def test_create_dataframe_per_train(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """Test the creation of a pandas DataFrame for a channel of type [per train]."""
    df = DataFrameCreator(config_dataframe, h5_paths[0])
    result_df = df.df_train

    channel = "delayStage"
    key, data = df.get_dataset_array(channel, slice_=True)

    # Check that the result_df is a DataFrame and has the correct shape
    assert isinstance(result_df, DataFrame)

    # check that all values are in the df for delayStage
    assert np.all(result_df[channel].dropna() == data[()])

    # check that dataframe contains all channels
    assert np.all(
        set(result_df.columns)
        == set(get_channels(config_dataframe, ["per_train"], extend_aux=True)),
    )

    # Ensure DataFrame has rows equal to unique keys from "per_train" channels, considering
    # different channels may have data for different trains. This checks the DataFrame's
    # completeness and integrity, especially important when channels record at varying trains.
    channels = get_channels(config_dataframe, ["per_train"])
    all_keys = Index([])
    for channel in channels:
        # Append unique keys from each channel, considering only training data
        all_keys = all_keys.append(df.get_dataset_array(channel, slice_=True)[0])
    # Verify DataFrame's row count matches unique train IDs count across channels
    assert result_df.shape[0] == len(all_keys.unique())

    # check index levels
    assert set(result_df.index.names) == {"trainId", "pulseId", "electronId"}

    # all pulseIds and electronIds should be 0
    assert np.all(result_df.index.get_level_values("pulseId") == 0)
    assert np.all(result_df.index.get_level_values("electronId") == 0)

    channel = "dldAux"
    key, data = df.get_dataset_array(channel, slice_=True)

    # Check if the subchannels are correctly sliced into the dataframe
    # The values are stored in DLD which is a 2D array
    # The subchannels are stored in the second dimension
    # Only index amount of values are stored in the first dimension, the rest are NaNs
    # hence the slicing
    subchannels = config_dataframe["channels"]["dldAux"]["sub_channels"]
    for subchannel, values in subchannels.items():
        assert np.all(df.df_train[subchannel].dropna().values == data[: key.size, values["slice"]])

    assert result_df.index.is_unique


def test_group_name_not_in_h5(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """Test ValueError when the group_name for a channel does not exist in the H5 file."""
    channel = "dldPosX"
    config = config_dataframe
    config["channels"][channel]["dataset_key"] = "foo"
    df = DataFrameCreator(config, h5_paths[0])

    with pytest.raises(KeyError):
        df.df_electron


def test_create_dataframe_per_file(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """Test the creation of pandas DataFrames for a given file."""
    df = DataFrameCreator(config_dataframe, h5_paths[0])
    result_df = df.df

    # Check that the result_df is a DataFrame and has the correct shape
    assert isinstance(result_df, DataFrame)
    all_keys = df.df_train.index.append(df.df_electron.index).append(df.df_pulse.index)
    all_keys = all_keys.unique()
    assert result_df.shape[0] == len(all_keys.unique())


def test_get_index_dataset_key_error(config_dataframe: dict, h5_paths: list[Path]) -> None:
    """
    Test that a ValueError is raised when the dataset_key is missing for a channel in the config.
    """
    config = config_dataframe
    channel = "dldPosX"
    df = DataFrameCreator(config, h5_paths[0])

    del config["channels"][channel]["dataset_key"]
    with pytest.raises(ValueError):
        df.get_index_dataset_key(channel)
