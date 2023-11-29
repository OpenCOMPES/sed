"""Tests for DataFrameCreator functionality"""
import os
from importlib.util import find_spec

import numpy as np
import pytest
from pandas import DataFrame
from pandas import Series

from sed.loader.fel import DataFrameCreator


package_dir = os.path.dirname(find_spec("sed").origin)
config_path = os.path.join(package_dir, "../tests/data/loader/flash/config.yaml")
H5_PATH = "FLASH1_USER3_stream_2_run43878_file1_20230130T153807.1.h5"
# Define expected channels for each format.
ELECTRON_CHANNELS = ["dldPosX", "dldPosY", "dldTimeSteps"]
PULSE_CHANNELS = [
    "sampleBias",
    "tofVoltage",
    "extractorVoltage",
    "extractorCurrent",
    "cryoTemperature",
    "sampleTemperature",
    "dldTimeBinSize",
    "gmdTunnel",
]
TRAIN_CHANNELS = ["timeStamp", "delayStage"]
INDEX_CHANNELS = ["trainId", "pulseId", "electronId"]


def test_get_channels_by_format(config_dataframe):
    """
    Test function to verify the 'get_channels' method in FlashLoader class for
    retrieving channels based on formats and index inclusion.
    """
    # Initialize the FlashLoader instance with the given config_file.
    df = DataFrameCreator(config_dataframe)

    # Call get_channels method with different format options.

    # Request channels for 'per_electron' format using a list.
    format_electron = df.get_channels(["per_electron"])

    # Request channels for 'per_pulse' format using a string.
    format_pulse = df.get_channels("per_pulse")

    # Request channels for 'per_train' format using a list.
    format_train = df.get_channels(["per_train"])

    # Request channels for 'all' formats using a list.
    format_all = df.get_channels(["all"])

    # Request index channels only.
    format_index = df.get_channels(index=True)

    # Request 'per_electron' format and include index channels.
    format_index_electron = df.get_channels(["per_electron"], index=True)

    # Request 'all' formats and include index channels.
    format_all_index = df.get_channels(["all"], index=True)

    # Assert that the obtained channels match the expected channels.
    assert set(ELECTRON_CHANNELS) == set(format_electron)
    assert set(PULSE_CHANNELS) == set(format_pulse)
    assert set(TRAIN_CHANNELS) == set(format_train)
    assert set(ELECTRON_CHANNELS + PULSE_CHANNELS + TRAIN_CHANNELS) == set(format_all)
    assert set(INDEX_CHANNELS) == set(format_index)
    assert set(INDEX_CHANNELS + ELECTRON_CHANNELS) == set(format_index_electron)
    assert set(INDEX_CHANNELS + ELECTRON_CHANNELS + PULSE_CHANNELS + TRAIN_CHANNELS) == set(
        format_all_index,
    )


def test_invalid_channel_format(config_dataframe, h5_file):
    """
    Test ValueError for an invalid channel format.
    """
    config = config_dataframe
    config["channels"]["dldPosX"]["format"] = "foo"

    df = DataFrameCreator(config_dataframe)

    with pytest.raises(ValueError):
        print(config["channels"]["dldPosX"]["group_name"])
        df.create_dataframe_per_channel(h5_file, "dldPosX")


def test_group_name_not_in_h5(config_dataframe, h5_file):
    """
    Test ValueError when the group_name for a channel does not exist in the H5 file.
    """
    config = config_dataframe
    config["channels"]["dldPosX"]["group_name"] = "foo"
    df = DataFrameCreator(config)

    with pytest.raises(ValueError) as e:
        df.concatenate_channels(h5_file)

    assert str(e.value.args[0]) == "The group_name for channel dldPosX does not exist."


def test_create_numpy_array_per_channel(config_dataframe, h5_file):
    """
    Test the creation of a numpy array for a given channel.
    """
    df = DataFrameCreator(config_dataframe)
    channel = "dldPosX"

    train_id, np_array = df.create_numpy_array_per_channel(h5_file, channel)
    print(np_array.shape)
    # Check that the train_id and np_array have the correct shapes and types
    assert isinstance(train_id, Series)
    assert isinstance(np_array, np.ndarray)
    assert train_id.name == "trainId"
    assert train_id.shape[0] == np_array.shape[0]
    assert np_array.shape[1] == 2048


def test_create_dataframe_per_electron(config_dataframe, h5_file, multiindex_electron):
    """
    Test the creation of a pandas DataFrame for a channel of type [per electron].
    """
    df = DataFrameCreator(config_dataframe)
    df.index_per_electron = multiindex_electron

    channel = "dldPosX"

    train_id, np_array = df.create_numpy_array_per_channel(h5_file, channel)
    # this data has no nan so size should only decrease with
    result_df = df.create_dataframe_per_electron(np_array, train_id, channel)

    # Check that the values are dropped for pulseId index below 0 (ubid_offset)
    # this data has no nan so size should only decrease with the dropped values
    assert np.all(result_df.values[:7] != [720, 718, 509, 510, 449, 448])
    assert np.all(result_df.index.get_level_values("pulseId") >= 0)
    assert isinstance(result_df, DataFrame)

    # check if the dataframe shape is correct after dropping
    filtered_index = [item for item in df.index_per_electron if item[1] >= 0]
    assert result_df.shape[0] == len(filtered_index)


def test_create_dataframe_per_pulse(config_dataframe, h5_file):
    """
    Test the creation of a pandas DataFrame for a channel of type [per pulse].
    """
    df = DataFrameCreator(config_dataframe)
    train_id, np_array = df.create_numpy_array_per_channel(h5_file, "gmdTunnel")
    result_df = df.create_dataframe_per_pulse(np_array, train_id, "gmdTunnel")

    # Check that the result_df is a DataFrame and has the correct shape
    assert isinstance(result_df, DataFrame)
    assert result_df.shape[0] == np_array.shape[0] * np_array.shape[1]

    train_id, np_array = df.create_numpy_array_per_channel(h5_file, "dldAux")
    result_df = df.create_dataframe_per_pulse(np_array, train_id, "dldAux")

    # Check if the subchannels are correctly sliced into the dataframe
    assert isinstance(result_df, DataFrame)
    assert result_df.shape[0] == len(train_id)
    channel = "sampleBias"
    assert np.all(result_df[channel].values == np_array[:, 0])

    # check that dataframe contains all subchannels
    assert np.all(
        set(result_df.columns)
        == set(config_dataframe["channels"]["dldAux"]["dldAuxChannels"].keys()),
    )


# def test_create_dataframe_per_train(config_dataframe, h5_file):
#     """
#     Test the creation of a pandas DataFrame for a channel of type [per train].
#     """
#     df = DataFrameCreator(config_dataframe)
#     channel = "timeStamp"

#     train_id, np_array = df.create_numpy_array_per_channel(h5_file, channel)
#     result_df = df.create_dataframe_per_train(np_array, train_id, channel)

#     # Check that the result_df is a DataFrame and has the correct shape
#     assert isinstance(result_df, DataFrame)
#     assert result_df.shape[0] == train_id.shape[0]


# def test_create_dataframe_per_channel(config_dataframe, h5_file):
#     """
#     Test the creation of a pandas Series or DataFrame for a channel from a given file.
#     """
#     df = DataFrameCreator(config_dataframe)
#     channel = "dldPosX"

#     result = df.create_dataframe_per_channel(h5_file, channel)

#     # Check that the result is a Series or DataFrame and has the correct shape
#     assert isinstance(result, (Series, DataFrame))
#     assert result.shape[0] == df.create_numpy_array_per_channel(h5_file, channel)[0].shape[0]


# def test_concatenate_channels(config_dataframe, h5_file):
#     """
#     Test the concatenation of channels from an h5py.File into a pandas DataFrame.
#     """
#     df = DataFrameCreator(config_dataframe)
#     result_df = df.concatenate_channels(h5_file)

#     # Check that the result_df is a DataFrame and has the correct shape
#     assert isinstance(result_df, DataFrame)
#     assert result_df.shape[0] == df.create_dataframe_per_file(Path(h5_file.filename)).shape[0]


# def test_create_dataframe_per_file(config_dataframe, h5_file):
#     """
#     Test the creation of pandas DataFrames for a given file.
#     """
#     df = DataFrameCreator(config_dataframe)
#     result_df = df.create_dataframe_per_file(Path(h5_file.filename))

#     # Check that the result_df is a DataFrame and has the correct shape
#     assert isinstance(result_df, DataFrame)
#     assert result_df.shape[0] == df.concatenate_channels(h5_file).shape[0]
