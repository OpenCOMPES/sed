"""Tests for DataFrameCreator functionality"""
import os
from importlib.util import find_spec

import h5py
import numpy as np
import pytest
from pandas import DataFrame
from pandas import Index

from sed.loader.fel import DataFrameCreator


package_dir = os.path.dirname(find_spec("sed").origin)
config_path = os.path.join(package_dir, "../tests/data/loader/flash/config.yaml")
H5_PATH = "FLASH1_USER3_stream_2_run43878_file1_20230130T153807.1.h5"
# Define expected channels for each format.
ELECTRON_CHANNELS = ["dldPosX", "dldPosY", "dldTimeSteps"]
PULSE_CHANNELS = ["pulserSignAdc", "gmdTunnel"]
TRAIN_CHANNELS = ["timeStamp", "delayStage", "dldAux"]
TRAIN_CHANNELS_EXTENDED = [
    "sampleBias",
    "tofVoltage",
    "extractorVoltage",
    "extractorCurrent",
    "cryoTemperature",
    "sampleTemperature",
    "dldTimeBinSize",
    "timeStamp",
    "delayStage",
]
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

    # Request channels for 'per_train' format without expanding the dldAuxChannels.
    format_train = df.get_channels("per_train", extend_aux=False)

    # Request channels for 'per_train' format using a list.
    format_train_extended = df.get_channels(["per_train"])

    # Request channels for 'all' formats using a list.
    format_all = df.get_channels(["all"])

    # Request index channels only.
    format_index = df.get_channels(index=True)

    # Request 'per_electron' format and include index channels.
    format_index_electron = df.get_channels(["per_electron"], index=True)

    # Request 'all' formats and include index channels.
    format_all_index = df.get_channels(["all"], index=True)

    # Request 'all' formats and include index channels and extend aux channels
    format_all_index_extend_aux = df.get_channels(["all"], index=True, extend_aux=False)

    # Assert that the obtained channels match the expected channels.
    assert set(ELECTRON_CHANNELS) == set(format_electron)
    assert set(TRAIN_CHANNELS_EXTENDED) == set(format_train_extended)
    assert set(TRAIN_CHANNELS) == set(format_train)
    assert set(PULSE_CHANNELS) == set(format_pulse)
    assert set(ELECTRON_CHANNELS + TRAIN_CHANNELS_EXTENDED + PULSE_CHANNELS) == set(format_all)
    assert set(INDEX_CHANNELS) == set(format_index)
    assert set(INDEX_CHANNELS + ELECTRON_CHANNELS) == set(format_index_electron)
    assert set(
        INDEX_CHANNELS + ELECTRON_CHANNELS + TRAIN_CHANNELS_EXTENDED + PULSE_CHANNELS,
    ) == set(
        format_all_index,
    )
    assert set(INDEX_CHANNELS + ELECTRON_CHANNELS + PULSE_CHANNELS + TRAIN_CHANNELS) == set(
        format_all_index_extend_aux,
    )


def test_get_index_dataset_key(config_dataframe):
    config = config_dataframe
    channel = "dldPosX"
    df = DataFrameCreator(config)
    index_key, dataset_key = df.get_index_dataset_key(channel)
    group_name = config["channels"][channel]["group_name"]
    assert index_key == group_name + "index"
    assert dataset_key == group_name + "value"

    # remove group_name key
    del config["channels"][channel]["group_name"]
    with pytest.raises(ValueError):
        df.get_index_dataset_key(channel)


def test_get_dataset_array(config_dataframe, h5_file):

    df = DataFrameCreator(config_dataframe)
    df.h5_file = h5_file
    channel = "dldPosX"

    train_id, dset = df.get_dataset_array(channel)
    # Check that the train_id and np_array have the correct shapes and types
    assert isinstance(train_id, Index)
    assert isinstance(dset, h5py.Dataset)
    assert train_id.name == "trainId"
    assert train_id.shape[0] == dset.shape[0]
    assert dset.shape[1] == 5
    assert dset.shape[2] == 181

    train_id, dset = df.get_dataset_array(channel, slice=True)
    assert train_id.shape[0] == dset.shape[0]
    assert dset.shape[1] == 181

    channel = "gmdTunnel"
    train_id, dset = df.get_dataset_array(channel, True)
    assert train_id.shape[0] == dset.shape[0]
    assert dset.shape[1] == 500


def test_empty_get_dataset_array(config_dataframe, h5_file, h5_file_copy):

    channel = "gmdTunnel"
    df = DataFrameCreator(config_dataframe)
    df.h5_file = h5_file
    train_id, dset = df.get_dataset_array(channel)

    channel_index_key = config_dataframe["channels"][channel]["group_name"] + "index"
    # channel_dataset_key = config_dataframe["channels"][channel]["group_name"] + "value"
    empty_dataset_key = config_dataframe["channels"][channel]["group_name"] + "empty"
    config_dataframe["channels"][channel]["index_key"] = channel_index_key
    config_dataframe["channels"][channel]["dataset_key"] = empty_dataset_key
    # Remove the 'group_name' key
    del config_dataframe["channels"][channel]["group_name"]

    # create an empty dataset
    empty_dataset = h5_file_copy.create_dataset(
        name=empty_dataset_key,
        shape=(train_id.shape[0], 0),
    )
    print(empty_dataset)

    print(h5_file_copy[empty_dataset_key])

    df = DataFrameCreator(config_dataframe)
    df.h5_file = h5_file_copy
    train_id, dset_empty = df.get_dataset_array(channel)

    assert dset_empty.shape[0] == train_id.shape[0]
    assert dset.shape[1] == 8
    assert dset_empty.shape[1] == 0


def test_df_electron(config_dataframe, h5_file, multiindex_electron):
    """
    Test the creation of a pandas DataFrame for a channel of type [per electron].
    """
    df = DataFrameCreator(config_dataframe)
    df.h5_file = h5_file

    result_df = df.df_electron

    # Check that the values are dropped for pulseId index below 0 (ubid_offset)
    # this data has no nan so size should only decrease with the dropped values
    print(np.all(result_df.values[:7] != [720, 718, 509, 510, 449, 448]))
    assert False
    assert np.all(result_df.values[:7] != [720, 718, 509, 510, 449, 448])
    assert np.all(result_df.index.get_level_values("pulseId") >= 0)
    assert isinstance(result_df, DataFrame)

    # check if the dataframe shape is correct after dropping
    filtered_index = [item for item in result_df.index if item[1] >= 0]
    assert result_df.shape[0] == len(filtered_index)

    assert len(result_df[result_df.index.duplicated(keep=False)]) == 0


# def test_create_dataframe_per_pulse(config_dataframe, h5_file):
#     """
#     Test the creation of a pandas DataFrame for a channel of type [per pulse].
#     """
#     df = DataFrameCreator(config_dataframe)
#     train_id, np_array = df.create_numpy_array_per_channel(h5_file, "pulserSignAdc")
#     result_df = df.create_dataframe_per_pulse(np_array, train_id, "pulserSignAdc")

#     # Check that the result_df is a DataFrame and has the correct shape
#     assert isinstance(result_df, DataFrame)
#     assert result_df.shape[0] == np_array.shape[0] * np_array.shape[1]

#     train_id, np_array = df.create_numpy_array_per_channel(h5_file, "dldAux")
#     result_df = df.create_dataframe_per_pulse(np_array, train_id, "dldAux")

#     # Check if the subchannels are correctly sliced into the dataframe
#     assert isinstance(result_df, DataFrame)
#     assert result_df.shape[0] == len(train_id)
#     channel = "sampleBias"
#     assert np.all(result_df[channel].values == np_array[:, 0])

#     # check that dataframe contains all subchannels
#     assert np.all(
#         set(result_df.columns)
#         == set(config_dataframe["channels"]["dldAux"]["dldAuxChannels"].keys()),
#     )

#     assert len(result_df[result_df.index.duplicated(keep=False)]) == 0


# def test_create_dataframe_per_train(config_dataframe, h5_file):
#     """
#     Test the creation of a pandas DataFrame for a channel of type [per train].
#     """
#     df = DataFrameCreator(config_dataframe)
#     channel = "delayStage"

#     train_id, np_array = df.create_numpy_array_per_channel(h5_file, channel)
#     result_df = df.create_dataframe_per_train(np_array, train_id, channel)

#     # Check that the result_df is a DataFrame and has the correct shape
#     assert isinstance(result_df, DataFrame)
#     assert result_df.shape[0] == train_id.shape[0]
#     assert np.all(np.equal(np.squeeze(result_df.values), np_array))

#     assert len(result_df[result_df.index.duplicated(keep=False)]) == 0


# @pytest.mark.parametrize(
#     "channel",
#     [ELECTRON_CHANNELS[0], "dldAux", PULSE_CHANNELS_EXTENDED[-1], TRAIN_CHANNELS[0]],
# )
# def test_create_dataframe_per_channel(config_dataframe, h5_file, multiindex_electron, channel):
#     """
#     Test the creation of a pandas Series or DataFrame for a channel from a given file.
#     """
#     df = DataFrameCreator(config_dataframe)
#     df.index_per_electron = multiindex_electron

#     result = df.create_dataframe_per_channel(h5_file, channel)

#     # Check that the result is a Series or DataFrame and has the correct shape
#     assert isinstance(result, DataFrame)


# def test_invalid_channel_format(config_dataframe, h5_file):
#     """
#     Test ValueError for an invalid channel format.
#     """
#     config = config_dataframe
#     config["channels"]["dldPosX"]["format"] = "foo"

#     df = DataFrameCreator(config_dataframe)

#     with pytest.raises(ValueError):
#         df.create_dataframe_per_channel(h5_file, "dldPosX")


# def test_concatenate_channels(config_dataframe, h5_file):
#     """
#     Test the concatenation of channels from an h5py.File into a pandas DataFrame.
#     """

#     df = DataFrameCreator(config_dataframe)
#     # Take channels for different formats as they have differing lengths
#     # (train_ids can also differ)
#     print(df.get_channels("all", extend_aux=False))
#     df_channels_list = [df.create_dataframe_per_channel(
#         h5_file, channel).index for channel in df.get_channels("all", extend_aux=False)]
#     # # print all indices
#     # for i, index in enumerate(df_channels_list):
#     #     print(df.available_channels[i], index)
#     # create union of all indices and sort them
#     union_index = sorted(set().union(*df_channels_list))
#     # print(union_index)

#     result_df = df.concatenate_channels(h5_file)

#     diff_index = result_df.index.difference(union_index)
#     print(diff_index)
#     print(result_df.shape)
#     # Check that the result_df is a DataFrame and has the correct shape
#     assert isinstance(result_df, DataFrame)
#     assert np.all(result_df.index == union_index)


# def test_group_name_not_in_h5(config_dataframe, h5_file):
#     """
#     Test ValueError when the group_name for a channel does not exist in the H5 file.
#     """
#     channel = "dldPosX"
#     config = config_dataframe
#     config["channels"][channel]["group_name"] = "foo"
#     index_key = "foo" + "index"
#     df = DataFrameCreator(config)

#     with pytest.raises(ValueError) as e:
#         df.concatenate_channels(h5_file)

#     assert str(e.value.args[0]
#                ) == f"The index key: {index_key} for channel {channel} does not exist."


# def test_create_dataframe_per_file(config_dataframe, h5_file):
#     """
#     Test the creation of pandas DataFrames for a given file.
#     """
#     df = DataFrameCreator(config_dataframe)
#     result_df = df.create_dataframe_per_file(Path(h5_file.filename))

#     # Check that the result_df is a DataFrame and has the correct shape
#     assert isinstance(result_df, DataFrame)
#     assert result_df.shape[0] == df.concatenate_channels(h5_file).shape[0]
