"""
This module provides functionality for creating pandas DataFrames from HDF5 files with multiple
channels, found using get_channels method.

The DataFrameCreator class requires a configuration dictionary with only the dataframe key and
an open h5 file. It validates if provided [index and dataset keys] or [group_name key] has
groups existing in the h5 file.
Three formats of channels are supported: [per electron], [per pulse], and [per train].
These can be accessed using the df_electron, df_pulse, and df_train properties respectively.
The combined DataFrame can be accessed using the df property.
Typical usage example:

    df_creator = DataFrameCreator(config, h5_file)
    dataframe = df_creator.df
"""
from __future__ import annotations

import h5py
import numpy as np
from pandas import concat
from pandas import DataFrame
from pandas import Index
from pandas import MultiIndex
from pandas import Series

from sed.loader.fel.utils import get_channels
from sed.loader.flash.config_model import DataFrameConfig


class DataFrameCreator:
    """
    Utility class for creating pandas DataFrames from HDF5 files with multiple channels.
    """

    def __init__(self, config: DataFrameConfig, h5_file: h5py.File) -> None:
        """
        Initializes the DataFrameCreator class.

        Args:
            config (DataFrameConfig): The dataframe section of the config model.
            h5_file (h5py.File): The open h5 file.
        """
        self.h5_file: h5py.File = h5_file
        self.failed_files_error: list[str] = []
        self.multi_index = get_channels(index=True)
        self._config = config

    def get_dataset_array(
        self,
        channel: str,
        slice_: bool = False,
    ) -> tuple[Index, h5py.Dataset]:
        """
        Returns a numpy array for a given channel name.

        Args:
            channel (str): The name of the channel.
            slice_ (bool): If True, applies slicing on the dataset.

        Returns:
            tuple[Index, h5py.Dataset]: A tuple containing the train ID Index and the numpy array
            for the channel's data.
        """
        # Get the data from the necessary h5 file and channel
        index_key = self._config.channels.get(channel).index_key
        dataset_key = self._config.channels.get(channel).dataset_key

        key = Index(self.h5_file[index_key], name="trainId")  # macrobunch
        dataset = self.h5_file[dataset_key]

        if slice_:
            slice_index = self._config.channels.get(channel).slice
            if slice_index is not None:
                dataset = np.take(dataset, slice_index, axis=1)
        # If np_array is size zero, fill with NaNs
        if dataset.shape[0] == 0:
            # Fill the np_array with NaN values of the same shape as train_id
            dataset = np.full_like(key, np.nan, dtype=np.double)

        return key, dataset

    def pulse_index(self, offset: int) -> tuple[MultiIndex, slice | np.ndarray]:
        """
        Computes the index for the 'per_electron' data.

        Args:
            offset (int): The offset value.

        Returns:
            tuple[MultiIndex, np.ndarray]: A tuple containing the computed MultiIndex and
            the indexer.
        """
        # Get the pulseId and the index_train
        index_train, dataset_pulse = self.get_dataset_array("pulseId", slice_=True)
        # Repeat the index_train by the number of pulses
        index_train_repeat = np.repeat(index_train, dataset_pulse.shape[1])
        # Explode the pulse dataset and subtract by the ubid_offset
        pulse_ravel = dataset_pulse.ravel() - offset
        # Create a MultiIndex with the index_train and the pulse
        microbunches = MultiIndex.from_arrays((index_train_repeat, pulse_ravel)).dropna()

        # Only sort if necessary
        indexer = slice(None)
        if not microbunches.is_monotonic_increasing:
            microbunches, indexer = microbunches.sort_values(return_indexer=True)

        # Count the number of electrons per microbunch and create an array of electrons
        electron_counts = microbunches.value_counts(sort=False).values
        electrons = np.concatenate([np.arange(count) for count in electron_counts])

        # Final index constructed here
        index = MultiIndex.from_arrays(
            (
                microbunches.get_level_values(0),
                microbunches.get_level_values(1).astype(int),
                electrons,
            ),
            names=self.multi_index,
        )
        return index, indexer

    @property
    def df_electron(self) -> DataFrame:
        """
        Returns a pandas DataFrame for a given channel name of type [per electron].

        Returns:
            DataFrame: The pandas DataFrame for the 'per_electron' channel's data.
        """
        offset = self._config.ubid_offset
        # Index
        index, indexer = self.pulse_index(offset)

        # Data logic
        channels = get_channels(self._config.channels, "per_electron")
        slice_index = [self._config.channels.get(channel).slice for channel in channels]

        # First checking if dataset keys are the same for all channels
        dataset_keys = [self._config.channels.get(channel).dataset_key for channel in channels]
        all_keys_same = all(key == dataset_keys[0] for key in dataset_keys)

        # If all dataset keys are the same, we can directly use the ndarray to create frame
        if all_keys_same:
            _, dataset = self.get_dataset_array(channels[0])
            data_dict = {
                channel: dataset[:, slice_, :].ravel()
                for channel, slice_ in zip(channels, slice_index)
            }
            dataframe = DataFrame(data_dict)
        # Otherwise, we need to create a Series for each channel and concatenate them
        else:
            series = {
                channel: Series(self.get_dataset_array(channel, slice_=True)[1].ravel())
                for channel in channels
            }
            dataframe = concat(series, axis=1)

        drop_vals = np.arange(-offset, 0)

        # Few things happen here:
        # Drop all NaN values like while creating the multiindex
        # if necessary, the data is sorted with [indexer]
        # MultiIndex is set
        # Finally, the offset values are dropped
        return (
            dataframe.dropna()[indexer]
            .set_index(index)
            .drop(index=drop_vals, level="pulseId", errors="ignore")
        )

    @property
    def df_pulse(self) -> DataFrame:
        """
        Returns a pandas DataFrame for a given channel name of type [per pulse].

        Returns:
            DataFrame: The pandas DataFrame for the 'per_pulse' channel's data.
        """
        series = []
        channels = get_channels(self._config.channels, "per_pulse")
        for channel in channels:
            # get slice
            key, dataset = self.get_dataset_array(channel, slice_=True)
            index = MultiIndex.from_product(
                (key, np.arange(0, dataset.shape[1]), [0]),
                names=self.multi_index,
            )
            series.append(Series(dataset[()].ravel(), index=index, name=channel))

        return concat(series, axis=1)  # much faster when concatenating similarly indexed data first

    @property
    def df_train(self) -> DataFrame:
        """
        Returns a pandas DataFrame for a given channel name of type [per train].

        Returns:
            DataFrame: The pandas DataFrame for the 'per_train' channel's data.
        """
        series = []

        channels = get_channels(self._config.channels, "per_train")

        for channel in channels:
            key, dataset = self.get_dataset_array(channel, slice_=True)
            index = MultiIndex.from_product(
                (key, [0], [0]),
                names=self.multi_index,
            )
            if channel == "dldAux":
                aux_channels = self._config.channels["dldAux"].dldAuxChannels
                for aux_ch_name in aux_channels:
                    aux_ch = aux_channels[aux_ch_name]
                    series.append(
                        Series(dataset[: key.size, aux_ch.slice], index, name=aux_ch.name),
                    )
            else:
                series.append(Series(dataset, index, name=channel))

        return concat(series, axis=1)

    def validate_channel_keys(self) -> None:
        """
        Validates if the index and dataset keys for all channels in config exist in the h5 file.

        Raises:
            KeyError: If the index or dataset keys do not exist in the file.
        """
        for channel in self._config.channels:
            index_key = self._config.channels.get(channel).index_key
            dataset_key = self._config.channels.get(channel).dataset_key
            if index_key not in self.h5_file:
                raise KeyError(f"Index key '{index_key}' doesn't exist in the file.")
            if dataset_key not in self.h5_file:
                raise KeyError(f"Dataset key '{dataset_key}' doesn't exist in the file.")

    @property
    def df(self) -> DataFrame:
        """
        Joins the 'per_electron', 'per_pulse', and 'per_train' using join operation,
        returning a single dataframe.

        Returns:
            DataFrame: The combined pandas DataFrame.
        """

        self.validate_channel_keys()
        return (
            self.df_electron.join(self.df_pulse, on=self.multi_index, how="outer")
            .join(self.df_train, on=self.multi_index, how="outer")
            .sort_index()
        )
