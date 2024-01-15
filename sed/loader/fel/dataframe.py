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

from functools import reduce

import h5py
import numpy as np
from pandas import concat
from pandas import DataFrame
from pandas import Index
from pandas import MultiIndex
from pandas import Series

from sed.loader.fel.config_model import DataFrameConfig
from sed.loader.fel.utils import get_channels


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

    @property
    def df_electron(self) -> DataFrame:
        """
        Returns a pandas DataFrame for a given channel name of type [per electron].

        Returns:
            DataFrame: The pandas DataFrame for the 'per_electron' channel's data.
        """
        raise NotImplementedError("This method must be implemented in a child class.")

    @property
    def df_pulse(self) -> DataFrame:
        """
        Returns a pandas DataFrame for a given channel name of type [per pulse].

        Returns:
            DataFrame: The pandas DataFrame for the 'per_pulse' channel's data.
        """
        series = []
        channels = get_channels(self._config.channels, "per_pulse")
        if not channels:
            return DataFrame()

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
        if not channels:
            return DataFrame()

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

        dfs_to_join = (self.df_electron, self.df_pulse, self.df_train)

        def conditional_join(left_df, right_df):
            """Performs conditional join of two dataframes.
            Logic: if both dataframes are empty, return empty dataframe.
            If one of the dataframes is empty, return the other dataframe.
            Otherwise, perform outer join on multiindex of the dataframes."""
            return (
                DataFrame()
                if left_df.empty and right_df.empty
                else right_df
                if left_df.empty
                else left_df
                if right_df.empty
                else left_df.join(right_df, on=self.multi_index, how="outer")
            )

        # Perform conditional join for each combination of dataframes using reduce
        joined_df = reduce(conditional_join, dfs_to_join)

        return joined_df.sort_index()
