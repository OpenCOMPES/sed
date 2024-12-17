"""
This module creates pandas DataFrames from HDF5 files for different levels of data granularity
[per electron, per pulse, and per train]. It efficiently handles concatenation of data from
various channels within the HDF5 file, making use of the structured nature data to optimize
join operations. This approach significantly enhances performance compared to earlier.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sed.loader.flash.utils import get_channels
from sed.loader.flash.utils import InvalidFileError


class DataFrameCreator:
    """
    A class for creating pandas DataFrames from an HDF5 file.

    Attributes:
        h5_file (h5py.File): The HDF5 file object.
        multi_index (pd.MultiIndex): The multi-index structure for the DataFrame.
        _config (dict): The configuration dictionary for the DataFrame.
    """

    def __init__(self, config_dataframe: dict, h5_path: Path) -> None:
        """
        Initializes the DataFrameCreator class.

        Args:
            config_dataframe (dict): The configuration dictionary with only the dataframe key.
            h5_path (Path): Path to the h5 file.
        """
        self.h5_file = h5py.File(h5_path, "r")
        self.multi_index = get_channels(index=True)
        self._config = config_dataframe

    def get_dataset_key(self, channel: str) -> str:
        """
        Checks if 'dataset_key' exists and returns that.

        Args:
            channel (str): The name of the channel.

        Returns:
            str: The 'dataset_key'.

        Raises:
            ValueError: If 'dataset_key' is not provided.
        """
        channel_config = self._config["channels"][channel]
        if "dataset_key" in channel_config:
            return channel_config["dataset_key"]
        error = f"For channel: {channel}, provide 'dataset_key'."
        raise ValueError(error)

    def get_dataset_array(
        self,
        channel: str,
    ) -> h5py.Dataset:
        """
        Returns a numpy array for a given channel name.

        Args:
            channel (str): The name of the channel.
            slice_ (bool): Applies slicing on the dataset. Default is True.

        Returns:
            tuple[pd.Index, np.ndarray | h5py.Dataset]: A tuple containing the train ID
            pd.Index and the channel's data.
        """
        # Get the data from the necessary h5 file and channel
        dataset_key = self.get_dataset_key(channel)
        dataset = self.h5_file[dataset_key]

        return dataset

    @property
    def df_electron(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame for channel names of type [per electron].

        Returns:
            pd.DataFrame: The pandas DataFrame for the 'per_electron' channel's data.
        """
        # Get the relevant channels and their slice index
        channels = get_channels(self._config, "per_electron")
        if channels == []:
            return pd.DataFrame()

        series = {channel: pd.Series(self.get_dataset_array(channel)) for channel in channels}
        dataframe = pd.concat(series, axis=1)
        return dataframe.dropna()

    @property
    def df_train(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame for given channel names of type [per pulse].

        Returns:
            pd.DataFrame: The pandas DataFrame for the 'per_train' channel's data.
        """
        series = []
        # Get the relevant channel names
        channels = get_channels(self._config, "per_train")
        # For each channel, a pd.Series is created and appended to the list
        for channel in channels:
            # train_index and (sliced) data is returned
            dataset = self.get_dataset_array(channel)
            # Electron and pulse resolved MultiIndex is created. Since this is train data,
            # the electron and pulse index is always 0
            index = np.cumsum([0, *self.get_dataset_array("numEvents")[:-1]])
            # Auxiliary dataset (which is stored in the same dataset as other DLD channels)
            # contains multiple channels inside. Even though they are resolved per train,
            # they come in pulse format, so the extra values are sliced and individual channels are
            # created and appended to the list
            aux_alias = self._config.get("aux_alias", "dldAux")
            if channel == aux_alias:
                try:
                    sub_channels = self._config["channels"][aux_alias]["subChannels"]
                except KeyError:
                    raise KeyError(
                        f"Provide 'subChannels' for auxiliary channel '{aux_alias}'.",
                    )
                for name, values in sub_channels.items():
                    series.append(
                        pd.Series(
                            dataset[:, values["slice"]],
                            index,
                            name=name,
                        ),
                    )
            else:
                series.append(pd.Series(dataset, index, name=channel))
        # All the channels are concatenated to a single DataFrame
        return pd.concat(series, axis=1)

    def validate_channel_keys(self) -> None:
        """
        Validates if the index and dataset keys for all channels in the config exist in the h5 file.

        Raises:
            InvalidFileError: If the index or dataset keys are missing in the h5 file.
        """
        invalid_channels = []
        for channel in self._config["channels"]:
            dataset_key = self.get_dataset_key(channel)
            if dataset_key not in self.h5_file:
                invalid_channels.append(channel)

        if invalid_channels:
            raise InvalidFileError(invalid_channels)

    @property
    def df(self) -> pd.DataFrame:
        """
        Joins the 'per_electron', 'per_pulse', and 'per_train' using concat operation,
        returning a single dataframe.

        Returns:
            pd.DataFrame: The combined pandas DataFrame.
        """

        self.validate_channel_keys()
        # been tested with merge, join and concat
        # concat offers best performance, almost 3 times faster
        df = pd.concat((self.df_electron, self.df_train), axis=1)
        df[self.df_train.columns] = df[self.df_train.columns].ffill()
        return df
