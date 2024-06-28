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

    def get_index_dataset_key(self, channel: str) -> tuple[str, str]:
        """
        Checks if 'group_name' and converts to 'index_key' and 'dataset_key' if so.

        Args:
            channel (str): The name of the channel.

        Returns:
            tuple[str, str]: Outputs a tuple of 'index_key' and 'dataset_key'.

        Raises:
            ValueError: If 'index_key' and 'dataset_key' are not provided.
        """
        channel_config = self._config["channels"][channel]

        if "index_key" in channel_config and "dataset_key" in channel_config:
            return channel_config["index_key"], channel_config["dataset_key"]
        else:
            print("'group_name' is no longer supported.")

        raise ValueError(
            "For channel:",
            channel,
            "Provide both 'index_key' and 'dataset_key'.",
        )

    def get_dataset_array(
        self,
        channel: str,
        slice_: bool = False,
    ) -> tuple[pd.Index, h5py.Dataset]:
        """
        Returns a numpy array for a given channel name.

        Args:
            channel (str): The name of the channel.
            slice_ (bool): If True, applies slicing on the dataset.

        Returns:
            tuple[pd.Index, h5py.Dataset]: A tuple containing the train ID
            pd.Index and the numpy array for the channel's data.
        """
        # Get the data from the necessary h5 file and channel
        index_key, dataset_key = self.get_index_dataset_key(channel)

        key = pd.Index(self.h5_file[index_key], name="trainId")  # macrobunch
        dataset = self.h5_file[dataset_key]

        if slice_:
            slice_index = self._config["channels"][channel].get("slice", None)
            if slice_index is not None:
                dataset = np.take(dataset, slice_index, axis=1)
        # If np_array is size zero, fill with NaNs
        if dataset.shape[0] == 0:
            # Fill the np_array with NaN values of the same shape as train_id
            dataset = np.full_like(key, np.nan, dtype=np.double)

        return key, dataset

    def pulse_dataset(self, offset: int) -> tuple[pd.MultiIndex, slice | np.ndarray]:
        """
        Creates a multi-level index that combines train IDs and pulse IDs, and handles
        sorting and electron counting within each pulse.

        Args:
            offset (int): The offset value.

        Returns:
            tuple[pd.MultiIndex, np.ndarray]: A tuple containing the computed pd.MultiIndex and
            the indexer.
        """
        # Get the pulse_dataset and the train_index
        train_index, pulse_dataset = self.get_dataset_array("pulseId", slice_=True)
        # pulse_dataset comes as a 2D array, resolved per train. Here it is flattened
        # the daq has an offset so no pulses are missed. This offset is subtracted here
        pulse_ravel = pulse_dataset.ravel() - offset
        # Here train_index is repeated to match the size of pulses
        train_index_repeated = np.repeat(train_index, pulse_dataset.shape[1])
        # A pulse resolved multi-index is finally created.
        # Since there can be NaN pulses, those are dropped
        pulse_index = pd.MultiIndex.from_arrays((train_index_repeated, pulse_ravel)).dropna()

        # Sometimes the pulse_index are not monotonic, so we might need to sort them
        # The indexer is also returned to sort the data in df_electron
        indexer = slice(None)
        if not pulse_index.is_monotonic_increasing:
            pulse_index, indexer = pulse_index.sort_values(return_indexer=True)

        # In the data, to signify different electrons, pulse_index is repeated by
        # the number of electrons in each pulse. Here the values are counted
        electron_counts = pulse_index.value_counts(sort=False).values
        # Now we resolve each pulse to its electrons
        electron_index = np.concatenate([np.arange(count) for count in electron_counts])

        # Final multi-index constructed here
        index = pd.MultiIndex.from_arrays(
            (
                pulse_index.get_level_values(0),
                pulse_index.get_level_values(1).astype(int),
                electron_index,
            ),
            names=self.multi_index,
        )
        return index, indexer

    @property
    def df_electron(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame for channel names of type [per electron].

        Returns:
            pd.DataFrame: The pandas DataFrame for the 'per_electron' channel's data.
        """
        offset = self._config["ubid_offset"]
        # Here we get the multi-index and the indexer to sort the data
        index, indexer = self.pulse_dataset(offset)

        # Get the relevant channels and their slice index
        channels = get_channels(self._config["channels"], "per_electron")
        slice_index = [self._config["channels"][channel].get("slice", None) for channel in channels]

        # First checking if dataset keys are the same for all channels
        # because DLD at FLASH stores all channels in the same h5 dataset
        dataset_keys = [self.get_index_dataset_key(channel)[1] for channel in channels]
        # Gives a true if all keys are the same
        all_keys_same = all(key == dataset_keys[0] for key in dataset_keys)

        # If all dataset keys are the same, we only need to load the dataset once and slice
        # the appropriate columns. This is much faster than loading the same dataset multiple times
        if all_keys_same:
            _, dataset = self.get_dataset_array(channels[0])
            data_dict = {
                channel: dataset[:, slice_, :].ravel()
                for channel, slice_ in zip(channels, slice_index)
            }
            dataframe = pd.DataFrame(data_dict)
        # In case channels do differ, we create a pd.Series for each channel and concatenate them
        else:
            series = {
                channel: pd.Series(self.get_dataset_array(channel, slice_=True)[1].ravel())
                for channel in channels
            }
            dataframe = pd.concat(series, axis=1)

        # after offset, the negative pulse values are dropped as they are not valid
        drop_vals = np.arange(-offset, 0)

        # Few things happen here:
        # Drop all NaN values like while creating the multiindex
        # if necessary, the data is sorted with [indexer]
        # pd.MultiIndex is set
        # Finally, the offset values are dropped
        return (
            dataframe.dropna()
            .iloc[indexer]
            .set_index(index)
            .drop(index=drop_vals, level="pulseId", errors="ignore")
        )

    @property
    def df_pulse(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame for given channel names of type [per pulse].

        Returns:
            pd.DataFrame: The pandas DataFrame for the 'per_pulse' channel's data.
        """
        series = []
        # Get the relevant channel names
        channels = get_channels(self._config["channels"], "per_pulse")
        # For each channel, a pd.Series is created and appended to the list
        for channel in channels:
            # train_index and (sliced) data is returned
            key, dataset = self.get_dataset_array(channel, slice_=True)
            # Electron resolved MultiIndex is created. Since this is pulse data,
            # the electron index is always 0
            index = pd.MultiIndex.from_product(
                (key, np.arange(0, dataset.shape[1]), [0]),
                names=self.multi_index,
            )
            # The dataset is opened and converted to numpy array by [()]
            # and flattened to resolve per pulse
            # The pd.Series is created with the MultiIndex and appended to the list
            series.append(pd.Series(dataset[()].ravel(), index=index, name=channel))

        # All the channels are concatenated to a single DataFrame
        return pd.concat(
            series,
            axis=1,
        )

    @property
    def df_train(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame for given channel names of type [per train].

        Returns:
            pd.DataFrame: The pandas DataFrame for the 'per_train' channel's data.
        """
        series = []
        # Get the relevant channel names
        channels = get_channels(self._config["channels"], "per_train")
        # For each channel, a pd.Series is created and appended to the list
        for channel in channels:
            # train_index and (sliced) data is returned
            key, dataset = self.get_dataset_array(channel, slice_=True)
            # Electron and pulse resolved MultiIndex is created. Since this is train data,
            # the electron and pulse index is always 0
            index = pd.MultiIndex.from_product(
                (key, [0], [0]),
                names=self.multi_index,
            )
            # Auxillary dataset (which is stored in the same dataset as other DLD channels)
            # contains multiple channels inside. Even though they are resolved per train,
            # they come in pulse format, so the extra values are sliced and individual channels are
            # created and appended to the list
            if channel == "dldAux":
                aux_channels = self._config["channels"]["dldAux"]["dldAuxChannels"].items()
                for name, slice_aux in aux_channels:
                    series.append(pd.Series(dataset[: key.size, slice_aux], index, name=name))
            else:
                series.append(pd.Series(dataset, index, name=channel))
        # All the channels are concatenated to a single DataFrame
        return pd.concat(series, axis=1)

    def validate_channel_keys(self) -> None:
        """
        Validates if the index and dataset keys for all channels in config exist in the h5 file.

        Raises:
            KeyError: If the index or dataset keys do not exist in the file.
        """
        for channel in self._config["channels"]:
            index_key, dataset_key = self.get_index_dataset_key(channel)
            if index_key not in self.h5_file:
                raise KeyError(f"pd.Index key '{index_key}' doesn't exist in the file.")
            if dataset_key not in self.h5_file:
                raise KeyError(f"Dataset key '{dataset_key}' doesn't exist in the file.")

    @property
    def df(self) -> pd.DataFrame:
        """
        Joins the 'per_electron', 'per_pulse', and 'per_train' using join operation,
        returning a single dataframe.

        Returns:
            pd.DataFrame: The combined pandas DataFrame.
        """

        self.validate_channel_keys()
        return (
            self.df_electron.join(self.df_pulse, on=self.multi_index, how="outer")
            .join(self.df_train, on=self.multi_index, how="outer")
            .sort_index()
        )
