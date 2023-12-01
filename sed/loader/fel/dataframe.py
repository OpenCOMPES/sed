from __future__ import annotations

from functools import reduce
from pathlib import Path

import h5py
import numpy as np
from pandas import DataFrame
from pandas import Series

from sed.loader.fel.multiindex import MultiIndexCreator
from sed.loader.utils import parse_h5_keys
from sed.loader.utils import split_dld_time_from_sector_id


class DataFrameCreator(MultiIndexCreator):
    """
    Utility class for creating pandas DataFrames from HDF5 files with multiple channels.
    """

    def __init__(self, config_dataframe: dict) -> None:
        """
        Initializes the DataFrameCreator class.

        Args:
            config_dataframe (dict): The configuration dictionary with only the dataframe key.
        """
        super().__init__()
        self.failed_files_error: list[str] = []
        self._config = config_dataframe

    @property
    def available_channels(self) -> list:
        """Returns the channel names that are available for use, excluding pulseId."""
        available_channels = list(self._config["channels"].keys())
        available_channels.remove("pulseId")
        return available_channels

    def get_channels(self, formats: str | list[str] = "", index: bool = False) -> list[str]:
        """
        Returns a list of channels associated with the specified format(s).

        Args:
            formats (Union[str, List[str]]): The desired format(s)
            ('per_pulse', 'per_electron', 'per_train', 'all').
            index (bool): If True, includes channels from the multi_index.

        Returns:
            List[str]: A list of channels with the specified format(s).
        """
        # If 'formats' is a single string, convert it to a list for uniform processing.
        if isinstance(formats, str):
            formats = [formats]

        # If 'formats' is a string "all", gather all possible formats.
        if formats == ["all"]:
            channels = self.get_channels(["per_pulse", "per_train", "per_electron"], index)
            return channels

        channels = []
        for format_ in formats:
            # Gather channels based on the specified format(s).
            channels.extend(
                key
                for key in self.available_channels
                if self._config["channels"][key]["format"] == format_ and key != "dldAux"
            )
            # Include 'dldAuxChannels' if the format is 'per_pulse'.
            if format_ == "per_pulse":
                channels.extend(
                    self._config["channels"]["dldAux"]["dldAuxChannels"].keys(),
                )

        # Include channels from multi_index if 'index' is True.
        if index:
            channels.extend(self.multi_index)

        return channels

    def get_index_dataset_key(self, channel: str) -> tuple[str, str]:
        """
        Checks if 'group_name' and converts to 'index_key' and 'dataset_key' if so.

        Args:
            channel (str): The name of the channel.

        Returns:
            tuple[str, str]: Outputs a tuple of 'index_key' and 'dataset_key'.

        Raises:
            ValueError: If neither 'group_name' nor both 'index_key' and 'dataset_key' are provided.
        """
        channel_config = self._config["channels"][channel]

        if "group_name" in channel_config:
            index_key = channel_config["group_name"] + "index"
            if channel == "timeStamp":
                dataset_key = channel_config["group_name"] + "time"
            else:
                dataset_key = channel_config["group_name"] + "value"
            return index_key, dataset_key
        elif "index_key" in channel_config and "dataset_key" in channel_config:
            return channel_config["index_key"], channel_config["dataset_key"]
        else:
            raise ValueError(
                "For channel:",
                channel,
                "Provide either both 'index_key' and 'dataset_key'.",
                "or 'group_name' (parses only 'index' and 'value' or 'time' keys.)",
            )

    def create_numpy_array_per_channel(
        self,
        h5_file: h5py.File,
        channel: str,
    ) -> tuple[Series, np.ndarray]:
        """
        Returns a numpy array for a given channel name for a given file.

        Args:
            h5_file (h5py.File): The h5py file object.
            channel (str): The name of the channel.

        Returns:
            Tuple[Series, np.ndarray]: A tuple containing the train ID Series and the numpy array
            for the channel's data.
        """
        # Get the data from the necessary h5 file and channel
        index_key, dataset_key = self.get_index_dataset_key(channel)

        slice = self._config["channels"][channel].get("slice", None)

        train_id = Series(h5_file[index_key], name="trainId")  # macrobunch
        np_array = h5_file[dataset_key][()]

        # Use predefined axis and slice from the json file
        # to choose correct dimension for necessary channel
        if slice is not None:
            np_array = np.take(
                np_array,
                slice,
                axis=1,
            )

        # If np_array is size zero, fill with NaNs
        if np_array.size == 0:
            # Fill the np_array with NaN values of the same shape as train_id
            np_array = np.full_like(train_id, np.nan, dtype=np.double)

        return train_id, np_array

    def create_dataframe_per_electron(
        self,
        np_array: np.ndarray,
        train_id: Series,
        channel: str,
    ) -> DataFrame:
        """
        Returns a pandas DataFrame for a given channel name of type [per electron].

        Args:
            np_array (np.ndarray): The numpy array containing the channel data.
            train_id (Series): The train ID Series.
            channel (str): The name of the channel.

        Returns:
            DataFrame: The pandas DataFrame for the channel's data.
        """
        return (
            Series((np_array[i] for i in train_id.index), name=channel)
            .explode()
            .dropna()
            .to_frame()
            .set_index(self.index_per_electron)
            .drop(
                index=np.arange(-self._config["ubid_offset"], 0),
                level=1,
                errors="ignore",
            )
        )

    def create_dataframe_per_pulse(
        self,
        np_array: np.ndarray,
        train_id: Series,
        channel: str,
    ) -> DataFrame:
        """
        Returns a pandas DataFrame for a given channel name of type [per pulse].

        Args:
            np_array (np.ndarray): The numpy array containing the channel data.
            train_id (Series): The train ID Series.
            channel (str): The name of the channel.

        Returns:
            DataFrame: The pandas DataFrame for the channel's data.
        """
        # Special case for auxillary channels
        if channel == "dldAux":
            # Checks the channel dictionary for correct slices and creates a multicolumn DataFrame
            aux_channels = self._config["channels"]["dldAux"]["dldAuxChannels"].items()
            data_frames = (
                Series(
                    (np_array[i, value] for i in train_id.index),
                    name=key,
                )
                .to_frame()
                .set_index(train_id)
                for key, value in aux_channels
            )

            # Multiindex set and combined dataframe returned
            data = reduce(DataFrame.combine_first, data_frames)

        # For all other pulse resolved channels
        else:
            # Macrobunch resolved data is exploded to a DataFrame and the MultiIndex is set
            # Creates the index_per_pulse for the given channel
            self.create_multi_index_per_pulse(train_id, np_array)
            data = (
                Series((np_array[i] for i in train_id.index), name=channel)
                .explode()
                .to_frame()
                .set_index(self.index_per_pulse)
            )

        return data

    def create_dataframe_per_train(
        self,
        np_array: np.ndarray,
        train_id: Series,
        channel: str,
    ) -> DataFrame:
        """
        Returns a pandas DataFrame for a given channel name of type [per train].

        Args:
            np_array (np.ndarray): The numpy array containing the channel data.
            train_id (Series): The train ID Series.
            channel (str): The name of the channel.

        Returns:
            DataFrame: The pandas DataFrame for the channel's data.
        """
        return (
            Series((np_array[i] for i in train_id.index), name=channel)
            .to_frame()
            .set_index(train_id)
        )

    def create_dataframe_per_channel(
        self,
        h5_file: h5py.File,
        channel: str,
    ) -> DataFrame:
        """
        Returns a pandas DataFrame for a given channel name from a given file.

        This method takes an h5py.File object `h5_file` and a channel name `channel`, and returns
        a pandas DataFrame containing the data for that channel from the file. The format of the
        DataFrame depends on the channel's format specified in the configuration.

        Args:
            h5_file (h5py.File): The h5py.File object representing the HDF5 file.
            channel (str): The name of the channel.

        Returns:
            Series: A pandas DataFrame representing the channel's data.

        Raises:
            ValueError: If the channel has an undefined format.
        """
        [train_id, np_array] = self.create_numpy_array_per_channel(
            h5_file,
            channel,
        )  # numpy Array created
        cformat = self._config["channels"][channel]["format"]  # channel format

        # Electron resolved data is treated here
        if cformat == "per_electron":
            # If index_per_electron is None, create it for the given file
            if self.index_per_electron is None:
                index_pulse, array_pulse = self.create_numpy_array_per_channel(h5_file, "pulseId")
                self.create_multi_index_per_electron(
                    index_pulse,
                    array_pulse,
                    self._config["ubid_offset"],
                )

            # Create a DataFrame for electron-resolved data
            data = self.create_dataframe_per_electron(
                np_array,
                train_id,
                channel,
            )

        # Pulse resolved data is treated here
        elif cformat == "per_pulse":
            # Create a DataFrame for pulse-resolved data
            data = self.create_dataframe_per_pulse(
                np_array,
                train_id,
                channel,
            )

        # Train resolved data is treated here
        elif cformat == "per_train":
            # Create a DataFrame for train-resolved data
            data = self.create_dataframe_per_train(np_array, train_id, channel)

        else:
            raise ValueError(
                f"{channel} has an undefined format",
                "Available formats are per_pulse, per_electron and per_train",
            )

        return data

    def concatenate_channels(
        self,
        h5_file: h5py.File,
    ) -> DataFrame:
        """
        Concatenates the channels from the provided h5py.File into a pandas DataFrame.

        This method takes an h5py.File object `h5_file` and concatenates the channels present in
        the file into a single pandas DataFrame. The concatenation is performed based on the
        available channels specified in the configuration.

        Args:
            h5_file (h5py.File): The h5py.File object representing the HDF5 file.

        Returns:
            DataFrame: A concatenated pandas DataFrame containing the channels.

        Raises:
            ValueError: If the group_name for any channel does not exist in the file.
        """
        all_keys = parse_h5_keys(h5_file)  # Parses all channels present

        # Check for if the provided group_name actually exists in the file
        for channel in self._config["channels"]:
            index_key, dataset_key = self.get_index_dataset_key(channel)

            if index_key or dataset_key not in all_keys:
                raise ValueError(
                    f"The index_key or dataset_key for channel {channel} does not exist.",
                )

        # Create a generator expression to generate data frames for each channel
        data_frames = (
            self.create_dataframe_per_channel(h5_file, each) for each in self.available_channels
        )

        # Use the reduce function to join the data frames into a single DataFrame
        return reduce(
            lambda left, right: left.join(right, how="outer"),
            data_frames,
        )

    def create_dataframe_per_file(
        self,
        file_path: Path,
    ) -> DataFrame:
        """
        Create pandas DataFrames for the given file.

        This method loads an HDF5 file specified by `file_path` and constructs a pandas DataFrame
        from the datasets within the file. The order of datasets in the DataFrames is the opposite
        of the order specified by channel names.

        Args:
            file_path (Path): Path to the input HDF5 file.

        Returns:
            DataFrame: pandas DataFrame
        """
        # Loads h5 file and creates a dataframe
        with h5py.File(file_path, "r") as h5_file:
            self.reset_multi_index()  # Reset MultiIndexes for the next file
            df = self.concatenate_channels(h5_file)
            df = df.dropna(subset=self._config.get("tof_column", "dldTimeSteps"))
            # Correct the 3-bit shift which encodes the detector ID in the 8s time
            if self._config.get("split_sector_id_from_dld_time", False):
                df = split_dld_time_from_sector_id(df, config=self._config)
            return df
