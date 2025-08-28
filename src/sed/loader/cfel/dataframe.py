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

from sed.core.logging import setup_logging
from sed.loader.flash.utils import get_channels
from sed.loader.flash.utils import InvalidFileError

logger = setup_logging("cfel_dataframe_creator")


class DataFrameCreator:
    """
    A class for creating pandas DataFrames from an HDF5 file for HEXTOF lab data at CFEL.

    Attributes:
        h5_file (h5py.File): The HDF5 file object.
        multi_index (pd.MultiIndex): The multi-index structure for the DataFrame.
        _config (dict): The configuration dictionary for the DataFrame.
    """

    def __init__(self, config_dataframe: dict, h5_path: Path, 
                 is_first_file: bool = True, base_timestamp: pd.Timestamp = None) -> None:
        """
        Initializes the DataFrameCreator class.

        Args:
            config_dataframe (dict): The configuration dictionary with only the dataframe key.
            h5_path (Path): Path to the h5 file.
            is_first_file (bool): Whether this is the first file in a multi-file run.
            base_timestamp (pd.Timestamp): Base timestamp from the first file (for subsequent files).
        """
        self.h5_file = h5py.File(h5_path, "r")
        self._config = config_dataframe
        self.is_first_file = is_first_file
        self.base_timestamp = base_timestamp

        index_alias = self._config.get("index", ["countId"])[0]
        # all values except the last as slow data starts from start of file
        self.index = np.cumsum([0, *self.get_dataset_array(index_alias)])

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
            h5py.Dataset: The channel's data as a h5py.Dataset object.
        """
        # Get the data from the necessary h5 file and channel
        dataset_key = self.get_dataset_key(channel)
        dataset = self.h5_file[dataset_key]

        return dataset

    def get_base_timestamp(self) -> pd.Timestamp:
        """
        Extracts the base timestamp from the first file to be used for subsequent files.
        
        Returns:
            pd.Timestamp: The base timestamp from the first file.
        """
        if not self.is_first_file:
            raise ValueError("get_base_timestamp() should only be called on the first file")
        
        first_timestamp = self.h5_file[self._config.get("first_event_time_stamp_key")][0]
        return pd.to_datetime(first_timestamp.decode())

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
        # auxiliary dataset (which is stored in the same dataset as other DLD channels)
        aux_alias = self._config.get("aux_alias", "dldAux")

        # For each channel, a pd.Series is created and appended to the list
        for channel in channels:
            dataset = self.get_dataset_array(channel)

            if channel == aux_alias:
                try:
                    sub_channels = self._config["channels"][aux_alias]["sub_channels"]
                except KeyError:
                    raise KeyError(
                        f"Provide 'sub_channels' for auxiliary channel '{aux_alias}'.",
                    )
                for name, values in sub_channels.items():
                    series.append(
                        pd.Series(
                            dataset[:, values["slice"]],
                            self.index[:-1],
                            name=name,
                        ),
                    )
            else:
                series.append(pd.Series(dataset, self.index[:-1], name=channel))
        # All the channels are concatenated to a single DataFrame
        return pd.concat(series, axis=1)

    @property
    def df_timestamp(self) -> pd.DataFrame:
        """
        For files with first_event_time_stamp_key: Uses that as initial timestamp.
        For files with only millis_counter_key: Uses that as absolute timestamp.
        Both use ms_markers_key for exposure times within the file.
        """

        # Try to determine which timestamp approach to use based on available data
        first_timestamp_key = self._config.get("first_event_time_stamp_key")
        millis_counter_key = self._config.get("millis_counter_key", "/DLD/millisecCounter")
        
        has_first_timestamp = (first_timestamp_key is not None and 
                             first_timestamp_key in self.h5_file and 
                             len(self.h5_file[first_timestamp_key]) > 0)
        
        has_millis_counter = (millis_counter_key in self.h5_file and 
                            len(self.h5_file[millis_counter_key]) > 0)

        # Log millisecond counter values for ALL files
        if has_millis_counter:
            millis_counter_values = self.h5_file[millis_counter_key][()]

        if has_first_timestamp:
            logger.warning("DEBUG: Taking first file with scan start timestamp path")
            # First file with scan start timestamp
            first_timestamp = self.h5_file[first_timestamp_key][0]
            base_ts = pd.to_datetime(first_timestamp.decode())
            
            # Check if we also have millisecond counter for more precise timing
            if has_millis_counter:
                millis_counter_values = self.h5_file[millis_counter_key][()]
                millis_min = millis_counter_values[0]   # First value
                millis_max = millis_counter_values[-1]  # Last value

                # Add the first millisecond counter value to the base timestamp
                ts_start = base_ts + pd.Timedelta(milliseconds=millis_min)   
            else:
                # Use base timestamp directly if no millisecond counter
                ts_start = base_ts   

        elif not self.is_first_file and self.base_timestamp is not None and has_millis_counter:
            # Subsequent files: use base timestamp + millisecond counter offset
            millis_counter_values = self.h5_file[millis_counter_key][()]  # Get all values
            
            # Get min (first) and max (last) millisecond values
            millis_min = millis_counter_values[0]   # First value
            millis_max = millis_counter_values[-1]  # Last value
            
            # Calculate timestamps for min and max
            ts_min = self.base_timestamp + pd.Timedelta(milliseconds=millis_min)
            ts_max = self.base_timestamp + pd.Timedelta(milliseconds=millis_max)
            
            logger.warning(f"DEBUG: Timestamp for min: {ts_min}")
            logger.warning(f"DEBUG: Timestamp for max: {ts_max}")
            
            # Use the first value (start time) for calculating offset
            millis_counter = millis_counter_values[0]  # First element is the start time
            offset = pd.Timedelta(milliseconds=millis_counter)
            ts_start = self.base_timestamp + offset
        else:
            try:
                start_time_key = "/ScanParam/StartTime"  
                if start_time_key in self.h5_file:
                    start_time = self.h5_file[start_time_key][0]
                    ts_start = pd.to_datetime(start_time.decode())
                    logger.warning(f"DEBUG: Using fallback startTime: {ts_start}")
                else:
                    raise KeyError(f"startTime key '{start_time_key}' not found in file")
            except (KeyError, IndexError, AttributeError) as e:
                raise ValueError(
                    f"Cannot determine timestamp: no valid timestamp source found. Error: {e}"
                ) from e

        # Get exposure times (in seconds) for this file
        exposure_time = self.h5_file[self._config.get("ms_markers_key")][()]

        # Calculate cumulative exposure times
        cumulative_exposure = np.cumsum(exposure_time)
        timestamps = [ts_start + pd.Timedelta(seconds=cum_exp) for cum_exp in cumulative_exposure]
        # add initial timestamp to the start of the list
        timestamps.insert(0, ts_start)

        timestamps = [(ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s") for ts in timestamps]
        # Create a DataFrame with the timestamps
        ts_alias = self._config["columns"].get("timestamp")
        df = pd.DataFrame({ts_alias: timestamps}, index=self.index)
        return df

    def validate_channel_keys(self) -> None:
        """
        Validates if the dataset keys for all channels in the config exist in the h5 file.

        Raises:
            InvalidFileError: If the dataset keys are missing in the h5 file.
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
        Joins the 'per_electron', 'per_pulse' using concat operation,
        returning a single dataframe.

        Returns:
            pd.DataFrame: The combined pandas DataFrame.
        """

        self.validate_channel_keys()
        df_train = self.df_train
        df_timestamp = self.df_timestamp
        df = pd.concat((self.df_electron, df_train, df_timestamp), axis=1)
        ffill_cols = list(df_train.columns) + list(df_timestamp.columns)
        df[ffill_cols] = df[ffill_cols].ffill()
        df.index.name = self._config.get("index", ["countId"])[0]
        return df

    @property
    def df_timed(self) -> pd.DataFrame:
        """
        Joins the 'per_electron', 'per_pulse' using concat operation,
        returning a single dataframe.

        Returns:
            pd.DataFrame: The combined pandas DataFrame.
        """

        self.validate_channel_keys()
        df_train = self.df_train
        df_timestamp = self.df_timestamp
        df = pd.concat((self.df_electron, df_train, df_timestamp), axis=1, join="inner")
        df.index.name = self._config.get("index", ["countId"])[0]
        return df
