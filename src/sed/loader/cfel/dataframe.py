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
        # # all values except the last as slow data starts from start of file
        # somehow written something else as this line is doing
        # self.index = np.cumsum([0, *self.get_dataset_array(index_alias)])
        # get cumulative counts, but drop last because slow data only covers N-1 intervals
        self.index = np.cumsum([0, *self.get_dataset_array(index_alias)])[:-1]
        # cumulative sum starting from the first acquisition count, No artificial 0 at the start
        # makes identical len of TimeStamp and index, but cuts last TimeStamp
        # self.index = np.cumsum(self.get_dataset_array(index_alias))
        print(f"len of self.index: {len(self.index)}")

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
        Returns a pandas DataFrame for given channel names of type [per train].

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
                            self.index,# changed together with __init__ line 52
                            # works together with  __init__ line 50, but has different len of TimeStamps and Index
                            # self.index[:-1],
                            name=name,
                        ),
                    )
            else:
                series.append(pd.Series(dataset, self.index, name=channel))# changed together with __init__ line 52
                # works together with  __init__ line 50, but has different len of TimeStamps and Index
                # series.append(pd.Series(dataset, self.index[:-1], name=channel))
        # All the channels are concatenated to a single DataFrame
        return pd.concat(series, axis=1)

    @property
    def df_timestamp(self) -> pd.DataFrame:
        """
        Generates a DataFrame of timestamps for each acquisition point.
    
        - Uses `first_event_time_stamp_key` from the first file as the global StartTime.
        - Uses `millisecCounter` (if available) as a monotonic global time across all files.
        - If `millisecCounter` is not available, uses cumulative exposure times from `ms_markers_key` 
          to approximate acquisition times.
        - Returns timestamps as seconds since the UNIX epoch (1970-01-01).
    
        Returns
        -------
        pd.DataFrame
            DataFrame with a single column containing the computed timestamps.
        """
        # ------------------------------------------------------------
        # 1) Establish global StartTime (absolute origin)
        # ------------------------------------------------------------
        start_time_key = self._config.get("first_event_time_stamp_key")#"/ScanParam/StartTime"
    
        if self.is_first_file:
            if start_time_key not in self.h5_file:
                raise KeyError("StartTime not found in first file")
    
            start_time_raw = self.h5_file[start_time_key][0]
            base_timestamp = pd.to_datetime(start_time_raw.decode())
            logger.warning(f"DEBUG: Taking first file with ScanStart as a timestamp: {base_timestamp}")
    
            # Persist base timestamp for subsequent files
            self.base_timestamp = base_timestamp
        else:
            if self.base_timestamp is None:
                raise RuntimeError("base_timestamp not initialized (first file missing?)")
            base_timestamp = self.base_timestamp
    
        # ------------------------------------------------------------
        # 2) Determine timing offsets
        # ------------------------------------------------------------
        millis_key = self._config.get("millis_counter_key", "/DLD/millisecCounter")
        exposure_key = self._config.get("ms_markers_key")
    
        if millis_key in self.h5_file and len(self.h5_file[millis_key]) > 0:
            # Preferred: global millisecond counter
            offsets = pd.to_timedelta(
                np.asarray(self.h5_file[millis_key], dtype=np.float64),
                unit="ms",
            )
            logger.warning(f"DEBUG: MillisecCounter available, offsets: {offsets}")
    
        elif exposure_key in self.h5_file:
            # Fallback: cumulative exposure time (seconds)
            exposure = np.asarray(self.h5_file[exposure_key], dtype=np.float64)
            offsets = pd.to_timedelta(np.cumsum(exposure), unit="s")
            logger.warning(f"DEBUG: Using cumulative exposure, offsets: {offsets}")
    
        else:
            raise ValueError(
                "Cannot construct timestamps: neither millisecCounter nor exposure times available"
            )
    
        # ------------------------------------------------------------
        # 3) Construct absolute timestamps
        # ------------------------------------------------------------
        timestamps = base_timestamp + offsets
    
        # Convert to UNIX seconds (float)
        unix_seconds = (timestamps - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")
    
        # ------------------------------------------------------------
        # 4) Build DataFrame
        # ------------------------------------------------------------
        ts_alias = self._config["columns"].get("timestamp", "timeStamp")
        df = pd.DataFrame({ts_alias: unix_seconds}, index=self.index)
        print(f"Len of TimeStamps: {len(unix_seconds)}, len of Index: {len(self.index)}")
        pd.set_option("display.float_format", "{:.6f}".format)
        print(df)

        # # # Suppose df is your timestamp DataFrame
        # print("DEBUG of df")
        # ts_alias = "timeStamp"  # or whatever your config uses
        # timestamps = df[ts_alias].to_numpy()
        
        # # Compare lengths
        # if len(timestamps) != len(df.index):
        #     print(f"Length mismatch: timestamps={len(timestamps)}, index={len(df.index)}")
        
        # # Detect NaNs (if any were introduced)
        # nan_rows = df[df[ts_alias].isna()]
        # print("Rows with NaN timestamps (if any):")
        # print(nan_rows)
        
        # # Detect where timestamp differences are huge (likely artificial or missing)
        # dt = np.diff(timestamps)
        # threshold = np.median(dt) * 10  # e.g., 10× median interval
        # anomalous_indices = np.where(dt > threshold)[0]
        # print("Indices where timestamp jump is unusually large:")
        # print(anomalous_indices)
        
        # # Optionally, see these rows in the DataFrame
        # print(df.iloc[anomalous_indices])

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
