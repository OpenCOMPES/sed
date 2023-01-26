"""
This module implements the flash data loader.
The raw hdf5 data is saved into parquet files and loaded as a pandas dataframe.
The class attributes are inherited by dataframeReader - a wrapper class.
"""
import os
from functools import reduce
from itertools import compress
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast
from typing import Union
from typing import List
from typing import Tuple

import dask.dataframe as dd
import h5py
import numpy as np
from pandas import DataFrame
from pandas import MultiIndex
from pandas import Series

from sed.loader.base.loader import BaseLoader
from sed.loader.utils import gather_flash_files, parse_h5_keys

class FlashLoader(BaseLoader):
    """
    The class generates multiindexed multidimensional pandas dataframes
    from the new FLASH dataformat resolved by both macro and microbunches
    alongside electrons.
    """

    __name__ = "flash"

    supported_file_types = ["h5"]

    def __init__(self, config: dict) -> None:

        self._config: dict = config.get("dataframe", {})
        self.config_parser()
        # Set all channels, exluding pulseId as default
        self.all_channels: dict = self._config.get("channels", {})
        self.files: List[Path] = []
        self.index_per_electron: Union[MultiIndex, None] = None
        self.index_per_pulse: Union[MultiIndex, None] = None
        self.parquet_names: List[Path] = []
        self.failed_files_error: List[Path] = []

    def config_parser(self):
        """
        Parser for the config.yaml file.
        """
        paths = self._config.get("paths")

        # Prases to locate the raw beamtime directory from config file
        if paths:
            if 'data_raw_dir' in paths:
                self.data_raw_dir = Path(paths['data_raw_dir'])
            if 'data_parquet_dir' in paths:
                self.data_parquet_dir = Path(paths['data_parquet_dir'])
                if not self.data_parquet_dir.exists():
                    os.mkdir(self.data_parquet_dir)

        if not {'ubid_offset', 'daq'}.issubset(self._config.keys()):
            raise ValueError('One of the values from ubid_offset or daq is missing. \
                        These are necessary.')

        self.ubid_offset = self._config.get("ubid_offset", 0)
        self.daq = self._config.get("daq", "")

        if not paths:
            if not {'beamtime_id', 'year'}.issubset(self._config.keys()):
                raise ValueError('The beamtime_id and year or data_raw_dir is required.')

            beamtime_id = self._config.get("beamtime_id")
            year = self._config.get("year")
            beamtime_dir = Path(f'/asap3/flash/gpfs/pg2/{year}/data/{beamtime_id}/')

            # Folder naming convention till end of October
            self.data_raw_dir = beamtime_dir.joinpath('raw/hdf/express')
            # Use new convention if express doesn't exist
            if not self.data_raw_dir.exists():
                self.data_raw_dir = beamtime_dir.joinpath(f'raw/hdf/{self.daq.upper()}')

            parquet_path = 'processed/parquet'
            self.data_parquet_dir = beamtime_dir.joinpath(parquet_path)

            if not self.data_parquet_dir.exists():
                os.mkdir(self.data_parquet_dir)

    @property
    def available_channels(self) -> List:
        """Returns the channel names that are available for use,
        excluding pulseId, defined by the json file"""
        available_channels = list(self.all_channels.keys())
        available_channels.remove("pulseId")
        return available_channels

    @property
    def channels_per_pulse(self) -> List:
        """Returns a list of channels with per_pulse format,
        including all auxillary channels"""
        channels_per_pulse = []
        for key in self.available_channels:
            if self.all_channels[key]["format"] == "per_pulse":
                if key == "dldAux":
                    for aux_key in self.all_channels[key]["dldAuxChannels"].keys():
                        channels_per_pulse.append(aux_key)
                else:
                    channels_per_pulse.append(key)
        return channels_per_pulse

    @property
    def channels_per_electron(self) -> List:
        """Returns a list of channels with per_electron format"""
        return [
            key
            for key in self.available_channels
            if self.all_channels[key]["format"] == "per_electron"
        ]

    @property
    def channels_per_train(self) -> List:
        """Returns a list of channels with per_train format"""
        return [
            key
            for key in self.available_channels
            if self.all_channels[key]["format"] == "per_train"
        ]

    def reset_multi_index(self) -> None:
        """Resets the index per pulse and electron"""
        self.index_per_electron = None
        self.index_per_pulse = None

    def create_multi_index_per_electron(self, h5_file: h5py.File) -> None:
        """Creates an index per electron using pulseId
        for usage with the electron resolved pandas dataframe"""

        # Macrobunch IDs obtained from the pulseId channel
        [train_id, np_array] = self.create_numpy_array_per_channel(h5_file, "pulseId")

        # Create a series with the macrobunches as index and
        # microbunches as values
        macrobunches = (Series((np_array[i] for i in train_id.index), name="pulseId", index=train_id)
                        - self.ubid_offset)

        # Explode dataframe to get all microbunch vales per macrobunch,
        # remove NaN values and convert to type int
        microbunches = macrobunches.explode().dropna().astype(int)

        # Create temporary index values
        index_temp = MultiIndex.from_arrays(
            (microbunches.index, microbunches.values), names=["trainId", "pulseId"],
        )

        # Calculate the electron counts per pulseId
        # unique preserves the order of appearance
        electron_counts = index_temp.value_counts()[index_temp.unique()].values

        # Series object for indexing with electrons
        electrons = Series(
            [np.arange(electron_counts[i]) for i in range(electron_counts.size)],
        ).explode().astype(int)

        # Create a pandas multiindex using the exploded datasets
        self.index_per_electron = MultiIndex.from_arrays(
            (microbunches.index, microbunches.values, electrons),
            names=["trainId", "pulseId", "electronId"],
        )

    def create_multi_index_per_pulse(self, train_id, np_array) -> None:
        """Creates an index per pulse using a pulse resovled channel's
        macrobunch ID, for usage with the pulse resolved pandas dataframe"""

        # Create a pandas multiindex, useful to compare electron and
        # pulse resolved dataframes
        self.index_per_pulse = MultiIndex.from_product(
            (train_id, np.arange(0, np_array.shape[1])), names=["trainId", "pulseId"],
        )

    def create_numpy_array_per_channel(
        self, h5_file: h5py.File, channel: str,
    ) -> Tuple[Series, np.ndarray]:
        """Returns a numpy Array for a given channel name for a given file"""
        # Get the data from the necessary h5 file and channel
        group = cast(h5py.Group, h5_file[self.all_channels[channel]["group_name"]])
        channel_dict = self.all_channels[channel]  # channel parameters

        train_id = Series(group["index"], name="trainId")  # macrobunch
        # unpacks the timeStamp or value
        if channel == "timeStamp":
            np_array = cast(h5py.Dataset, group["time"])[()]
        else:
            np_array = cast(h5py.Dataset, group["value"])[()]
        np_array = cast(np.ndarray, np_array)
        # Uses predefined axis and slice from the json file
        # to choose correct dimension for necessary channel
        if "axis" in channel_dict:
            np_array = np.take(
                np_array, channel_dict["slice"], axis=channel_dict["axis"],
            )
        return train_id, np_array

    def create_dataframe_per_channel(
        self, h5_file: h5py.File, channel: str,
    ) -> Union[Series, DataFrame]:
        """Returns a pandas DataFrame for a given channel name for
        a given file. The Dataframe contains the MultiIndex and returns
        depending on the channel's format"""
        [train_id, np_array] = self.create_numpy_array_per_channel(
            h5_file, channel,
        )  # numpy Array created
        channel_dict = self.all_channels[channel]  # channel parameters

        # If np_array is size zero, fill with NaNs
        if np_array.size == 0:
            np_array = np.full_like(train_id, np.nan, dtype=np.double)
            return Series(
                (np_array[i] for i in train_id.index), name=channel, index=train_id,
            )

        # Electron resolved data is treated here
        if channel_dict["format"] == "per_electron":
            # Creates the index_per_electron if it does not
            # exist for a given file
            if self.index_per_electron is None:
                self.create_multi_index_per_electron(h5_file)

            # The microbunch resolved data is exploded and
            # converted to dataframe, afterwhich the MultiIndex is set
            # The NaN values are dropped, alongside the
            # pulseId = 0 (meaningless)
            return (
                Series((np_array[i] for i in train_id.index), name=channel)
                .explode()
                .dropna()
                .to_frame()
                .set_index(self.index_per_electron)
                .drop(
                    index=cast(List[int], np.arange(-self.ubid_offset, 0)),
                    level=1,
                    errors="ignore",
                )
            )

        # Pulse resolved data is treated here
        elif channel_dict["format"] == "per_pulse":
            # Special case for auxillary channels which checks the channel
            # dictionary for correct slices and creates a multicolumn
            # pandas dataframe
            if channel == "dldAux":
                # The macrobunch resolved data is repeated 499 times to be
                # comapred to electron resolved data for each auxillary channel
                # and converted to a multicolumn dataframe
                data_frames = (
                    Series(
                        (np_array[i, value] for i in train_id.index),
                        name=key,
                        index=train_id,
                    ).to_frame()
                    for key, value in channel_dict["dldAuxChannels"].items()
                )

                # Multiindex set and combined dataframe returned
                return reduce(DataFrame.combine_first, data_frames)

            else:
                # For all other pulse resolved channels, macrobunch resolved
                # data is exploded to a dataframe and the MultiIndex set

                # Creates the index_per_pulse for the given channel
                self.create_multi_index_per_pulse(train_id, np_array)
                return (
                    Series((np_array[i] for i in train_id.index), name=channel)
                    .explode()
                    .to_frame()
                    .set_index(self.index_per_pulse)
                )

        elif channel_dict["format"] == "per_train":
            return (
                Series((np_array[i] for i in train_id.index), name=channel)
                .to_frame()
                .set_index(train_id)
            )

        else:
            raise ValueError(
                channel
                + "has an undefined format. Available formats are \
                per_pulse, per_electron and per_train",
            )

    def concatenate_channels(
        self, h5_file: h5py.File, format_: str = "",
    ) -> Union[Series, DataFrame]:
        """Returns a concatenated pandas DataFrame for either all pulse,
        train or electron resolved channels."""
        all_keys = parse_h5_keys(h5_file)

        for channel in self.all_channels:
            group_name = self.all_channels[channel]['group_name'] + 'value'
            if group_name not in all_keys:
                raise ValueError(f'The group_name for channel {channel} does not exist.')
        # filters for valid channels
        valid_names = [
            each_name for each_name in self.available_channels if each_name in self.all_channels
        ]
        # Only channels with the defined format are selected and stored
        # in an iterable list
        if format_:
            channels = [
                each_name
                for each_name in valid_names
                if self.all_channels[each_name]["format"] == format_
            ]
        else:
            channels = list(valid_names)

        # if the defined format has channels, returns a concatenatd Dataframe.
        # Otherwise returns empty Dataframe.
        if channels:
            data_frames = (
                self.create_dataframe_per_channel(h5_file, each) for each in channels
            )
            return reduce(
                lambda left, right: left.join(right, how="outer"), data_frames,
            )
        else:
            return DataFrame()

    def create_dataframe_per_file(self, file_path: Path) -> Union[Series, DataFrame]:
        """Returns two pandas DataFrames constructed for the given file.
        The DataFrames contains the datasets from the iterable in the
        order opposite to specified by channel names. One DataFrame is
        pulse resolved and the other electron resolved.
        """
        # Loads h5 file and creates two dataframes
        with h5py.File(file_path, "r") as h5_file:
            self.reset_multi_index()  # Reset MultiIndexes for next file
            return self.concatenate_channels(h5_file)

    def h5_to_parquet(self, h5_path: Path, parquet_path: Path) -> None:
        """Uses the createDataFramePerFile method and saves
        the dataframes to a parquet file."""
        try:
            (
                self.create_dataframe_per_file(h5_path)
                .reset_index(level=["trainId", "pulseId", "electronId"])
                .to_parquet(parquet_path, index=False)
            )
        except ValueError:
            self.failed_files_error.append(parquet_path)
            self.parquet_names.remove(parquet_path)

    def fill_na(self, dataframes: List):
        """Routine to fill the NaN values with intrafile forward filling."""
        # First use forward filling method to fill each file's
        # pulse and train resolved channels.
        channels: List[str] = self.channels_per_pulse + self.channels_per_train
        for i, _ in enumerate(dataframes):
            dataframes[i][channels] = dataframes[i][channels].ffill()

        # This loop forward fills between the consective files.
        # The first run file will have NaNs, unless another run
        # before it has been defined.
        for i in range(1, len(dataframes)):
            # Take only pulse channels
            subset = dataframes[i][channels]
            # Find which column(s) contain NaNs.
            is_null = subset.loc[0].isnull().values.compute()
            # Statement executed if there is more than one NaN value in the
            # first row from all columns
            if is_null.sum() > 0:
                # Select channel names with only NaNs
                channels_to_overwrite = list(compress(channels, is_null[0]))
                # Get the values for those channels from previous file
                values = dataframes[i - 1][channels].tail(1).values[0]
                # Fill all NaNs by those values
                subset[channels_to_overwrite] =subset[channels_to_overwrite].fillna(
                    dict(zip(channels_to_overwrite, values)),
                )
                # Overwrite the dataframes with filled dataframes
                dataframes[i][channels] = subset

        return dd.concat(dataframes)
    
    def parse_metadata(self, files) -> dict:
        """Dummy

        Args:
            files (Sequence[str]): _description_

        Returns:
            dict: _description_
        """
        return {}

    def read_dataframe(
        self, files: Union[List[int], int] = 0, folder: str = "", ftype: str = "h5") -> Tuple[dd.DataFrame, dict]:
        """Read express data from DAQ, generating a parquet in between."""
        # create a per_file directory
        temp_parquet_dir = self.data_parquet_dir.joinpath("per_file")
        if not temp_parquet_dir.exists():
            os.mkdir(temp_parquet_dir)
        
        self.files = files
        runs = files
        folder = str(self.data_raw_dir)
        # Prepare a list of names for the files to read and parquets to write
        try:
            runs = cast(list, runs)
            runs_str = f"Runs {runs[0]} to {runs[-1]}"
        except TypeError:
            runs = cast(int, runs)
            runs_str = f"Run {runs}"
            runs = [runs]
        parquet_name = f"{temp_parquet_dir}/"
        all_files = []
        for run in runs:
            run_ = cast(int, run)
            files_ = gather_flash_files(run_, self.daq, folder, extension = ftype)
            for file in files_:
                all_files.append(file)
            if len(files_) == 0:
                raise FileNotFoundError(f"No file found for run {run}")

        self.parquet_names = [parquet_name + file.stem for file in all_files]
        missing_files: List[Path] = []
        missing_parquet_names: List[Path] = []

        # only read and write files which were not read already
        for i, _ in enumerate(self.parquet_names):
            if not Path(self.parquet_names[i]).exists():
                missing_files.append(all_files[i])
                missing_parquet_names.append(Path(self.parquet_names[i]))

        print(
                f"Reading {runs_str}: {len(missing_files)} new files of "
                f"{len(all_files)} total.",
        )
        self.failed_files_error = []

        self.reset_multi_index()  # initializes the indices for h5_to_parquet

        # Read missing files
        if len(missing_files) > 0:
            # with ThreadPoolExecutor() as executor:
            #     for h5_path, parquet_path in zip(missing_files, missing_parquet_names):
            #         executor.submit(self.h5_to_parquet, h5_path, parquet_path)
            # for param in zip(missing_files, missing_parquet_names):
            #     self.h5_to_parquet(*param)
        
            for h5_path, parquet_path in zip(missing_files, missing_parquet_names):
                self.h5_to_parquet(h5_path, parquet_path)

        if len(self.failed_files_error) > 0:
            print(
                    f"Failed reading {len(self.failed_files_error)}"
                    f"files of{len(all_files)}:",
            )
            for failed_string in self.failed_files_error:
                print(f"\t- Failed to read {failed_string}")
        if len(self.parquet_names) == 0:
            raise ValueError("No data available. Probably failed reading all h5 files")

        print(
                f"Loading {len(self.parquet_names)} dataframes. Failed reading "
                f"{len(all_files)-len(self.parquet_names)} files.",
        )
        # Read all parquet files using dask and concatenate into one dataframe after filling
        
        dataframe = self.fill_na([dd.read_parquet(parquet_file) for parquet_file in self.parquet_names])
        dataframe = dataframe.dropna(subset=self.channels_per_electron)
        # pulse_columns = (
        #     ["trainId", "pulseId", "electronId"]
        #     + self.channels_per_pulse
        #     + self.channels_per_train
        # )
        # df_pulse = dataframe[pulse_columns]
        # df_pulse = df_pulse[
        #     (df_pulse["electronId"] == 0) | (np.isnan(df_pulse["electronId"]))
        # ]

        # dataframe = df_electron.repartition(npartitions=len(self.parquet_names))

        metadata = self.parse_metadata(files=all_files)

        return dataframe, metadata

LOADER = FlashLoader
