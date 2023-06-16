"""
This module implements the flash data loader.
The raw hdf5 data is saved into parquet files and loaded as a dask dataframe.
If there are multiple files, the NaNs are forward filled.
"""
import os
from functools import reduce
from itertools import compress
from pathlib import Path
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import dask.dataframe as dd
import h5py
import numpy as np
from joblib import delayed
from joblib import Parallel
from natsort import natsorted
from pandas import DataFrame
from pandas import MultiIndex
from pandas import Series

from sed.loader.base.loader import BaseLoader
from sed.loader.flash.metadata import MetadataRetriever
from sed.loader.utils import parse_h5_keys


class FlashLoader(BaseLoader):
    """
    The class generates multiindexed multidimensional pandas dataframes
    from the new FLASH dataformat resolved by both macro and microbunches
    alongside electrons.
    """

    __name__ = "flash"

    supported_file_types = ["h5"]

    def __init__(self, config: dict) -> None:

        super().__init__(config=config)
        self.index_per_electron: MultiIndex = None
        self.index_per_pulse: MultiIndex = None
        self.parquet_names: List[Path] = []
        self.failed_files_error: List[str] = []

    @property
    def available_channels(self) -> List:
        """Returns the channel names that are available for use,
        excluding pulseId, defined by the json file"""
        available_channels = list(self._config["dataframe"]["channels"].keys())
        available_channels.remove("pulseId")
        return available_channels

    def get_channels_by_format(self, formats: List[str]) -> List:
        """
        Returns a list of channels with the specified format.

        Args:
            formats (List[str]): The desired formats ('per_pulse', 'per_electron',
                or 'per_train').

        Returns:
            List: A list of channels with the specified format(s).
        """
        channels = []
        for format_ in formats:
            for key in self.available_channels:
                channel_format = self._config["dataframe"]["channels"][key][
                    "format"
                ]
                if channel_format == format_:
                    if key == "dldAux":
                        aux_channels = self._config["dataframe"]["channels"][
                            key
                        ]["dldAuxChannels"].keys()
                        channels.extend(aux_channels)
                    else:
                        channels.append(key)
        return channels

    def reset_multi_index(self) -> None:
        """Resets the index per pulse and electron"""
        self.index_per_electron = None
        self.index_per_pulse = None

    def create_multi_index_per_electron(self, h5_file: h5py.File) -> None:
        """
        Creates an index per electron using pulseId for usage with the electron
            resolved pandas DataFrame.

        Args:
            h5_file (h5py.File): The HDF5 file object.

        Notes:
            - This method relies on the 'pulseId' channel to determine
                the macrobunch IDs.
            - It creates a MultiIndex with trainId, pulseId, and electronId
                as the index levels.
        """

        # Macrobunch IDs obtained from the pulseId channel
        [train_id, np_array] = self.create_numpy_array_per_channel(
            h5_file,
            "pulseId",
        )

        # Create a series with the macrobunches as index and
        # microbunches as values
        macrobunches = (
            Series(
                (np_array[i] for i in train_id.index),
                name="pulseId",
                index=train_id,
            )
            - self._config["dataframe"]["ubid_offset"]
        )

        # Explode dataframe to get all microbunch vales per macrobunch,
        # remove NaN values and convert to type int
        microbunches = macrobunches.explode().dropna().astype(int)

        # Create temporary index values
        index_temp = MultiIndex.from_arrays(
            (microbunches.index, microbunches.values),
            names=["trainId", "pulseId"],
        )

        # Calculate the electron counts per pulseId
        # unique preserves the order of appearance
        electron_counts = index_temp.value_counts()[index_temp.unique()].values

        # Series object for indexing with electrons
        electrons = (
            Series(
                [
                    np.arange(electron_counts[i])
                    for i in range(electron_counts.size)
                ],
            )
            .explode()
            .astype(int)
        )

        # Create a pandas MultiIndex using the exploded datasets
        self.index_per_electron = MultiIndex.from_arrays(
            (microbunches.index, microbunches.values, electrons),
            names=["trainId", "pulseId", "electronId"],
        )

    def create_multi_index_per_pulse(
        self,
        train_id: Series,
        np_array: np.ndarray,
    ) -> None:
        """
        Creates an index per pulse using a pulse resolved channel's macrobunch ID,
        for usage with the pulse resolved pandas DataFrame.

        Args:
            train_id (Series): The train ID Series.
            np_array (np.ndarray): The numpy array containing the pulse resolved data.

        Notes:
            - This method creates a MultiIndex with trainId and pulseId as the
                index levels.
        """

        # Create a pandas MultiIndex, useful for comparing electron and
        # pulse resolved dataframes
        self.index_per_pulse = MultiIndex.from_product(
            (train_id, np.arange(0, np_array.shape[1])),
            names=["trainId", "pulseId"],
        )

    def create_numpy_array_per_channel(
        self,
        h5_file: h5py.File,
        channel: str,
    ) -> Tuple[Series, np.ndarray]:
        """
        Returns a numpy array for a given channel name for a given file.

        Args:
            h5_file (h5py.File): The h5py file object.
            channel (str): The name of the channel.

        Returns:
            Tuple[Series, np.ndarray]: A tuple containing the train ID Series
            and the numpy array for the channel's data.

        """
        # Get the data from the necessary h5 file and channel
        group = h5_file[
            self._config["dataframe"]["channels"][channel]["group_name"]
        ]

        channel_dict = self._config["dataframe"]["channels"][
            channel
        ]  # channel parameters

        train_id = Series(group["index"], name="trainId")  # macrobunch

        # unpacks the timeStamp or value
        if channel == "timeStamp":
            np_array = group["time"][()]
        else:
            np_array = group["value"][()]

        # Use predefined axis and slice from the json file
        # to choose correct dimension for necessary channel
        if "slice" in channel_dict:
            np_array = np.take(
                np_array,
                channel_dict["slice"],
                axis=1,
            )
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

        Notes:
            The microbunch resolved data is exploded and converted to a DataFrame.
            The MultiIndex is set, and the NaN values are dropped, alongside the
            pulseId = 0 (meaningless).

        """
        return (
            Series((np_array[i] for i in train_id.index), name=channel)
            .explode()
            .dropna()
            .to_frame()
            .set_index(self.index_per_electron)
            .drop(
                index=np.arange(-self._config["dataframe"]["ubid_offset"], 0),
                level=1,
                errors="ignore",
            )
        )

    def create_dataframe_per_pulse(
        self,
        np_array: np.ndarray,
        train_id: Series,
        channel: str,
        channel_dict: dict,
    ) -> DataFrame:
        """
        Returns a pandas DataFrame for a given channel name of type [per pulse].

        Args:
            np_array (np.ndarray): The numpy array containing the channel data.
            train_id (Series): The train ID Series.
            channel (str): The name of the channel.
            channel_dict (dict): The dictionary containing channel parameters.

        Returns:
            DataFrame: The pandas DataFrame for the channel's data.

        Notes:
            - For auxillary channels, the macrobunch resolved data is repeated 499
              times to be compared to electron resolved data for each auxillary
              channel. The data is then converted to a multicolumn DataFrame.
            - For all other pulse resolved channels, the macrobunch resolved
              data is exploded to a DataFrame and the MultiIndex is set.

        """

        # Special case for auxillary channels
        if channel == "dldAux":
            # Checks the channel dictionary for correct slices and creates a
            # multicolumn DataFrame
            data_frames = (
                Series(
                    (np_array[i, value] for i in train_id.index),
                    name=key,
                    index=train_id,
                ).to_frame()
                for key, value in channel_dict["dldAuxChannels"].items()
            )

            # Multiindex set and combined dataframe returned
            data = reduce(DataFrame.combine_first, data_frames)

        # For all other pulse resolved channels
        else:
            # Macrobunch resolved data is exploded to a DataFrame and
            # the MultiIndex is set

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
    ) -> Union[Series, DataFrame]:
        """
        Returns a pandas DataFrame for a given channel name from a given file.

        This method takes an h5py.File object `h5_file` and a channel name `channel`,
        and returns a pandas DataFrame containing the data for that channel from the
        file. The format of the DataFrame depends on the channel's format specified
        in the configuration.

        Args:
            h5_file (h5py.File): The h5py.File object representing the HDF5 file.
            channel (str): The name of the channel.

        Returns:
            Union[Series, DataFrame]: A pandas Series or DataFrame representing the
            channel's data.

        Raises:
            ValueError: If the channel has an undefined format.

        """
        [train_id, np_array] = self.create_numpy_array_per_channel(
            h5_file,
            channel,
        )  # numpy Array created
        channel_dict = self._config["dataframe"]["channels"][
            channel
        ]  # channel parameters

        # If np_array is size zero, fill with NaNs
        if np_array.size == 0:
            # Fill the np_array with NaN values of the same shape as train_id
            np_array = np.full_like(train_id, np.nan, dtype=np.double)
            # Create a Series using np_array, with train_id as the index
            data = Series(
                (np_array[i] for i in train_id.index),
                name=channel,
                index=train_id,
            )

        # Electron resolved data is treated here
        if channel_dict["format"] == "per_electron":
            # If index_per_electron is None, create it for the given file
            if self.index_per_electron is None:
                self.create_multi_index_per_electron(h5_file)

            # Create a DataFrame for electron-resolved data
            data = self.create_dataframe_per_electron(
                np_array,
                train_id,
                channel,
            )

        # Pulse resolved data is treated here
        elif channel_dict["format"] == "per_pulse":
            # Create a DataFrame for pulse-resolved data
            data = self.create_dataframe_per_pulse(
                np_array,
                train_id,
                channel,
                channel_dict,
            )

        # Train resolved data is treated here
        elif channel_dict["format"] == "per_train":
            # Create a DataFrame for train-resolved data
            data = self.create_dataframe_per_train(np_array, train_id, channel)

        else:
            raise ValueError(
                channel
                + "has an undefined format. Available formats are \
                per_pulse, per_electron and per_train",
            )

        return data

    def concatenate_channels(
        self,
        h5_file: h5py.File,
    ) -> DataFrame:
        """
        Concatenates the channels from the provided h5py.File into a pandas DataFrame.

        This method takes an h5py.File object `h5_file` and concatenates the channels
        present in the file into a single pandas DataFrame. The concatenation is
        performed based on the available channels specified in the configuration.

        Args:
            h5_file (h5py.File): The h5py.File object representing the HDF5 file.

        Returns:
            DataFrame: A concatenated pandas DataFrame containing the channels.

        Raises:
            ValueError: If the group_name for any channel does not exist in the file.

        """
        all_keys = parse_h5_keys(h5_file)  # Parses all channels present

        # Check for if the provided group_name actually exists in the file
        for channel in self._config["dataframe"]["channels"]:
            if channel == "timeStamp":
                group_name = (
                    self._config["dataframe"]["channels"][channel][
                        "group_name"
                    ]
                    + "time"
                )
            else:
                group_name = (
                    self._config["dataframe"]["channels"][channel][
                        "group_name"
                    ]
                    + "value"
                )

            if group_name not in all_keys:
                raise ValueError(
                    f"The group_name for channel {channel} does not exist.",
                )

        # Create a generator expression to generate data frames for each channel
        data_frames = (
            self.create_dataframe_per_channel(h5_file, each)
            for each in self.available_channels
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

        This method loads an HDF5 file specified by `file_path` and constructs a pandas
        DataFrame from the datasets within the file. The order of datasets in the
        DataFrames is the opposite of the order specified by channel names.

        Args:
            file_path (Path): Path to the input HDF5 file.

        Returns:
            DataFrame: pandas DataFrame

        """
        # Loads h5 file and creates a dataframe
        with h5py.File(file_path, "r") as h5_file:
            self.reset_multi_index()  # Reset MultiIndexes for next file
            return self.concatenate_channels(h5_file)

    def h5_to_parquet(self, h5_path: Path, parquet_path: Path) -> None:
        """
        Convert HDF5 file to Parquet format.

        This method uses the `create_dataframe_per_file` method to create dataframes
        from individual files within an HDF5 file. The resulting dataframe is then
        saved to a Parquet file.

        Args:
            h5_path (Path): Path to the input HDF5 file.
            parquet_path (Path): Path to the output Parquet file.

        Raises:
            ValueError: If an error occurs during the conversion process.

        """
        try:
            (
                self.create_dataframe_per_file(h5_path)
                .reset_index(level=["trainId", "pulseId", "electronId"])
                .to_parquet(parquet_path, index=False)
            )
        except ValueError as failed_string_error:
            self.failed_files_error.append(
                f"{parquet_path}: {failed_string_error}",
            )
            self.parquet_names.remove(parquet_path)

    def fill_na(
        self,
        dataframes: List[dd.DataFrame],
    ) -> dd.DataFrame:
        """
        Fill NaN values in the given dataframes using intrafile forward filling.

        Args:
            dataframes (List[dd.DataFrame]): List of dataframes to fill NaN values.

        Returns:
            dd.DataFrame: Concatenated dataframe with filled NaN values.

        Notes:
            This method is specific to the flash data structure and is used to fill NaN
            values in certain channels that only store information at a lower frequency
            The low frequency channels are exploded to match the dimensions of higher
            frequency channels, but they may contain NaNs in the other columns. This
            method fills the NaNs for the specific channels (per_pulse and per_train).

        """
        # Channels to fill NaN values
        channels: List[str] = self.get_channels_by_format(
            ["per_pulse", "per_train"],
        )

        # Fill NaN values within each dataframe
        for i, _ in enumerate(dataframes):
            dataframes[i][channels] = dataframes[i][channels].fillna(
                method="ffill",
            )

        # Forward fill between consecutive dataframes
        for i in range(1, len(dataframes)):
            # Select pulse channels from current dataframe
            subset = dataframes[i][channels]
            # Find columns with NaN values in the first row
            is_null = subset.loc[0].isnull().values.compute()
            # Execute if there are NaN values in the first row
            if is_null.sum() > 0:
                # Select channel names with only NaNs
                channels_to_overwrite = list(compress(channels, is_null[0]))
                # Get values for those channels from the previous dataframe
                values = dataframes[i - 1][channels].tail(1).values[0]
                # Create a dictionary to fill NaN values
                fill_dict = dict(zip(channels, values))
                fill_dict = {
                    k: v
                    for k, v in fill_dict.items()
                    if k in channels_to_overwrite
                }
                # Fill NaN values with the corresponding values from the
                # previous dataframe
                dataframes[i][channels_to_overwrite] = subset[
                    channels_to_overwrite
                ].fillna(fill_dict)

        # Concatenate the filled dataframes
        return dd.concat(dataframes)

    def parse_metadata(
        self,
        files: Sequence[str],  # pylint: disable=unused-argument
    ) -> dict:
        """Dummy

        Args:
            files (Sequence[str]): _description_

        Returns:
            dict: _description_
        """
        return {}

    def get_count_rate(
        self,
        fids: Sequence[int] = None,
        **kwds,
    ):
        return None, None

    def get_elapsed_time(self, fids=None, **kwds):
        return None

    def read_dataframe(
        self,
        files: Union[str, Sequence[str]] = None,
        folders: Union[str, Sequence[str]] = None,
        runs: Union[str, Sequence[str]] = None,
        ftype: str = "h5",
        metadata: dict = None,
        collect_metadata: bool = False,
        **kwds,
    ) -> Tuple[dd.DataFrame, dict]:
        """
        Read express data from the DAQ, generating a parquet in between.

        Args:
            files (Union[str, Sequence[str]], optional): File path(s) to process.
                Defaults to None.
            folders (Union[str, Sequence[str]], optional): Path to folder(s) where files
                are stored. Path has priority such that if it's specified, the specified
                files will be ignored. Defaults to None.
            runs (Union[str, Sequence[str]], optional): Run identifier(s). Corresponding
                files will be located in the location provided by ``folders``. Takes
                precendence over ``files`` and ``folders``. Defaults to None.
            ftype (str, optional): The file extension type. Defaults to "h5".
            metadata (dict, optional): Additional metadata. Defaults to None.
            collect_metadata (bool, optional): Whether to collect metadata.
                Defaults to False.


        Returns:
            Tuple[dd.DataFrame, dict]: A tuple containing the concatenated DataFrame
            and metadata.

        Raises:
            ValueError: If neither 'runs' nor 'files'/'data_raw_dir' is provided.
            FileNotFoundError: If the conversion fails for some files or no
            data is available.
        """

        data_raw_dir, data_parquet_dir = self.initialize_paths()

        # Create a per_file directory
        temp_parquet_dir = data_parquet_dir.joinpath("per_file")
        os.makedirs(temp_parquet_dir, exist_ok=True)

        # Prepare a list of names for the runs to read and parquets to write

        if runs is not None:
            files = []
            if isinstance(runs, (str, int)):
                runs = [runs]
            for run in runs:
                run_files = self.get_files_from_run_id(
                    run_id=run,
                    folders=[str(folder.resolve()) for folder in data_raw_dir],
                    extension=ftype,
                    daq=self._config["dataframe"]["daq"],
                )
                files.extend(run_files)
            self.runs = list(runs)
            super().read_dataframe(files=files, ftype=ftype)

        else:
            # This call takes care of files and folders. As we have converted runs
            # into files already, they are just stored in the class by this call.
            super().read_dataframe(
                files=files,
                folders=folders,
                ftype=ftype,
                metadata=metadata,
            )

        parquet_name = f"{temp_parquet_dir}/"
        self.parquet_names = [
            Path(parquet_name + Path(file).stem) for file in self.files
        ]
        missing_files: List[Path] = []
        missing_parquet_names: List[Path] = []

        # Only read and write files which were not read already
        for i, parquet_file in enumerate(self.parquet_names):
            if not parquet_file.exists():
                missing_files.append(Path(self.files[i]))
                missing_parquet_names.append(parquet_file)

        print(
            f"Reading files: {len(missing_files)} new files of {len(self.files)} total.",
        )

        self.reset_multi_index()  # Initializes the indices for h5_to_parquet

        # Run self.h5_to_parquet in parallel
        if len(missing_files) > 0:
            Parallel(n_jobs=len(missing_files), verbose=10)(
                delayed(self.h5_to_parquet)(h5_path, parquet_path)
                for h5_path, parquet_path in zip(
                    missing_files,
                    missing_parquet_names,
                )
            )

        if self.failed_files_error:
            raise FileNotFoundError(
                "Conversion failed for the following files: \n"
                + "\n".join(self.failed_files_error),
            )

        print("All files converted successfully!")

        if len(self.parquet_names) == 0:
            raise ValueError(
                "No data available. Probably failed reading all h5 files",
            )

        print(
            f"Loading {len(self.parquet_names)} dataframes. Failed reading "
            f"{len(self.files)-len(self.parquet_names)} files.",
        )
        # Read all parquet files using dask and concatenate into one dataframe
        # after filling
        dataframe = self.fill_na(
            [
                dd.read_parquet(parquet_file)
                for parquet_file in self.parquet_names
            ],
        )
        dataframe = dataframe.dropna(
            subset=self.get_channels_by_format(["per_electron"]),
        )

        if collect_metadata:
            metadata_retriever = MetadataRetriever(self._config["metadata"])
            metadata = metadata_retriever.get_metadata(
                beamtime_id=self._config["dataframe"]["beamtime_id"],
                runs=list(runs),
                metadata=self.metadata,
            )
        else:
            metadata = self.metadata

        return dataframe, metadata

    def get_files_from_run_id(
        self,
        run_id: str,
        folders: Union[str, Sequence[str]] = None,
        extension: str = "h5",
        **kwds,
    ) -> List[str]:
        """Returns a list of filenames for a given run located in the specified directory
        for the specified data acquisition (daq).

        Args:
            run_id (str): The run identifier to locate.
            folders (Union[str, Sequence[str]], optional): The directory(ies) where the raw
                data is located. Defaults to config["core"]["base_folder"].
            extension (str, optional): The file extension. Defaults to "h5".
            kwds: Keyword arguments:
                - daq (str): The data acquisition identifier.
                  Defaults to config["dataframe"]["daq"].

        Returns:
            List[str]: A list of path strings representing the collected file names.

        Raises:
            FileNotFoundError: If no files are found for the given run in the directory.
        """
        # Define the stream name prefixes based on the data acquisition identifier
        stream_name_prefixes = self._config["dataframe"][
            "stream_name_prefixes"
        ]

        if folders is None:
            folders = self._config["core"]["base_folder"]

        if isinstance(folders, str):
            folders = [folders]

        daq = kwds.pop("daq", self._config.get("dataframe", {}).get("daq"))

        # Generate the file patterns to search for in the directory
        file_pattern = (
            f"{stream_name_prefixes[daq]}_run{run_id}_*." + extension
        )

        files: List[Path] = []
        # Use pathlib to search for matching files in each directory
        for folder in folders:
            files.extend(
                natsorted(
                    Path(folder).glob(file_pattern),
                    key=lambda filename: str(filename).rsplit("_", maxsplit=1)[-1],
                ),
            )

        # Check if any files are found
        if not files:
            raise FileNotFoundError(
                f"No files found for run {run_id} in directory {str(folders)}",
            )

        # Return the list of found files
        return [str(file.resolve()) for file in files]

    def initialize_paths(self) -> Tuple[List[Path], Path]:
        """
        Initializes the paths based on the configuration.

        Returns:
            Tuple[List[Path], Path]: A tuple containing a list of raw data directories
            paths and the parquet data directory path.

        Raises:
            ValueError: If required values are missing from the configuration.
            FileNotFoundError: If the raw data directories are not found.
        """
        # Parses to locate the raw beamtime directory from config file
        if "paths" in self._config["core"]:
            data_raw_dir = [
                Path(self._config["core"]["paths"].get("data_raw_dir", "")),
            ]
            data_parquet_dir = Path(
                self._config["core"]["paths"].get("data_parquet_dir", ""),
            )

        else:
            try:
                beamtime_id = self._config["core"]["beamtime_id"]
                year = self._config["core"]["year"]
                daq = self._config["dataframe"]["daq"]
            except KeyError as exc:
                raise ValueError(
                    "The beamtime_id, year and daq are required.",
                ) from exc

            beamtime_dir = Path(
                self._config["dataframe"]["beamtime_dir"][
                    self._config["loader"]["instrument"]
                ],
            )
            beamtime_dir = beamtime_dir.joinpath(f"{year}/data/{beamtime_id}/")

            # Use os walk to reach the raw data directory
            data_raw_dir = []
            for root, dirs, files in os.walk(beamtime_dir.joinpath("raw/")):  # pylint: disable=W0612
                for dir_name in dirs:
                    if dir_name.startswith("express-") or dir_name.startswith(
                        "online-",
                    ):
                        data_raw_dir.append(Path(root, dir_name, daq))
                    elif dir_name == daq.upper():
                        data_raw_dir.append(Path(root, dir_name))

            if not data_raw_dir:
                raise FileNotFoundError("Raw data directories not found.")

            parquet_path = "processed/parquet"
            data_parquet_dir = beamtime_dir.joinpath(parquet_path)

        # TODO: This will fail of more than one level of directories needs to be created...
        if not data_parquet_dir.exists():
            os.mkdir(data_parquet_dir)

        return data_raw_dir, data_parquet_dir


LOADER = FlashLoader
