"""
This module implements the flash data loader.
This loader currently supports hextof, wespe and instruments with similar structure.
The raw hdf5 data is combined and saved into buffer files and loaded as a dask dataframe.
The dataframe is an amalgamation of all h5 files for a combination of runs, where the NaNs are
automatically forward-filled across different files.
This can then be saved as a parquet for out-of-sed processing and reread back to access other
sed functionality.
"""
from __future__ import annotations

import os
import time
from itertools import compress
from pathlib import Path
from typing import Sequence

import dask.dataframe as dd
import h5py
import numpy as np
import pyarrow.parquet as pq
from joblib import delayed
from joblib import Parallel
from natsort import natsorted
from pandas import concat
from pandas import DataFrame
from pandas import Index
from pandas import MultiIndex
from pandas import Series

from sed.core.dfops import forward_fill_lazy
from sed.loader.base.loader import BaseLoader
from sed.loader.flash.metadata import MetadataRetriever
from sed.loader.flash.utils import get_channels
from sed.loader.flash.utils import initialize_parquet_paths
from sed.loader.utils import split_dld_time_from_sector_id


class FlashLoader(BaseLoader):
    """
    The class generates multiindexed multidimensional pandas dataframes from the new FLASH
    dataformat resolved by both macro and microbunches alongside electrons.
    Only the read_dataframe (inherited and implemented) method is accessed by other modules.
    """

    __name__ = "flash"

    supported_file_types = ["h5"]

    def __init__(self, config: dict) -> None:
        """
        Initializes the FlashLoader.

        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__(config=config)

    def initialize_dir(self) -> tuple[list[Path], Path]:
        """
        Initializes the directories based on the configuration. If paths is provided in the
        configuration, the raw data directories and parquet data directory are taken from there.
        Otherwise, the beamtime_id and year are used to locate the data directories.

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
                self._config["dataframe"]["beamtime_dir"][self._config["core"]["beamline"]],
            )
            beamtime_dir = beamtime_dir.joinpath(f"{year}/data/{beamtime_id}/")

            # Use pathlib walk to reach the raw data directory
            data_raw_dir = []
            raw_path = beamtime_dir.joinpath("raw")

            for path in raw_path.glob("**/*"):
                if path.is_dir():
                    dir_name = path.name
                    if dir_name.startswith("express-") or dir_name.startswith(
                        "online-",
                    ):
                        data_raw_dir.append(path.joinpath(daq))
                    elif dir_name == daq.upper():
                        data_raw_dir.append(path)

            if not data_raw_dir:
                raise FileNotFoundError("Raw data directories not found.")

            parquet_path = "processed/parquet"
            data_parquet_dir = beamtime_dir.joinpath(parquet_path)

        data_parquet_dir.mkdir(parents=True, exist_ok=True)

        return data_raw_dir, data_parquet_dir

    def get_files_from_run_id(
        self,
        run_id: str,
        folders: str | Sequence[str] = None,
        extension: str = "h5",
        **kwds,
    ) -> list[str]:
        """
        Returns a list of filenames for a given run located in the specified directory
        for the specified data acquisition (daq).

        Args:
            run_id (str): The run identifier to locate.
            folders (Union[str, Sequence[str]], optional): The directory(ies) where the raw
                data is located. Defaults to config["core"]["base_folder"].
            extension (str, optional): The file extension. Defaults to "h5".
            kwds: Keyword arguments:
                - daq (str): The data acquisition identifier.

        Returns:
            List[str]: A list of path strings representing the collected file names.

        Raises:
            FileNotFoundError: If no files are found for the given run in the directory.
        """
        # Define the stream name prefixes based on the data acquisition identifier
        stream_name_prefixes = self._config["dataframe"]["stream_name_prefixes"]

        if folders is None:
            folders = self._config["core"]["base_folder"]

        if isinstance(folders, str):
            folders = [folders]

        daq = kwds.pop("daq", self._config.get("dataframe", {}).get("daq"))

        # Generate the file patterns to search for in the directory
        file_pattern = f"{stream_name_prefixes[daq]}_run{run_id}_*." + extension

        files: list[Path] = []
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

    def parse_metadata(self, scicat_token: str = None) -> dict:
        """Uses the MetadataRetriever class to fetch metadata from scicat for each run.

        Returns:
            dict: Metadata dictionary
            scicat_token (str, optional):: The scicat token to use for fetching metadata
        """
        metadata_retriever = MetadataRetriever(self._config["metadata"], scicat_token)
        metadata = metadata_retriever.get_metadata(
            beamtime_id=self._config["core"]["beamtime_id"],
            runs=self.runs,
            metadata=self.metadata,
        )

        return metadata

    def get_count_rate(
        self,
        fids: Sequence[int] = None,  # noqa: ARG002
        **kwds,  # noqa: ARG002
    ):
        return None, None

    def get_elapsed_time(self, fids=None, **kwds):  # noqa: ARG002
        return None

    def read_parquet(self, parquet_paths: Sequence[Path] = None):
        dfs = []
        for parquet_path in parquet_paths:
            if not parquet_path.exists():
                raise FileNotFoundError(
                    f"The Parquet file at {parquet_path} does not exist. ",
                    "If it is in another location, provide the correct path as parquet_path.",
                )
            dfs.append(dd.read_parquet(parquet_path))
        return dfs

    def read_dataframe(
        self,
        files: str | Sequence[str] = None,
        folders: str | Sequence[str] = None,
        runs: str | Sequence[str] = None,
        ftype: str = "h5",
        metadata: dict = None,
        collect_metadata: bool = False,
        converted: bool = False,
        load_parquet: bool = False,
        save_parquet: bool = False,
        detector: str = "",
        force_recreate: bool = False,
        parquet_dir: str | Path = None,
        debug: bool = False,
        **kwds,
    ) -> tuple[dd.DataFrame, dd.DataFrame, dict]:
        """
        Read express data from the DAQ, generating a parquet in between.

        Args:
            files (Union[str, Sequence[str]], optional): File path(s) to process. Defaults to None.
            folders (Union[str, Sequence[str]], optional): Path to folder(s) where files are stored
                Path has priority such that if it's specified, the specified files will be ignored.
                Defaults to None.
            runs (Union[str, Sequence[str]], optional): Run identifier(s). Corresponding files will
                be located in the location provided by ``folders``. Takes precedence over
                ``files`` and ``folders``. Defaults to None.
            ftype (str, optional): The file extension type. Defaults to "h5".
            metadata (dict, optional): Additional metadata. Defaults to None.
            collect_metadata (bool, optional): Whether to collect metadata. Defaults to False.

        Returns:
            Tuple[dd.DataFrame, dd.DataFrame, dict]: A tuple containing the concatenated DataFrame
                and metadata.

        Raises:
            ValueError: If neither 'runs' nor 'files'/'data_raw_dir' is provided.
            FileNotFoundError: If the conversion fails for some files or no data is available.
        """
        t0 = time.time()

        data_raw_dir, data_parquet_dir = self.initialize_dir()

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
            # This call takes care of files and folders. As we have converted runs into files
            # already, they are just stored in the class by this call.
            super().read_dataframe(
                files=files,
                folders=folders,
                ftype=ftype,
                metadata=metadata,
            )

        # if parquet_dir is None, use data_parquet_dir
        parquet_dir = parquet_dir or data_parquet_dir
        parquet_dir = Path(parquet_dir)
        filename = "_".join(str(run) for run in self.runs)
        converted_str = "converted" if converted else ""

        # Create parquet paths for saving and loading the parquet files of df and timed_df
        parquet_paths = initialize_parquet_paths(
            parquet_names=[filename, filename + "_timed"],
            folder=parquet_dir,
            subfolder=converted_str,
            prefix="run_",
            suffix=detector,
        )

        # Check if load_parquet is flagged and then load the file if it exists
        if load_parquet:
            df, df_timed = self.read_parquet(parquet_paths)

        # Default behavior is to create the buffer files and load them
        else:
            # Obtain the parquet filenames, metadata, and schema from the method
            # which handles buffer file creation/reading
            h5_paths = [Path(file) for file in self.files]
            buffer = BufferHandler(
                config_dataframe=self._config["dataframe"],
            )
            buffer.run(
                h5_paths=h5_paths,
                folder=parquet_dir,
                force_recreate=force_recreate,
                suffix=detector,
                debug=debug,
            )
            df = buffer.dataframe_electron
            df_timed = buffer.dataframe_pulse

        # Save the dataframe as parquet if requested
        if save_parquet:
            df.compute().reset_index(drop=True).to_parquet(parquet_paths[0])
            df_timed.compute().reset_index(drop=True).to_parquet(parquet_paths[1])

        metadata = self.parse_metadata(**kwds) if collect_metadata else {}
        print(f"loading complete in {time.time() - t0: .2f} s")

        return df, df_timed, metadata


class DataFrameCreator:
    """
    Utility class for creating pandas DataFrames from HDF5 files with multiple channels.
    """

    def __init__(self, config_dataframe: dict, h5_file: h5py.File) -> None:
        """
        Initializes the DataFrameCreator class.

        Args:
            config_dataframe (dict): The configuration dictionary with only the dataframe key.
            h5_file (h5py.File): The open h5 file.
        """
        self.h5_file: h5py.File = h5_file
        self.failed_files_error: list[str] = []
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
        if "index_key" in channel_config and "dataset_key" in channel_config:
            return channel_config["index_key"], channel_config["dataset_key"]

        raise ValueError(
            "For channel:",
            channel,
            "Provide either both 'index_key' and 'dataset_key'.",
            "or 'group_name' (parses only 'index' and 'value' or 'time' keys.)",
        )

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
        index_key, dataset_key = self.get_index_dataset_key(channel)

        key = Index(self.h5_file[index_key], name="trainId")  # macrobunch
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
        offset = self._config["ubid_offset"]
        # Index
        index, indexer = self.pulse_index(offset)

        # Data logic
        channels = get_channels(self._config["channels"], "per_electron")
        slice_index = [self._config["channels"][channel].get("slice", None) for channel in channels]

        # First checking if dataset keys are the same for all channels
        dataset_keys = [self.get_index_dataset_key(channel)[1] for channel in channels]
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
            dataframe.dropna()
            .iloc[indexer]
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
        channels = get_channels(self._config["channels"], "per_pulse")
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

        channels = get_channels(self._config["channels"], "per_train")

        for channel in channels:
            key, dataset = self.get_dataset_array(channel, slice_=True)
            index = MultiIndex.from_product(
                (key, [0], [0]),
                names=self.multi_index,
            )
            if channel == "dldAux":
                aux_channels = self._config["channels"]["dldAux"]["dldAuxChannels"].items()
                for name, slice_aux in aux_channels:
                    series.append(Series(dataset[: key.size, slice_aux], index, name=name))
            else:
                series.append(Series(dataset, index, name=channel))

        return concat(series, axis=1)

    def validate_channel_keys(self) -> None:
        """
        Validates if the index and dataset keys for all channels in config exist in the h5 file.

        Raises:
            KeyError: If the index or dataset keys do not exist in the file.
        """
        for channel in self._config["channels"]:
            index_key, dataset_key = self.get_index_dataset_key(channel)
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


class BufferHandler:
    """
    A class for handling the creation and manipulation of buffer files using DataFrameCreator
    and ParquetHandler.
    """

    def __init__(
        self,
        config_dataframe: dict,
    ) -> None:
        """
        Initializes the BufferFileHandler.

        Args:
            config_dataframe (dict): The configuration dictionary with only the dataframe key.
            h5_paths (List[Path]): List of paths to H5 files.
            folder (Path): Path to the folder for buffer files.
            force_recreate (bool): Flag to force recreation of buffer files.
            prefix (str): Prefix for buffer file names.
            suffix (str): Suffix for buffer file names.
            debug (bool): Flag to enable debug mode.
        """
        self._config = config_dataframe

        self.buffer_paths: list[Path] = []
        self.missing_h5_files: list[Path] = []
        self.save_paths: list[Path] = []

        self.dataframe_electron: dd.DataFrame = None
        self.dataframe_pulse: dd.DataFrame = None

    def schema_check(self) -> None:
        """
        Checks the schema of the Parquet files.

        Raises:
            ValueError: If the schema of the Parquet files does not match the configuration.
        """
        existing_parquet_filenames = [file for file in self.buffer_paths if file.exists()]
        parquet_schemas = [pq.read_schema(file) for file in existing_parquet_filenames]
        config_schema = set(
            get_channels(self._config["channels"], formats="all", index=True, extend_aux=True),
        )

        for i, schema in enumerate(parquet_schemas):
            schema_set = set(schema.names)
            if schema_set != config_schema:
                missing_in_parquet = config_schema - schema_set
                missing_in_config = schema_set - config_schema

                missing_in_parquet_str = (
                    f"Missing in parquet: {missing_in_parquet}" if missing_in_parquet else ""
                )
                missing_in_config_str = (
                    f"Missing in config: {missing_in_config}" if missing_in_config else ""
                )

                raise ValueError(
                    "The available channels do not match the schema of file",
                    f"{existing_parquet_filenames[i]}",
                    f"{missing_in_parquet_str}",
                    f"{missing_in_config_str}",
                    "Please check the configuration file or set force_recreate to True.",
                )

    def get_files_to_read(
        self,
        h5_paths: list[Path],
        folder: Path,
        prefix: str,
        suffix: str,
        force_recreate: bool,
    ) -> None:
        """
        Determines the list of files to read and the corresponding buffer files to create.

        Args:
            h5_paths (List[Path]): List of paths to H5 files.
            folder (Path): Path to the folder for buffer files.
            prefix (str): Prefix for buffer file names.
            suffix (str): Suffix for buffer file names.
            force_recreate (bool): Flag to force recreation of buffer files.
        """
        # Getting the paths of the buffer files, with subfolder as buffer and no extension
        self.buffer_paths = initialize_parquet_paths(
            parquet_names=[Path(h5_path).stem for h5_path in h5_paths],
            folder=folder,
            subfolder="buffer",
            prefix=prefix,
            suffix=suffix,
            extension="",
        )
        # read only the files that do not exist or if force_recreate is True
        files_to_read = [
            force_recreate or not parquet_path.exists() for parquet_path in self.buffer_paths
        ]

        # Get the list of H5 files to read and the corresponding buffer files to create
        self.missing_h5_files = list(compress(h5_paths, files_to_read))
        self.save_paths = list(compress(self.buffer_paths, files_to_read))

        self.num_files = len(self.missing_h5_files)

        print(f"Reading files: {self.num_files} new files of {len(h5_paths)} total.")

    def _create_buffer_file(self, h5_path: Path, parquet_path: Path) -> None:
        """
        Creates a single buffer file. Useful because h5py.File cannot be pickled if left open.

        Args:
            h5_path (Path): Path to the H5 file.
            parquet_path (Path): Path to the buffer file.
        """
        # Open the h5 file in read mode
        h5_file = h5py.File(h5_path, "r")

        # Create a DataFrameCreator instance with the configuration and the h5 file
        dfc = DataFrameCreator(config_dataframe=self._config, h5_file=h5_file)

        # Get the DataFrame from the DataFrameCreator instance
        df = dfc.df

        # Close the h5 file
        h5_file.close()

        # Reset the index of the DataFrame and save it as a parquet file
        df.reset_index().to_parquet(parquet_path)

    def create_buffer_files(self, debug: bool) -> None:
        """
        Creates the buffer files.

        Args:
            debug (bool): Flag to enable debug mode, which serializes the creation.
        """
        # make sure to not create more jobs than cores available
        # TODO: This value should be taken from the configuration
        n_cores = min(self.num_files, os.cpu_count() - 1)
        if n_cores > 0:
            if debug:
                for h5_path, parquet_path in zip(self.missing_h5_files, self.save_paths):
                    self._create_buffer_file(h5_path, parquet_path)
            else:
                Parallel(n_jobs=n_cores, verbose=10)(
                    delayed(self._create_buffer_file)(h5_path, parquet_path)
                    for h5_path, parquet_path in zip(self.missing_h5_files, self.save_paths)
                )

    def fill_dataframes(self) -> None:
        """
        Reads all parquet files into one dataframe using dask and fills NaN values.
        """
        dataframe = dd.read_parquet(self.buffer_paths, calculate_divisions=True)
        metadata = [pq.read_metadata(file) for file in self.buffer_paths]

        channels: list[str] = get_channels(
            self._config["channels"],
            ["per_pulse", "per_train"],
            extend_aux=True,
        )
        index: list[str] = get_channels(index=True)
        overlap = min(file.num_rows for file in metadata)

        print("Filling nan values...")
        dataframe = forward_fill_lazy(
            df=dataframe,
            columns=channels,
            before=overlap,
            iterations=self._config.get("forward_fill_iterations", 2),
        )

        # Drop rows with nan values in the tof column
        tof_column = self._config.get("tof_column", "dldTimeSteps")
        dataframe_electron = dataframe.dropna(subset=tof_column)

        # Set the dtypes of the channels here as there should be no null values
        channel_dtypes = get_channels(self._config["channels"], "all")
        config_channels = self._config["channels"]
        dtypes = {
            channel: config_channels[channel].get("dtype")
            for channel in channel_dtypes
            if config_channels[channel].get("dtype") is not None
        }

        # Correct the 3-bit shift which encodes the detector ID in the 8s time
        if self._config.get("split_sector_id_from_dld_time", False):
            dataframe_electron = split_dld_time_from_sector_id(
                dataframe_electron,
                config=self._config,
            )
        self.dataframe_electron = dataframe_electron.astype(dtypes)
        self.dataframe_pulse = dataframe[index + channels]

    def run(
        self,
        h5_paths: list[Path],
        folder: Path,
        force_recreate: bool = False,
        prefix: str = "",
        suffix: str = "",
        debug: bool = False,
    ) -> None:
        """
        Runs the buffer file creation process.

        Args:
            h5_paths (List[Path]): List of paths to H5 files.
            folder (Path): Path to the folder for buffer files.
            force_recreate (bool): Flag to force recreation of buffer files.
            prefix (str): Prefix for buffer file names.
            suffix (str): Suffix for buffer file names.
            debug (bool): Flag to enable debug mode.):
        """

        self.get_files_to_read(h5_paths, folder, prefix, suffix, force_recreate)

        if not force_recreate:
            self.schema_check()

        self.create_buffer_files(debug)

        self.fill_dataframes()


LOADER = FlashLoader
