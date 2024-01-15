"""
The BufferFileHandler uses the DataFrameCreator class and uses the ParquetHandler class to
manage buffer files. It provides methods for initializing paths, checking the schema,
determining the list of files to read, serializing and parallelizing the creation, and reading
all files into one Dask DataFrame.

After initialization, the electron and timed dataframes can be accessed as:

    buffer_handler = BufferFileHandler(config, h5_paths, folder)

    buffer_handler.electron_dataframe
    buffer_handler.pulse_dataframe

Force_recreate flag forces recreation of buffer files. Useful when the schema has changed.
Debug mode serializes the creation of buffer files.
"""
from __future__ import annotations

from itertools import compress
from pathlib import Path
from typing import Type

import dask.dataframe as ddf
import h5py
import pyarrow.parquet as pq
from joblib import delayed
from joblib import Parallel

from sed.core.dfops import forward_fill_lazy
from sed.loader.fel.config_model import DataFrameConfig
from sed.loader.fel.dataframe import DataFrameCreator
from sed.loader.fel.parquet import ParquetHandler
from sed.loader.fel.utils import get_channels
from sed.loader.utils import split_dld_time_from_sector_id


class BufferHandler:
    """
    A class for handling the creation and manipulation of buffer files using DataFrameCreator
    and ParquetHandler.
    """

    def __init__(
        self,
        df_creator: type[DataFrameCreator],
        config: DataFrameConfig,
        h5_paths: list[Path],
        folder: Path,
        force_recreate: bool = False,
        prefix: str = "",
        suffix: str = "",
        debug: bool = False,
        auto: bool = True,
    ) -> None:
        """
        Initializes the BufferFileHandler.

        Args:
            df_creator (Type[DataFrameCreator]): Derived class based on DataFrameCreator.
            config (DataFrameConfig): The dataframe section of the config model.
            h5_paths (List[Path]): List of paths to H5 files.
            folder (Path): Path to the folder for buffer files.
            force_recreate (bool): Flag to force recreation of buffer files.
            prefix (str): Prefix for buffer file names.
            suffix (str): Suffix for buffer file names.
            debug (bool): Flag to enable debug mode.
            auto (bool): Flag to automatically create buffer files and fill the dataframe.
        """
        self.df_creator = df_creator
        self._config = config

        self.buffer_paths: list[Path] = []
        self.h5_to_create: list[Path] = []
        self.buffer_to_create: list[Path] = []

        self.dataframe_electron: ddf.DataFrame = None
        self.dataframe_pulse: ddf.DataFrame = None

        # In auto mode, these methods are called automatically
        if auto:
            self.get_files_to_read(h5_paths, folder, prefix, suffix, force_recreate)

            if not force_recreate:
                self.schema_check()

            self.create_buffer_files(debug)

            self.get_filled_dataframe()

    def schema_check(self) -> None:
        """
        Checks the schema of the Parquet files.

        Raises:
            ValueError: If the schema of the Parquet files does not match the configuration.
        """
        existing_parquet_filenames = [file for file in self.buffer_paths if file.exists()]
        parquet_schemas = [pq.read_schema(file) for file in existing_parquet_filenames]
        config_schema = set(
            get_channels(self._config.channels, formats="all", index=True, extend_aux=True),
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
        pq_handler = ParquetHandler(
            [Path(h5_path).stem for h5_path in h5_paths],
            folder,
            "buffer",
            prefix,
            suffix,
            extension="",
        )
        self.buffer_paths = pq_handler.parquet_paths
        # read only the files that do not exist or if force_recreate is True
        files_to_read = [
            force_recreate or not parquet_path.exists() for parquet_path in self.buffer_paths
        ]

        # Get the list of H5 files to read and the corresponding buffer files to create
        self.h5_to_create = list(compress(h5_paths, files_to_read))
        self.buffer_to_create = list(compress(self.buffer_paths, files_to_read))

        self.num_files = len(self.h5_to_create)

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
        dfc = self.df_creator(self._config, h5_file)

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
        if self.num_files > 0:
            if debug:
                for h5_path, parquet_path in zip(self.h5_to_create, self.buffer_to_create):
                    self._create_buffer_file(h5_path, parquet_path)
            else:
                Parallel(n_jobs=self.num_files, verbose=10)(
                    delayed(self._create_buffer_file)(h5_path, parquet_path)
                    for h5_path, parquet_path in zip(self.h5_to_create, self.buffer_to_create)
                )

    def get_filled_dataframe(self) -> None:
        """
        Reads all parquet files into one dataframe using dask and fills NaN values.
        """
        dataframe = ddf.read_parquet(self.buffer_paths, calculate_divisions=True)
        metadata = [pq.read_metadata(file) for file in self.buffer_paths]

        channels: list[str] = get_channels(
            self._config.channels,
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
            iterations=self._config.forward_fill_iterations,
        )

        # Drop rows with nan values in the tof column
        dataframe_electron = dataframe.dropna(subset=self._config.tof_column)

        # Set the dtypes of the channels here as there should be no null values
        ch_names = get_channels(self._config.channels, "all")
        cfg_ch = self._config.channels
        dtypes = {
            channel: cfg_ch[channel].dtype
            for channel in ch_names
            if cfg_ch[channel].dtype is not None
        }

        # Correct the 3-bit shift which encodes the detector ID in the 8s time
        if self._config.split_sector_id_from_dld_time:
            dataframe_electron = split_dld_time_from_sector_id(
                dataframe_electron,
                self._config.tof_column,
                self._config.sector_id_column,
                self._config.sector_id_reserved_bits,
            )
        self.dataframe_electron = dataframe_electron.astype(dtypes)
        self.dataframe_pulse = dataframe[index + channels]
