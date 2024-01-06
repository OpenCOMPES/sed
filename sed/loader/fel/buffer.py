"""
The BufferFileHandler class extends the DataFrameCreator class and uses the ParquetHandler class to
manage buffer files. It provides methods for initializing paths, checking the schema of Parquet
files, determining the list of files to read, serializing and parallelizing the creation of buffer
files, and reading all Parquet files into one Dask DataFrame.

Typical usage example:

    buffer_handler = BufferFileHandler(config_dataframe, h5_paths, folder)

Combined DataFrames for electrons and pulses are then available as:
    buffer_handler.electron_dataframe
    buffer_handler.pulse_dataframe
"""
from __future__ import annotations

from itertools import compress
from pathlib import Path

import dask.dataframe as ddf
import pyarrow.parquet as pq
from joblib import delayed
from joblib import Parallel

from sed.core.dfops import forward_fill_lazy
from sed.loader.fel.dataframe import DataFrameCreator
from sed.loader.fel.parquet import ParquetHandler
from sed.loader.fel.utils import get_channels
from sed.loader.utils import split_dld_time_from_sector_id


class BufferFileHandler:
    """
    A class for handling the creation and manipulation of buffer files using DataFrameCreator
    and ParquetHandler.
    """

    def __init__(
        self,
        config_dataframe: dict,
        h5_paths: list[Path],
        folder: Path,
        force_recreate: bool,
        prefix: str = "",
        suffix: str = "",
        debug: bool = False,
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
        self.pq_handler = ParquetHandler(
            [Path(h5_path).stem for h5_path in h5_paths],
            folder,
            "buffer",
            prefix,
            suffix,
            extension="",
        )
        self.buffer_paths = self.pq_handler.parquet_paths
        self.h5_to_create: list[Path] = []
        self.buffer_to_create: list[Path] = []

        self.dataframe_electron: ddf.DataFrame = None
        self.dataframe_pulse: ddf.DataFrame = None

        if not force_recreate:
            self.schema_check()

        self.get_files_to_read(h5_paths, force_recreate)

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
            get_channels(self._config["channels"], formats="all", index=True, extend_aux=True),
        )

        if self._config.get("split_sector_id_from_dld_time", False):
            config_schema.add(self._config.get("sector_id_column", False))

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

    def get_files_to_read(self, h5_paths: list[Path], force_recreate: bool) -> None:
        """
        Determines the list of files to read from the H5 files.

        Args:
            h5_paths (List[Path]): List of paths to H5 files.
            force_recreate (bool): Flag to force recreation of buffer files.
        """
        files_to_read = [
            force_recreate or not parquet_path.exists() for parquet_path in self.buffer_paths
        ]

        self.h5_to_create = list(compress(h5_paths, files_to_read))
        self.buffer_to_create = list(compress(self.buffer_paths, files_to_read))
        self.num_files = len(self.h5_to_create)

        print(f"Reading files: {self.num_files} new files of {len(h5_paths)} total.")

    def create_buffer_files(self, debug: bool) -> None:
        """
        Creates the buffer files.

        Args:
            debug (bool): Flag to enable debug mode, which serializes the creation.
        """
        dataframes: list[ddf.DataFrame] = []
        if self.num_files > 0:
            df_creator_instances = [
                DataFrameCreator(self._config, h5_path) for h5_path in self.h5_to_create
            ]
        if debug:
            dataframes = [df_creator.df for df_creator in df_creator_instances]
        else:
            dataframes = Parallel(n_jobs=self.num_files, verbose=10)(
                delayed(df_creator.df) for df_creator in df_creator_instances
            )
        for df, prq in zip(dataframes, self.buffer_to_create):
            df.to_parquet(prq)

    def get_filled_dataframe(self) -> None:
        """
        Reads all parquet files into one dataframe using dask and fills NaN values.
        """
        dataframe = ddf.read_parquet(self.buffer_paths, calculate_divisions=True)
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
        dataframe_electron = dataframe.dropna(subset=self._config.get(tof_column))

        # Correct the 3-bit shift which encodes the detector ID in the 8s time
        if self._config.get("split_sector_id_from_dld_time", False):
            self.dataframe_electron = split_dld_time_from_sector_id(
                dataframe_electron,
                config=self._config,
            )
        else:
            self.dataframe_electron = dataframe_electron
        self.dataframe_pulse = dataframe[index + channels]
