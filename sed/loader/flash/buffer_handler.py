from __future__ import annotations

import os
from itertools import compress
from pathlib import Path

import dask.dataframe as dd
import pyarrow.parquet as pq
from joblib import delayed
from joblib import Parallel

from sed.core.dfops import forward_fill_lazy
from sed.loader.flash.dataframe import DataFrameCreator
from sed.loader.flash.utils import get_channels
from sed.loader.flash.utils import initialize_paths
from sed.loader.utils import get_parquet_metadata
from sed.loader.utils import split_dld_time_from_sector_id


class BufferHandler:
    """
    A class for handling the creation and manipulation of buffer files using DataFrameCreator.
    """

    def __init__(
        self,
        config: dict,
    ) -> None:
        """
        Initializes the BufferHandler.

        Args:
            config (dict): The configuration dictionary.
        """
        self._config = config["dataframe"]
        self.n_cores = config["core"].get("num_cores", os.cpu_count() - 1)

        self.buffer_paths: list[Path] = []
        self.missing_h5_files: list[Path] = []
        self.save_paths: list[Path] = []

        self.df_electron: dd.DataFrame = None
        self.df_pulse: dd.DataFrame = None
        self.metadata: dict = {}

    def _schema_check(self) -> None:
        """
        Checks the schema of the Parquet files.

        Raises:
            ValueError: If the schema of the Parquet files does not match the configuration.
        """
        existing_parquet_filenames = [file for file in self.buffer_paths if file.exists()]
        parquet_schemas = [pq.read_schema(file) for file in existing_parquet_filenames]
        config_schema_set = set(
            get_channels(self._config["channels"], formats="all", index=True, extend_aux=True),
        )

        for filename, schema in zip(existing_parquet_filenames, parquet_schemas):
            # for retro compatibility when sectorID was also saved in buffer
            if self._config["sector_id_column"] in schema.names:
                config_schema_set.add(
                    self._config["sector_id_column"],
                )
            schema_set = set(schema.names)
            if schema_set != config_schema_set:
                missing_in_parquet = config_schema_set - schema_set
                missing_in_config = schema_set - config_schema_set

                errors = []
                if missing_in_parquet:
                    errors.append(f"Missing in parquet: {missing_in_parquet}")
                if missing_in_config:
                    errors.append(f"Missing in config: {missing_in_config}")

                raise ValueError(
                    f"The available channels do not match the schema of file {filename}. "
                    f"{' '.join(errors)}. "
                    "Please check the configuration file or set force_recreate to True.",
                )

    def _get_files_to_read(
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
        self.buffer_paths = initialize_paths(
            filenames=[h5_path.stem for h5_path in h5_paths],
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

        print(f"Reading files: {len(self.missing_h5_files)} new files of {len(h5_paths)} total.")

    def _save_buffer_file(self, h5_path: Path, parquet_path: Path) -> None:
        """
        Creates a single buffer file.

        Args:
            h5_path (Path): Path to the H5 file.
            parquet_path (Path): Path to the buffer file.
        """

        # Create a DataFrameCreator instance and the h5 file
        df = DataFrameCreator(config_dataframe=self._config, h5_path=h5_path).df

        # Reset the index of the DataFrame and save it as a parquet file
        df.reset_index().to_parquet(parquet_path)

    def _save_buffer_files(self, debug: bool) -> None:
        """
        Creates the buffer files.

        Args:
            debug (bool): Flag to enable debug mode, which serializes the creation.
        """
        n_cores = min(len(self.missing_h5_files), self.n_cores)
        paths = zip(self.missing_h5_files, self.save_paths)
        if n_cores > 0:
            if debug:
                for h5_path, parquet_path in paths:
                    self._save_buffer_file(h5_path, parquet_path)
            else:
                Parallel(n_jobs=n_cores, verbose=10)(
                    delayed(self._save_buffer_file)(h5_path, parquet_path)
                    for h5_path, parquet_path in paths
                )

    def _fill_dataframes(self):
        """
        Reads all parquet files into one dataframe using dask and fills NaN values.
        """
        dataframe = dd.read_parquet(self.buffer_paths, calculate_divisions=True)
        file_metadata = get_parquet_metadata(
            self.buffer_paths,
            time_stamp_col=self._config.get("time_stamp_alias", "timeStamp"),
        )
        self.metadata["file_statistics"] = file_metadata

        fill_channels: list[str] = get_channels(
            self._config["channels"],
            ["per_pulse", "per_train"],
            extend_aux=True,
        )
        index: list[str] = get_channels(index=True)
        overlap = min(file["num_rows"] for file in file_metadata.values())

        dataframe = forward_fill_lazy(
            df=dataframe,
            columns=fill_channels,
            before=overlap,
            iterations=self._config.get("forward_fill_iterations", 2),
        )
        self.metadata["forward_fill"] = {
            "columns": fill_channels,
            "overlap": overlap,
            "iterations": self._config.get("forward_fill_iterations", 2),
        }

        # Drop rows with nan values in electron channels
        df_electron = dataframe.dropna(
            subset=get_channels(self._config["channels"], ["per_electron"]),
        )

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
            df_electron, meta = split_dld_time_from_sector_id(
                df_electron,
                config=self._config,
            )
            self.metadata.update(meta)

        self.df_electron = df_electron.astype(dtypes)
        self.df_pulse = dataframe[index + fill_channels]

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

        self._get_files_to_read(h5_paths, folder, prefix, suffix, force_recreate)

        if not force_recreate:
            self._schema_check()

        self._save_buffer_files(debug)

        self._fill_dataframes()
