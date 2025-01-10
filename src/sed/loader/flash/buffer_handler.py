from __future__ import annotations

import os
from pathlib import Path
import time

import dask.dataframe as dd
import pyarrow.parquet as pq
from joblib import delayed
from joblib import Parallel

from sed.core.dfops import forward_fill_lazy
from sed.core.logging import setup_logging
from sed.loader.flash.dataframe import DataFrameCreator
from sed.loader.flash.utils import get_channels
from sed.loader.flash.utils import get_dtypes
from sed.loader.flash.utils import InvalidFileError
from sed.loader.utils import get_parquet_metadata
from sed.loader.utils import split_dld_time_from_sector_id


DF_TYP = ["electron", "timed"]

logger = setup_logging("flash_buffer_handler")


class BufferFilePaths:
    """
    A class for handling the paths to the raw and buffer files of electron and timed dataframes.
    A list of file sets (dict) are created for each H5 file containing the paths to the raw file
    and the electron and timed buffer files.

    Structure of the file sets:
    {
        "raw": Path to the H5 file,
        "electron": Path to the electron buffer file,
        "timed": Path to the timed buffer file,
    }
    """

    def __init__(
        self,
        config: dict,
        h5_paths: list[Path],
        folder: Path,
        suffix: str,
        remove_invalid_files: bool,
    ) -> None:
        """Initializes the BufferFilePaths.

        Args:
            h5_paths (list[Path]): List of paths to the H5 files.
            folder (Path): Path to the folder for processed files.
            suffix (str): Suffix for buffer file names.
        """
        suffix = f"_{suffix}" if suffix else ""
        folder = folder / "buffer"
        folder.mkdir(parents=True, exist_ok=True)

        if remove_invalid_files:
            h5_paths = self.remove_invalid_files(config, h5_paths)

        self._file_paths = self._create_file_paths(h5_paths, folder, suffix)

    def _create_file_paths(
        self,
        h5_paths: list[Path],
        folder: Path,
        suffix: str,
    ) -> list[dict[str, Path]]:
        return [
            {
                "raw": h5_path,
                **{typ: folder / f"{typ}_{h5_path.stem}{suffix}" for typ in DF_TYP},
            }
            for h5_path in h5_paths
        ]

    def __getitem__(self, key) -> list[Path]:
        if isinstance(key, str):
            return [file_set[key] for file_set in self._file_paths]
        return self._file_paths[key]

    def __iter__(self):
        return iter(self._file_paths)

    def __len__(self):
        return len(self._file_paths)

    def file_sets_to_process(self, force_recreate: bool = False) -> list[dict[str, Path]]:
        """Returns a list of file sets that need to be processed."""
        if force_recreate:
            return self._file_paths
        return [file_set for file_set in self if any(not file_set[key].exists() for key in DF_TYP)]

    def remove_invalid_files(self, config, h5_paths: list[Path]) -> list[Path]:
        valid_h5_paths = []
        for h5_path in h5_paths:
            try:
                dfc = DataFrameCreator(config_dataframe=config, h5_path=h5_path)
                dfc.validate_channel_keys()
                valid_h5_paths.append(h5_path)
            except InvalidFileError as e:
                logger.info(f"Skipping invalid file: {h5_path.stem}\n{e}")

        return valid_h5_paths


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
        self._config: dict = config["dataframe"]
        self.n_cores: int = config["core"].get("num_cores", os.cpu_count() - 1)
        self.fp: BufferFilePaths = None
        self.df: dict[str, dd.DataFrame] = {typ: None for typ in DF_TYP}
        self.fill_channels: list[str] = get_channels(
            self._config,
            ["per_pulse", "per_train"],
            extend_aux=True,
        )
        self.metadata: dict = {}
        self.filter_timed_by_electron: bool = None

    def _schema_check(self, files: list[Path], expected_schema_set: set) -> None:
        """
        Checks the schema of the Parquet files.
        """
        logger.debug(f"Checking schema for {len(files)} files")
        existing = [file for file in files if file.exists()]
        parquet_schemas = [pq.read_schema(file) for file in existing]

        for filename, schema in zip(existing, parquet_schemas):
            schema_set = set(schema.names)
            if schema_set != expected_schema_set:
                logger.error(f"Schema mismatch in file: {filename}")
                missing_in_parquet = expected_schema_set - schema_set
                missing_in_config = schema_set - expected_schema_set

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
        logger.debug("Schema check passed successfully")

    def _create_timed_dataframe(self, df: dd.DataFrame) -> dd.DataFrame:
        """Creates the timed dataframe, optionally filtering by electron events.

        Args:
            df (dd.DataFrame): The input dataframe containing all data

        Returns:
            dd.DataFrame: The timed dataframe
        """
        # Get channels that should be in timed dataframe
        timed_channels = self.fill_channels

        if self.filter_timed_by_electron:
            # Get electron channels to use for filtering
            electron_channels = get_channels(self._config, "per_electron")
            # Filter rows where electron data exists
            df_timed = df.dropna(subset=electron_channels)[timed_channels]
        else:
            # Take all timed data rows without filtering
            df_timed = df[timed_channels]

        # Take only first electron per event
        return df_timed.loc[:, :, 0]

    def _save_buffer_file(self, paths: dict[str, Path]) -> None:
        """Creates the electron and timed buffer files from the raw H5 file."""
        logger.debug(f"Processing file: {paths['raw'].stem}")
        start_time = time.time()
        # Create DataFrameCreator and get get dataframe
        df = DataFrameCreator(config_dataframe=self._config, h5_path=paths["raw"]).df

        # Forward fill non-electron channels
        logger.debug(f"Forward filling {len(self.fill_channels)} channels")
        df[self.fill_channels] = df[self.fill_channels].ffill()

        # Save electron resolved dataframe
        electron_channels = get_channels(self._config, "per_electron")
        dtypes = get_dtypes(self._config, df.columns.values)
        electron_df = df.dropna(subset=electron_channels).astype(dtypes).reset_index()
        logger.debug(f"Saving electron buffer with shape: {electron_df.shape}")
        electron_df.to_parquet(paths["electron"])

        # Create and save timed dataframe
        df_timed = self._create_timed_dataframe(df)
        dtypes = get_dtypes(self._config, df_timed.columns.values)
        timed_df = df_timed.astype(dtypes).reset_index()
        logger.debug(f"Saving timed buffer with shape: {timed_df.shape}")
        timed_df.to_parquet(paths["timed"])

        logger.debug(f"Processed {paths['raw'].stem} in {time.time() - start_time:.2f}s")

    def _save_buffer_files(self, force_recreate: bool, debug: bool) -> None:
        """
        Creates the buffer files that are missing.

        Args:
            force_recreate (bool): Flag to force recreation of buffer files.
            debug (bool): Flag to enable debug mode, which serializes the creation.
        """
        file_sets = self.fp.file_sets_to_process(force_recreate)
        logger.info(f"Reading files: {len(file_sets)} new files of {len(self.fp)} total.")
        n_cores = min(len(file_sets), self.n_cores)
        if n_cores > 0:
            if debug:
                for file_set in file_sets:
                    self._save_buffer_file(file_set)
            else:
                Parallel(n_jobs=n_cores, verbose=10)(
                    delayed(self._save_buffer_file)(file_set) for file_set in file_sets
                )

    def _get_dataframes(self) -> None:
        """
        Reads the buffer files from a folder.

        First the buffer files are read as a dask dataframe is accessed.
        The dataframe is forward filled lazily with non-electron channels.
        For the electron dataframe, all values not in the electron channels
        are dropped, and splits the sector ID from the DLD time.
        For the timed dataframe, only the train and pulse channels are taken and
        it pulse resolved (no longer electron resolved). If time_index is True,
        the timeIndex is calculated and set as the index (slow operation).
        """
        if not self.fp:
            raise FileNotFoundError("Buffer files do not exist.")
        # Loop over the electron and timed dataframes
        file_stats = {}
        filling = {}
        for typ in DF_TYP:
            # Read the parquet files into a dask dataframe
            df = dd.read_parquet(self.fp[typ], calculate_divisions=True)
            # Get the metadata from the parquet files
            file_stats[typ] = get_parquet_metadata(self.fp[typ])

            # Forward fill the non-electron channels across files
            overlap = min(file["num_rows"] for file in file_stats[typ].values())
            iterations = self._config.get("forward_fill_iterations", 2)
            df = forward_fill_lazy(
                df=df,
                columns=self.fill_channels,
                before=overlap,
                iterations=iterations,
            )
            # TODO: This dict should be returned by forward_fill_lazy
            filling[typ] = {
                "columns": self.fill_channels,
                "overlap": overlap,
                "iterations": iterations,
            }

            self.df[typ] = df
        self.metadata.update({"file_statistics": file_stats, "filling": filling})
        # Correct the 3-bit shift which encodes the detector ID in the 8s time
        if (
            self._config.get("split_sector_id_from_dld_time", False)
            and self._config.get("tof_column", "dldTimeSteps") in self.df["electron"].columns
        ):
            self.df["electron"], meta = split_dld_time_from_sector_id(
                self.df["electron"],
                config=self._config,
            )
            self.metadata.update(meta)

    def process_and_load_dataframe(
        self,
        h5_paths: list[Path],
        folder: Path,
        force_recreate: bool = False,
        suffix: str = "",
        debug: bool = False,
        remove_invalid_files: bool = False,
        filter_timed_by_electron: bool = True,
    ) -> tuple[dd.DataFrame, dd.DataFrame]:
        """
        Runs the buffer file creation process.
        Does a schema check on the buffer files and creates them if they are missing.
        Performs forward filling and splits the sector ID from the DLD time lazily.

        Args:
            h5_paths (List[Path]): List of paths to H5 files.
            folder (Path): Path to the folder for processed files.
            force_recreate (bool): Flag to force recreation of buffer files.
            suffix (str): Suffix for buffer file names.
            debug (bool): Flag to enable debug mode.):
            remove_invalid_files (bool): Flag to remove invalid files.
            filter_timed_by_electron (bool): Flag to filter timed data by valid electron events.

        Returns:
            Tuple[dd.DataFrame, dd.DataFrame]: The electron and timed dataframes.
        """
        self.fp = BufferFilePaths(self._config, h5_paths, folder, suffix, remove_invalid_files)
        self.filter_timed_by_electron = filter_timed_by_electron

        if not force_recreate:
            schema_set = set(
                get_channels(self._config, formats="all", index=True, extend_aux=True),
            )
            self._schema_check(self.fp["electron"], schema_set)
            schema_set = set(
                get_channels(
                    self._config,
                    formats=["per_pulse", "per_train"],
                    index=True,
                    extend_aux=True,
                ),
            ) - {"electronId"}
            self._schema_check(self.fp["timed"], schema_set)

        self._save_buffer_files(force_recreate, debug)

        self._get_dataframes()

        return self.df["electron"], self.df["timed"]
