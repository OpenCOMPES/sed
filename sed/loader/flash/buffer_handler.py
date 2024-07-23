from __future__ import annotations

import os
from pathlib import Path

import dask.dataframe as dd
import pyarrow.parquet as pq
from joblib import delayed
from joblib import Parallel

from sed.core.dfops import forward_fill_lazy
from sed.loader.flash.dataframe import DataFrameCreator
from sed.loader.flash.utils import get_channels
from sed.loader.flash.utils import get_dtypes
from sed.loader.utils import get_parquet_metadata
from sed.loader.utils import split_dld_time_from_sector_id


DF_TYP = ["electron", "timed"]


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

    def __init__(self, h5_paths: list[Path], folder: Path, suffix: str) -> None:
        """Initializes the BufferFilePaths.

        Args:
            h5_paths (list[Path]): List of paths to the H5 files.
            folder (Path): Path to the folder for processed files.
            suffix (str): Suffix for buffer file names.
        """
        suffix = f"_{suffix}" if suffix else ""
        folder = folder / "buffer"
        folder.mkdir(parents=True, exist_ok=True)

        # a list of file sets containing the paths to the raw, electron and timed buffer files
        self._file_paths = [
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

    def _schema_check(self, files: list[Path], expected_schema_set: set) -> None:
        """
        Checks the schema of the Parquet files.

        Raises:
            ValueError: If the schema of the Parquet files does not match the configuration.
        """
        existing = [file for file in files if file.exists()]
        parquet_schemas = [pq.read_schema(file) for file in existing]

        for filename, schema in zip(existing, parquet_schemas):
            schema_set = set(schema.names)
            if schema_set != expected_schema_set:
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

    def _save_buffer_file(self, paths: dict[str, Path]) -> None:
        """
        Creates the electron and timed buffer files from the raw H5 file.
        First the dataframe is accessed and forward filled in the non-electron channels.
        Then the data types are set. For the electron dataframe, all values not in the electron
        channels are dropped. For the timed dataframe, only the train and pulse channels are taken
        and it pulse resolved (no longer electron resolved). Both are saved as parquet files.

        Args:
            paths (dict[str, Path]): Dictionary containing the paths to the H5 and buffer files.
        """

        # Create a DataFrameCreator instance and the h5 file
        df = DataFrameCreator(config_dataframe=self._config, h5_path=paths["raw"]).df

        # forward fill all the non-electron channels
        df[self.fill_channels] = df[self.fill_channels].ffill()

        # Reset the index of the DataFrame and save both the electron and timed dataframes
        # electron resolved dataframe
        electron_channels = get_channels(self._config, "per_electron")
        dtypes = get_dtypes(self._config, df.columns.values)
        df.dropna(subset=electron_channels).astype(dtypes).reset_index().to_parquet(
            paths["electron"],
        )

        # timed dataframe
        # drop the electron channels and only take rows with the first electronId
        df_timed = df[self.fill_channels].loc[:, :, 0]
        dtypes = get_dtypes(self._config, df_timed.columns.values)
        df_timed.astype(dtypes).reset_index().to_parquet(paths["timed"])

    def _save_buffer_files(self, force_recreate: bool, debug: bool) -> None:
        """
        Creates the buffer files that are missing.

        Args:
            force_recreate (bool): Flag to force recreation of buffer files.
            debug (bool): Flag to enable debug mode, which serializes the creation.
        """
        file_sets = self.fp.file_sets_to_process(force_recreate)
        print(f"Reading files: {len(file_sets)} new files of {len(self.fp)} total.")
        n_cores = min(len(file_sets), self.n_cores)
        if n_cores > 0:
            if debug:
                for file_set in file_sets:
                    self._save_buffer_file(file_set)
                    print(f"Processed {file_set['raw'].stem}")
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
        if self._config.get("split_sector_id_from_dld_time", False):
            self.df["electron"], meta = split_dld_time_from_sector_id(
                self.df["electron"],
                config=self._config,
            )
            self.metadata.update(meta)

    def run(
        self,
        h5_paths: list[Path],
        folder: Path,
        force_recreate: bool = False,
        suffix: str = "",
        debug: bool = False,
    ) -> None:
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
        """
        self.fp = BufferFilePaths(h5_paths, folder, suffix)

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
