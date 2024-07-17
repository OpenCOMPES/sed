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
    A class for handling the paths to the raw and buffer files.
    A list of file sets (dict) are created for each H5 file containing the paths to the raw file
    and the electron and timed buffer files.

    Structure of the file sets:
    {
        "raw": Path to the H5 file,
        "buffer": Path to the buffer file
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
                "buffer": folder / h5_path.stem,
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
        return [file_set for file_set in self._file_paths if not file_set["buffer"].exists()]


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
            self._config["channels"],
            ["per_pulse", "per_train"],
            extend_aux=True,
        )
        self.metadata: dict = {}

    def _schema_check(self) -> None:
        """
        Checks the schema of the Parquet files.

        Raises:
            ValueError: If the schema of the Parquet files does not match the configuration.
        """
        expected_schema_set = set(
            get_channels(self._config["channels"], formats="all", index=True, extend_aux=True),
        )
        existing = [file for file in self.fp["buffer"] if file.exists()]
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

    def _save_buffer_file(self, path: dict[str, Path]) -> None:
        """
        Creates the electron and timed buffer files from the raw H5 file.


        Args:
            paths (dict[str, Path]): Dictionary containing the paths to the H5 and buffer files.
        """

        # Create a DataFrameCreator instance and the h5 file
        DataFrameCreator(config_dataframe=self._config, h5_path=path["raw"]).df.to_parquet(
            path["buffer"],
        )

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
            else:
                Parallel(n_jobs=n_cores, verbose=10)(
                    delayed(self._save_buffer_file)(file_set) for file_set in file_sets
                )

    def _fill_dataframes(self, df):
        """
        Reads all parquet files into one dataframe using dask and fills NaN values lazily.
        """
        # Forward fill the non-electron channels across files
        overlap = min(file["num_rows"] for file in self.metadata["file_statistics"].values())
        iterations = self._config.get("forward_fill_iterations", 2)
        df = forward_fill_lazy(
            df=df,
            columns=self.fill_channels,
            before=overlap,
            iterations=iterations,
        )
        # TODO: This dict should be returned by forward_fill_lazy
        filling = {
            "columns": self.fill_channels,
            "overlap": overlap,
            "iterations": iterations,
        }
        self.metadata.update({"filling": filling})
        return df

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
        df = dd.read_parquet(self.fp["buffer"], calculate_divisions=True)
        # Get the metadata from the parquet files
        file_stats = get_parquet_metadata(
            self.fp["buffer"],
        )
        self.metadata.update({"file_statistics": file_stats})
        df = self._fill_dataframes(df)

        # Electron dataframe
        electron_dtypes = get_dtypes(self._config["channels"], "per_electron")
        self.df["electron"] = df.dropna(
            subset=get_channels(self._config["channels"], "per_electron"),
        ).astype(electron_dtypes)
        # Correct the 3-bit shift which encodes the detector ID in the 8s time
        if self._config.get("split_sector_id_from_dld_time", False):
            self.df["electron"], meta = split_dld_time_from_sector_id(
                self.df["electron"],
                config=self._config,
            )
            self.metadata.update(meta)

        # Timed dataframe
        df_timed = df[df["electronId"] == 0]
        timed_channels = get_channels(
            self._config["channels"],
            ["per_pulse", "per_train"],
            extend_aux=True,
        ) + ["pulseId"]
        df_timed = df_timed[timed_channels].dropna()
        timed_dtypes = get_dtypes(
            self._config["channels"],
            ["per_pulse", "per_train"],
            extend_aux=True,
        )
        self.df["timed"] = df_timed.astype(timed_dtypes)

        if self._config.get("time_index", False):
            # Calculate the time delta for each pulseId (1 microsecond per pulse)
            df_timed["timeIndex"] = dd.to_datetime(
                df_timed["timeStamp"],
                unit="s",
            ) + dd.to_timedelta(df_timed["pulseId"], unit="us")

            # Set the new fine timeStamp as the index if needed
            self.df["timed"] = df_timed.set_index("timeIndex")

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

        Args:
            h5_paths (List[Path]): List of paths to H5 files.
            folder (Path): Path to the folder for processed files.
            force_recreate (bool): Flag to force recreation of buffer files.
            suffix (str): Suffix for buffer file names.
            debug (bool): Flag to enable debug mode.):
        """
        self.fp = BufferFilePaths(h5_paths, folder, suffix)

        self._schema_check()

        self._save_buffer_files(force_recreate, debug)

        self._get_dataframes()
