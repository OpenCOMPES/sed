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


class BufferFilePaths:
    """
    A class for handling the paths to the raw and buffer files of electron and timed dataframes.
    A list of file sets (dict) are created for each H5 file containing the paths to the raw file
    and the electron and timed buffer files.
    """

    SUBDIRECTORIES = ["electron", "timed"]

    def __init__(self, h5_paths: list[Path], folder: Path, suffix: str) -> None:
        suffix = f"_{suffix}" if suffix else ""
        folder = folder / "buffer"

        # Create subdirectories if they do not exist
        for subfolder in self.SUBDIRECTORIES:
            (folder / subfolder).mkdir(parents=True, exist_ok=True)

        # a list of file sets containing the paths to the raw, electron and timed buffer files
        self._file_paths = [
            {
                "raw": h5_path,
                **{
                    subfolder: folder / subfolder / f"{h5_path.stem}{suffix}"
                    for subfolder in self.SUBDIRECTORIES
                },
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

    def to_process(self, force_recreate: bool = False) -> list[dict[str, Path]]:
        """Returns a list of file sets that need to be processed."""
        if not force_recreate:
            return [
                file_set
                for file_set in self
                if any(not file_set[key].exists() for key in self.SUBDIRECTORIES)
            ]
        else:
            return list(self)


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
        self.df_electron: dd.DataFrame = None
        self.df_pulse: dd.DataFrame = None
        self.metadata: dict = {}

    def _schema_check(self) -> None:
        """
        Checks the schema of the Parquet files.

        Raises:
            ValueError: If the schema of the Parquet files does not match the configuration.
        """
        buffer_filenames = self.fp["electron"] + self.fp["timed"]
        existing = [file for file in buffer_filenames if file.exists()]
        parquet_schemas = [pq.read_schema(file) for file in existing]
        config_schema_set = set(
            get_channels(self._config["channels"], formats="all", index=True, extend_aux=True),
        )

        for filename, schema in zip(existing, parquet_schemas):
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

        config_channels = self._config["channels"]

        fill_channels: list[str] = get_channels(
            config_channels,
            ["per_pulse", "per_train"],
            extend_aux=True,
        )
        # forward fill all the non-electron channels
        df[fill_channels] = df[fill_channels].ffill()

        # Reset the index of the DataFrame and save it as a parquet file
        electron_channels = get_channels(self._config["channels"], "per_electron")
        dtypes = get_dtypes(config_channels, "all", extend_aux=True)
        # electron resolved dataframe
        df.dropna(subset=electron_channels).reset_index().astype(dtypes).to_parquet(
            paths["electron"],
        )

        # timed resolved dataframe
        dtypes = get_dtypes(config_channels, ["per_pulse", "per_train"], extend_aux=True)
        df[fill_channels].loc[:, :, 0].reset_index().astype(dtypes).to_parquet(paths["timed"])

    def _save_buffer_files(self, force_recreate: bool, debug: bool) -> None:
        """
        Creates the buffer files that are missing.

        Args:
            force_recreate (bool): Flag to force recreation of buffer files.
            debug (bool): Flag to enable debug mode, which serializes the creation.
        """
        to_process = self.fp.to_process(force_recreate)
        print(f"Reading files: {len(to_process)} new files of {len(self.fp)} total.")
        n_cores = min(len(to_process), self.n_cores)
        if n_cores > 0:
            if debug:
                for file_set in to_process:
                    self._save_buffer_file(file_set)
            else:
                Parallel(n_jobs=n_cores, verbose=10)(
                    delayed(self._save_buffer_file)(file_set) for file_set in to_process
                )

    def _fill_dataframes(self):
        """
        Reads all parquet files into one dataframe using dask and fills NaN values.
        """

        df_electron = dd.read_parquet(self.fp["electron"], calculate_divisions=True)
        df_pulse = dd.read_parquet(self.fp["timed"], calculate_divisions=True)

        file_metadata = get_parquet_metadata(
            self.fp["electron"],
            time_stamp_col=self._config.get("time_stamp_alias", "timeStamp"),
        )
        self.metadata["file_statistics"] = file_metadata

        fill_channels: list[str] = get_channels(
            self._config["channels"],
            ["per_pulse", "per_train"],
            extend_aux=True,
        )
        overlap = min(file["num_rows"] for file in file_metadata.values())

        df_electron = forward_fill_lazy(
            df=df_electron,
            columns=fill_channels,
            before=overlap,
            iterations=self._config.get("forward_fill_iterations", 2),
        )

        self.metadata["forward_fill"] = {
            "columns": fill_channels,
            "overlap": overlap,
            "iterations": self._config.get("forward_fill_iterations", 2),
        }

        df_pulse = forward_fill_lazy(
            df=df_pulse,
            columns=fill_channels,
            before=overlap,
            iterations=self._config.get("forward_fill_iterations", 2),
        )

        # Correct the 3-bit shift which encodes the detector ID in the 8s time
        if self._config.get("split_sector_id_from_dld_time", False):
            df_electron, meta = split_dld_time_from_sector_id(
                df_electron,
                config=self._config,
            )
            self.metadata.update(meta)

        self.df_electron = df_electron
        self.df_pulse = df_pulse

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

        if not force_recreate:
            self._schema_check()

        self._save_buffer_files(force_recreate, debug)

        self._fill_dataframes()
