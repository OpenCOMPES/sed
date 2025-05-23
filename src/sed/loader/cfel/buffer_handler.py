from __future__ import annotations

import time
from pathlib import Path

import dask.dataframe as dd

from sed.core.logging import setup_logging
from sed.loader.cfel.dataframe import DataFrameCreator
from sed.loader.flash.buffer_handler import BufferFilePaths
from sed.loader.flash.buffer_handler import BufferHandler as BaseBufferHandler
from sed.loader.flash.utils import InvalidFileError
from sed.loader.flash.utils import get_channels
from sed.loader.flash.utils import get_dtypes

logger = setup_logging("cfel_buffer_handler")


class BufferHandler(BaseBufferHandler):
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
        super().__init__(config)

    def _validate_h5_files(self, config, h5_paths: list[Path]) -> list[Path]:
        valid_h5_paths = []
        for h5_path in h5_paths:
            try:
                dfc = DataFrameCreator(config_dataframe=config, h5_path=h5_path)
                dfc.validate_channel_keys()
                valid_h5_paths.append(h5_path)
            except InvalidFileError as e:
                logger.info(f"Skipping invalid file: {h5_path.stem}\n{e}")

        return valid_h5_paths    

    def _save_buffer_file(self, paths: dict[str, Path]) -> None:
        """Creates the electron and timed buffer files from the raw H5 file."""
        logger.debug(f"Processing file: {paths['raw'].stem}")
        start_time = time.time()

        # Create DataFrameCreator and get get dataframe
        dfc = DataFrameCreator(config_dataframe=self._config, h5_path=paths["raw"])
        df = dfc.df

        # Save electron resolved dataframe
        electron_channels = get_channels(self._config, "per_electron")
        dtypes = get_dtypes(self._config, df.columns.values)
        electron_df = df.dropna(subset=electron_channels).astype(dtypes).reset_index()
        logger.debug(f"Saving electron buffer with shape: {electron_df.shape}")
        electron_df.to_parquet(paths["electron"])

        # Create and save timed dataframe
        df_timed = dfc.df_timed
        dtypes = get_dtypes(self._config, df_timed.columns.values)
        timed_df = df_timed.astype(dtypes)
        logger.debug(f"Saving timed buffer with shape: {timed_df.shape}")
        timed_df.to_parquet(paths["timed"])

        logger.debug(f"Processed {paths['raw'].stem} in {time.time() - start_time:.2f}s")

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
        self.filter_timed_by_electron = filter_timed_by_electron
        if remove_invalid_files:
            h5_paths = self._validate_h5_files(self._config, h5_paths)

        self.fp = BufferFilePaths(h5_paths, folder, suffix)

        if not force_recreate:
            schema_set = set(
                get_channels(self._config, formats="all", index=True, extend_aux=True)
                + [self._config["columns"].get("timestamp")],
            )
            self._schema_check(self.fp["timed"], schema_set)

            self._schema_check(self.fp["electron"], schema_set)

        self._save_buffer_files(force_recreate, debug)

        self._get_dataframes()

        return self.df["electron"], self.df["timed"]
