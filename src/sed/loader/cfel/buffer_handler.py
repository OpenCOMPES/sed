from __future__ import annotations

import time
from pathlib import Path

import dask.dataframe as dd
from joblib import delayed
from joblib import Parallel

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

    def _save_buffer_files(self, force_recreate: bool, debug: bool) -> None:
        """
        Creates the buffer files that are missing, handling multi-file runs properly.

        Args:
            force_recreate (bool): Flag to force recreation of buffer files.
            debug (bool): Flag to enable debug mode, which serializes the creation.
        """
        file_sets = self.fp.file_sets_to_process(force_recreate)
        logger.info(f"Reading files: {len(file_sets)} new files of {len(self.fp)} total.")
        
        if len(file_sets) == 0:
            return
            
        # Sort file sets by filename to ensure proper order
        file_sets = sorted(file_sets, key=lambda x: x['raw'].name)
        
        # Get base timestamp from the first file if we have multiple files
        base_timestamp = None
        if len(file_sets) > 1:
            try:
                # Find the first file (ends with _0000)
                first_file_set = None
                for file_set in file_sets:
                    if file_set['raw'].stem.endswith('_0000'):
                        first_file_set = file_set
                        break
                
                if first_file_set:
                    # Create a temporary DataFrameCreator to extract base timestamp
                    first_dfc = DataFrameCreator(
                        config_dataframe=self._config, 
                        h5_path=first_file_set['raw'],
                        is_first_file=True
                    )
                    base_timestamp = first_dfc.get_base_timestamp()
                    first_dfc.h5_file.close()  # Clean up
                    logger.info(f"Multi-file run detected. Base timestamp: {base_timestamp}")
            except Exception as e:
                logger.warning(f"Could not extract base timestamp: {e}. Processing files independently.")
                base_timestamp = None
        
        n_cores = min(len(file_sets), self.n_cores)
        if n_cores > 0:
            if debug:
                for file_set in file_sets:
                    is_first_file = file_set['raw'].stem.endswith('_0000')
                    self._save_buffer_file(file_set, is_first_file, base_timestamp)
            else:
                # For parallel processing, we need to be careful about the order
                # Process all files in parallel with the correct parameters
                from joblib import delayed, Parallel
                
                Parallel(n_jobs=n_cores, verbose=10)(
                    delayed(self._save_buffer_file)(
                        file_set, 
                        file_set['raw'].stem.endswith('_0000'),
                        base_timestamp
                    ) 
                    for file_set in file_sets
                )

    def _save_buffer_file(self, file_set, is_first_file=True, base_timestamp=None):
        """
        Saves an HDF5 file to a Parquet file using the DataFrameCreator class.
        
        Args:
            file_set: Dictionary containing file paths
            is_first_file: Whether this is the first file in a multi-file run
            base_timestamp: Base timestamp from the first file (for subsequent files)
        """
        start_time = time.time()  # Add this line
        paths = file_set
        
        dfc = DataFrameCreator(
            config_dataframe=self._config, 
            h5_path=paths["raw"],
            is_first_file=is_first_file,
            base_timestamp=base_timestamp
        )
        df = dfc.df
        df_timed = dfc.df_timed

        # Save electron resolved dataframe
        electron_channels = get_channels(self._config, "per_electron")
        dtypes = get_dtypes(self._config, df.columns.values)
        electron_df = df.dropna(subset=electron_channels).astype(dtypes).reset_index()
        logger.debug(f"Saving electron buffer with shape: {electron_df.shape}")
        electron_df.to_parquet(paths["electron"])

        # Create and save timed dataframe
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
