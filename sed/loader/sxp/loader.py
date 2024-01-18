"""
This module implements the SXP data loader.
This loader currently supports the SXP momentum microscope instrument.
The raw hdf5 data is combined and saved into buffer files and loaded as a dask dataframe.
The dataframe is an amalgamation of all h5 files for a combination of runs, where the NaNs are
automatically forward-filled across different files.
This can then be saved as a parquet for out-of-sed processing and reread back to access other
sed functionality.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import cast
from typing import Sequence

import dask.dataframe as dd
import numpy as np
from natsort import natsorted

from sed.loader.base.loader import BaseLoader
from sed.loader.fel import BufferHandler
from sed.loader.fel.config_model import LoaderConfig
from sed.loader.sxp.dataframe import SXPDataFrameCreator


class SXPLoader(BaseLoader):
    """
    The class generates multiindexed multidimensional pandas dataframes from the new SXP
    dataformat resolved by both macro and microbunches alongside electrons.
    Only the read_dataframe (inherited and implemented) method is accessed by other modules.
    """

    __name__ = "sxp"

    supported_file_types = ["h5"]

    def __init__(self, config: dict) -> None:
        """
        Initializes the FlashLoader.

        Args:
            config (dict): The configuration dictionary or model.
        """
        super().__init__(config=config)
        self.config = LoaderConfig(**self._config)

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

        Returns:
            List[str]: A list of path strings representing the collected file names.

        Raises:
            FileNotFoundError: If no files are found for the given run in the directory.
        """
        # Define the stream name prefixes based on the data acquisition identifier
        stream_name_prefix = self.config.dataframe.stream_name_prefix
        stream_name_postfix = self.config.dataframe.stream_name_postfix

        if isinstance(run_id, (int, np.integer)):
            run_id = str(run_id).zfill(4)

        if folders is None:
            folders = self.config.core.base_folder

        if isinstance(folders, str):
            folders = [folders]

        # Generate the file patterns to search for in the directory
        file_pattern = f"**/{stream_name_prefix}{run_id}{stream_name_postfix}*." + extension

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

    def gather_metadata(self, metadata: dict = None) -> dict:
        """Dummy function returning empty metadata dictionary for now.

        Args:
            metadata (dict, optional): Manual meta data dictionary. Auto-generated
                meta data are added to it. Defaults to None.

        Returns:
            dict: Metadata dictionary
        """
        if metadata is None:
            metadata = {}
        return metadata

    def get_count_rate(
        self,
        fids: Sequence[int] = None,
        **kwds,
    ):
        return None, None

    def get_elapsed_time(self, fids=None, **kwds):
        return None

    def read_dataframe(
        self,
        files: str | Sequence[str] = None,
        folders: str | Sequence[str] = None,
        runs: str | Sequence[str] = None,
        ftype: str = "h5",
        metadata: dict = None,
        collect_metadata: bool = False,
        force_recreate: bool = False,
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

        paths = self.config.core.paths
        data_raw_dir = paths.data_raw_dir
        data_parquet_dir = paths.data_parquet_dir

        # Prepare a list of names for the runs to read and parquets to write
        if runs is not None:
            files = []
            if isinstance(runs, (str, int)):
                runs = [runs]
            for run in runs:
                run_files = self.get_files_from_run_id(
                    run_id=run,
                    folders=[str(folder.resolve()) for folder in [data_raw_dir]],
                    extension=ftype,
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

        # Obtain the parquet filenames, metadata, and schema from the method
        # which handles buffer file creation/reading
        h5_paths = [Path(file) for file in self.files]
        buffer = BufferHandler(
            SXPDataFrameCreator,
            self.config.dataframe,
            h5_paths,
            data_parquet_dir,
            force_recreate,
            debug=debug,
        )
        df = buffer.dataframe_electron
        df_timed = buffer.dataframe_pulse

        if collect_metadata:
            metadata = self.gather_metadata(
                metadata=self.metadata,
            )
        else:
            metadata = self.metadata
        print(f"loading complete in {time.time() - t0: .2f} s")

        return df, df_timed, metadata


LOADER = SXPLoader
