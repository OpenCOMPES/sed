"""The abstract class off of which to implement loaders.
"""
from __future__ import annotations

import os
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import dask.dataframe as ddf
import numpy as np

from sed.loader.utils import gather_files


class BaseLoader(ABC):
    """
    The abstract class off of which to implement loaders.

    The reader's folder name is the identifier.
    For this BaseLoader with filename base/loader.py the ID  becomes 'base'

    Args:
        config (dict, optional): Config dictionary. Defaults to None.
        verbose (bool, optional): Option to print out diagnostic information.
            Defaults to True.
    """

    __name__ = "BaseLoader"

    supported_file_types: list[str] = []

    def __init__(
        self,
        config: dict = None,
        verbose: bool = True,
    ):
        self._config = config if config is not None else {}

        self.files: list[str] = []
        self.runs: list[str] = []
        self.metadata: dict[Any, Any] = {}
        self._verbose = verbose

    @property
    def verbose(self) -> bool:
        """Accessor to the verbosity flag.

        Returns:
            bool: Verbosity flag.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """Setter for the verbosity.

        Args:
            verbose (bool): Option to turn on verbose output. Sets loglevel to INFO.
        """
        self._verbose = verbose

    @abstractmethod
    def read_dataframe(
        self,
        files: str | Sequence[str] = None,
        folders: str | Sequence[str] = None,
        runs: str | Sequence[str] = None,
        ftype: str = None,
        metadata: dict = None,
        collect_metadata: bool = False,
        **kwds,
    ) -> tuple[ddf.DataFrame, ddf.DataFrame, dict]:
        """Reads data from given files, folder, or runs and returns a dask dataframe
        and corresponding metadata.

        Args:
            files (str | Sequence[str], optional): File path(s) to process.
                Defaults to None.
            folders (str | Sequence[str], optional): Path to folder(s) where files
                are stored. Path has priority such that if it's specified, the specified
                files will be ignored. Defaults to None.
            runs (str | Sequence[str], optional): Run identifier(s). Corresponding
                files will be located in the location provided by ``folders``. Takes
                precedence over ``files`` and ``folders``. Defaults to None.
            ftype (str, optional): File type to read ('parquet', 'json', 'csv', etc).
                If a folder path is given, all files with the specified extension are
                read into the dataframe in the reading order. Defaults to None.
            metadata (dict, optional): Manual metadata dictionary. Auto-generated
                metadata will be added to it. Defaults to None.
            collect_metadata (bool): Option to collect metadata from files. Requires
                a valid config dict. Defaults to False.
            **kwds: keyword arguments. See description in respective loader.

        Returns:
            tuple[ddf.DataFrame, ddf.DataFrame, dict]: Dask dataframe, timed dataframe and metadata
            read from specified files.
        """

        if metadata is None:
            metadata = {}

        if runs is not None:
            if isinstance(runs, (str, int)):
                runs = [runs]
            self.runs = list(runs)
            files = []
            for run in runs:
                files.extend(self.get_files_from_run_id(run, folders, **kwds))

        elif folders is not None:
            if isinstance(folders, str):
                folders = [folders]
            files = []
            for folder in folders:
                folder = os.path.realpath(folder)
                files.extend(
                    gather_files(
                        folder=folder,
                        extension=ftype,
                        file_sorting=True,
                        **kwds,
                    ),
                )

        elif files is None:
            raise ValueError(
                "Either folders, files, or runs have to be provided!",
            )

        if files is not None:
            if isinstance(files, str):
                files = [files]
            files = [os.path.realpath(file) for file in files]
            self.files = files

        self.metadata = deepcopy(metadata)

        if not files:
            raise FileNotFoundError("No valid files or runs found!")

        return None, None, None

    @abstractmethod
    def get_files_from_run_id(
        self,
        run_id: str,
        folders: str | Sequence[str] = None,
        extension: str = None,
        **kwds,
    ) -> list[str]:
        """Locate the files for a given run identifier.

        Args:
            run_id (str): The run identifier to locate.
            folders (str | Sequence[str], optional): The directory(ies) where the raw
                data is located. Defaults to None.
            extension (str, optional): The file extension. Defaults to None.
            kwds: Keyword arguments

        Return:
            list[str]: List of files for the given run.
        """
        raise NotImplementedError

    @abstractmethod
    def get_count_rate(
        self,
        fids: Sequence[int] = None,
        **kwds,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create count rate data for the files specified in ``fids``.

        Args:
            fids (Sequence[int], optional): fids (Sequence[int]): the file ids to
                include. Defaults to list of all file ids.
            kwds: Keyword arguments

        Return:
            tuple[np.ndarray, np.ndarray]: Arrays containing countrate and seconds
            into the scan.
        """
        return None, None

    @abstractmethod
    def get_elapsed_time(self, fids: Sequence[int] = None, **kwds) -> float:
        """Return the elapsed time in the specified in ``fids``.

        Args:
            fids (Sequence[int], optional): fids (Sequence[int]): the file ids to
                include. Defaults to list of all file ids.
            kwds: Keyword arguments

        Return:
            float: The elapsed time in the files in seconds.
        """
        return None


LOADER = BaseLoader
