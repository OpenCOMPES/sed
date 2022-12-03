"""
module sed.loader.mpes, code for loading hdf5 files delayed into a dask dataframe.
Mostly ported from https://github.com/mpes-kit/mpes.
@author: L. Rettig
"""
import os
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

import dask.dataframe as ddf
import numpy as np

from sed.loader.base.loader import BaseLoader
from sed.loader.utils import gather_files


class GenericLoader(BaseLoader):
    """Dask implementation of the Loader. Reads from various file types using the
    utilities of Dask."""

    __name__ = "dask"

    supported_file_types = ["parquet", "csv", "json"]

    def read_dataframe(
        self,
        files: Sequence[str] = None,
        folder: str = None,
        ftype: str = "parquet",
        **kwds,
    ) -> Tuple[ddf.DataFrame, dict]:
        """Read stored files from a folder into a dataframe.

        Parameters:
        folder, files: str, list/tuple | None, None
            Folder path of the files or a list of file paths. The folder path has
            the priority such that if it's specified, the specified files will
            be ignored.
        ftype: str | 'parquet'
            File type to read ('parquet', 'json', 'csv', etc).
            If a folder path is given, all files of the specified type are read
            into the dataframe in the reading order.
        **kwds: keyword arguments
            See the keyword arguments for the specific file parser in
            ``dask.dataframe`` module.

        **Return**\n
            Dask dataframe read from specified files.
        """

        metadata: Dict[Any, Any] = {}
        # pylint: disable=duplicate-code
        if folder is not None:
            folder = os.path.realpath(folder)
            files = gather_files(
                folder=folder,
                extension=ftype,
                file_sorting=True,
                **kwds,
            )

        elif folder is None:
            if files is None:
                raise ValueError(
                    "Either the folder or file path should be provided!",
                )
            files = [os.path.realpath(file) for file in files]

        self.files = files

        if not files:
            raise FileNotFoundError("No valid files found!")

        if ftype == "parquet":
            return (ddf.read_parquet(files, **kwds), metadata)

        if ftype == "json":
            return (ddf.read_json(files, **kwds), metadata)

        if ftype == "csv":
            return (ddf.read_csv(files, **kwds), metadata)

        try:
            return (ddf.read_table(files, **kwds), metadata)
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise Exception(
                "The file format cannot be understood!",
            ) from exc

    def get_count_rate(  # Pylint: disable=unused_parameter
        self,
        fids: Sequence[int] = None,
        **kwds,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create count rate data for the files specified in ``fids``.

        Parameters:
            fids: the file ids to include. None | list of file ids.
            kwds: Keyword arguments

        Return:
            countrate, seconds: Arrays containing countrate and seconds
            into the scan.
        """
        return None, None

    def get_elapsed_time(  # Pylint: disable=unused_parameter
        self,
        fids: Sequence[int] = None,
        **kwds,
    ) -> float:
        """
        Return the elapsed time in the files.

        Parameters:
            fids: the file ids to include. None | list of file ids.
            kwds: Keyword arguments

        Return:
            The elapsed time in the files in seconds.
        """
        return None


LOADER = GenericLoader
