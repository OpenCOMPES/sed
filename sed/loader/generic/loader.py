"""
module sed.loader.mpes, code for loading hdf5 files delayed into a dask dataframe.
Mostly ported from https://github.com/mpes-kit/mpes.
@author: L. Rettig
"""
from typing import Sequence
from typing import Tuple

import dask.dataframe as ddf
import numpy as np

from sed.loader.base.loader import BaseLoader


class GenericLoader(BaseLoader):
    """Dask implementation of the Loader. Reads from various file types using the
    utilities of Dask.

    Args:
        config (dict, optional): Config dictionary. Defaults to None.
        meta_handler (MetaHandler, optional): MetaHandler object. Defaults to None.
    """

    __name__ = "generic"

    supported_file_types = ["parquet", "csv", "json"]

    def read_dataframe(
        self,
        files: Sequence[str] = None,
        folder: str = None,
        ftype: str = "parquet",
        runs: Sequence[str] = None,
        metadata: dict = None,
        collect_metadata: bool = False,
        **kwds,
    ) -> Tuple[ddf.DataFrame, dict]:
        """Read stored files from a folder into a dataframe.

        Args:
            files (Sequence[str], optional): List of file paths. Defaults to None.
            folder (str, optional): Path to folder where files are stored. Path has
                the priority such that if it's specified, the specified files will
                be ignored. Defaults to None.
            ftype (str, optional): File type to read ('parquet', 'json', 'csv', etc).
                If a folder path is given, all files with the specified extension are
                read into the dataframe in the reading order. Defaults to "parquet".
            runs (Sequence[str], optional): List of run identifiers. Defaults to None.
            metadata (dict, optional): Manual meta data dictionary. Auto-generated
                meta data are added to it. Defaults to None.
            collect_metadata (bool): Option to collect metadata from files. Requires
                a valid config dict. Defaults to False.
            **kwds: keyword arguments. See the keyword arguments for the specific file
                parser in``dask.dataframe`` module.

        Raises:
            ValueError: Raised if neither files nor folder provided.
            FileNotFoundError: Raised if the fileds or folder cannot be found.
            ValueError: Raised if the file type is not supported.

        Returns:
            Tuple[ddf.DataFrame, dict]: Dask dataframe and metadata read from specified
            files.
        """
        # pylint: disable=duplicate-code
        super().read_dataframe(
            files=files,
            folder=folder,
            ftype=ftype,
            runs=runs,
            metadata=metadata,
        )

        if not self.files:
            raise FileNotFoundError("No valid files found!")

        if collect_metadata:
            # TODO implementation
            self.metadata = self.metadata

        if ftype == "parquet":
            return (ddf.read_parquet(self.files, **kwds), self.metadata)

        if ftype == "json":
            return (ddf.read_json(self.files, **kwds), self.metadata)

        if ftype == "csv":
            return (ddf.read_csv(self.files, **kwds), self.metadata)

        try:
            return (ddf.read_table(self.files, **kwds), self.metadata)
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise ValueError(
                "The file format cannot be understood!",
            ) from exc

    def get_count_rate(  # Pylint: disable=unused_parameter
        self,
        fids: Sequence[int] = None,
        **kwds,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create count rate data for the files specified in ``fids``.

        Args:
            fids (Sequence[int], optional): fids (Sequence[int]): the file ids to
                include. Defaults to list of all file ids.
            kwds: Keyword arguments

        Return:
            Tuple[np.ndarray, np.ndarray]: Arrays containing countrate and seconds
            into the scan.
        """
        # TODO
        return None, None

    def get_elapsed_time(  # Pylint: disable=unused_parameter
        self,
        fids: Sequence[int] = None,
        **kwds,
    ) -> float:
        """Return the elapsed time in the files specified in ``fids``.

        Args:
            fids (Sequence[int], optional): fids (Sequence[int]): the file ids to
                include. Defaults to list of all file ids.
            kwds: Keyword arguments

        Returns:
            float: The elapsed time in the files in seconds.
        """
        return None


LOADER = GenericLoader
