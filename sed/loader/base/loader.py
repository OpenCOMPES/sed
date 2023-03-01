"""The abstract class off of which to implement loaders."""
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Sequence
from typing import Tuple

import dask.dataframe as ddf
import numpy as np

from sed.core.metadata import MetaHandler


class BaseLoader(ABC):
    """
    The abstract class off of which to implement loaders.

    The reader's folder name is the identifier.
    For this BaseLoader with filename base/loader.py the ID  becomes 'base'

    Args:
        config (dict, optional): Config dictionary. Defaults to None.
        meta_handler (MetaHandler, optional): MetaHandler object. Defaults to None.
    """

    # pylint: disable=too-few-public-methods

    __name__ = "BaseLoader"

    supported_file_types: List[str] = []

    def __init__(
        self,
        config: dict = None,
        meta_handler: MetaHandler = None,
    ):
        self._config = config if config is not None else {}

        self._meta_handler = (
            meta_handler if meta_handler is not None else MetaHandler()
        )

        self.files: List[str] = []

    @abstractmethod
    def read_dataframe(
        self,
        files: Sequence[str] = None,
        folder: str = None,
        ftype: str = None,
        **kwds,
    ) -> ddf.DataFrame:
        """Reads data from given files or folder and returns a dask dataframe.
        Metadata are added to the meta_handler object in the class.

        Args:
            files (Sequence[str], optional): List of file paths. Defaults to None.
            folder (str, optional): Path to folder where files are stored. Path has
                the priority such that if it's specified, the specified files will
                be ignored. Defaults to None.
            ftype (str, optional): File type to read ('parquet', 'json', 'csv', etc).
                If a folder path is given, all files with the specified extension are
                read into the dataframe in the reading order. Defaults to None.
            **kwds: keyword arguments. See the keyword arguments for the specific file
                parser in``dask.dataframe`` module.

        Returns:
            ddf.DataFrame: Dask dataframe read from specified files.
        """

        return None

    @abstractmethod
    def get_count_rate(
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
