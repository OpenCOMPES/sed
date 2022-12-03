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
    ) -> Tuple[ddf.DataFrame, dict]:
        """Reads data from given files or folder and returns a dask dataframe,
        and a dictionary with metadata"""
        return None, None

    @abstractmethod
    def get_count_rate(
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

    @abstractmethod
    def get_elapsed_time(self, fids: Sequence[int] = None, **kwds) -> float:
        """
        Return the elapsed time in the files.

        Parameters:
            fids: the file ids to include. None | list of file ids.
            kwds: Keyword arguments

        Return:
            The elapsed time in the files in seconds.
        """
        return None


LOADER = BaseLoader
