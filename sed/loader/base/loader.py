"""The abstract class off of which to implement loaders."""
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Sequence
from typing import Tuple

import dask.dataframe as ddf

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

    @abstractmethod
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


LOADER = BaseLoader
