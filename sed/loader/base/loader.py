"""The abstract class off of which to implement loaders."""
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Sequence
from typing import Tuple

import dask.dataframe as ddf


class BaseLoader(ABC):
    """
    The abstract class off of which to implement loaders.

    The reader's folder name is the identifier.
    For this BaseLoader with filename base/loader.py the ID  becomes 'base'
    """

    # pylint: disable=too-few-public-methods

    __name__ = "BaseLoader"

    @abstractmethod
    def __init__(
        self,
        config: dict = None,
    ):
        if config is None:
            config = {}

        self._config = config

        self.files: List[str] = []

    @abstractmethod
    def read_dataframe(
        self,
        files: Sequence[str] = None,
        **kwds,
    ) -> Tuple[ddf.DataFrame, dict]:
        """Reads data from given files or folder and returns a dask dataframe,
        and a dictionary with metadata"""
        return (ddf.DataFrame(), {})


LOADER = BaseLoader
