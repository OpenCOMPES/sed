from abc import ABC
from abc import abstractmethod
from typing import Sequence
from typing import Union

import dask.dataframe as ddf
import pandas as pd

__version__ = "0.0.1_alpha"  # TODO: infer from sed package version


class WorkflowStep(ABC):
    """A generic worflow step class intended to be subclassed by any workflow step"""

    def __init__(
        self,
        out_cols: Union[str, Sequence[str]],
        duplicate_policy: str = "raise",  # TODO implement duplicate policy
        notes: str = "",
    ) -> None:
        assert isinstance(out_cols, (str, list, tuple)), (
            "New columns defined in out_cols"
            " must be a string or list of strings"
        )

        self.out_cols = out_cols
        self.duplicate_policy = duplicate_policy
        self.notes = notes
        self.version = __version__
        self.name = str(self.__class__).split(".")[-1]

    @property
    def metadata(self):
        """generate a dictionary with all relevant metadata

        Returns:
            dictionary containing metadata
        """
        varnames = [
            s for s in self.__init__.__code__.co_varnames if s not in ["self"]
        ]
        return {n: getattr(self, n) for n in varnames}

    def apply_to(self, dd) -> None:  # TODO: add inplace option?
        """Map the main function self.func on a dataframe.

        Args:
            dd: the dataframe on which to map the function

        Raises:
            TypeError: if the dataframe is of an unsupported format.
        """
        if isinstance(dd, ddf.DataFrame):
            dd[self.out_cols] = dd.map_partitions(
                self.func,
            )  # ,**self._kwargs)
        elif isinstance(dd, pd.DataFrame):
            dd[self.out_cols] = dd.map(self.func)  # ,**self._kwargs)
        else:
            raise TypeError("Only Dask or Pandas DataFrames are supported")

    def __call__(
        self,
        dd,
    ) -> None:  # Alternative use format, maybe less intuitive
        """Allows the usage of this class as a function

        alternative application method, maybe less intuitive than "apply_to"
        """
        self.apply_to(dd)

    @abstractmethod
    def func(
        self,
        x,
    ) -> ddf.DataFrame:
        """The main function to map on the dataframe.
        Args:
            x: the input column(s)

        Returns:
            the generated series or dataframe (column or columns)
        """
        pass

    def __repr__(self):
        s = f"Workflow step: {self.__class__}\n"
        s += "Parameters:\n"
        for k, v in self.metadata.items():
            s += f" - {k}: {v}\n"
        return s
