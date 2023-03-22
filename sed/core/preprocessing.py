from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Any
from typing import Sequence

import dask.dataframe as ddf
import numpy as np
import pandas as pd

import sed
from .workflow import __version__


class PreProcessingStep(ABC):
    """A generic worflow step class intended to be subclassed by any workflow step"""

    def __init__(
        self,
        out_cols: str | Sequence[str],
        duplicate_policy: str = "raise",  # TODO implement duplicate policy
        notes: str = "",
        name: str | None = None,
        step_class: str | None = None,
        **kwargs,
    ) -> None:
        assert isinstance(out_cols, (str, list, tuple)), (
            "New columns defined in out_cols"
            " must be a string or list of strings"
        )
        if step_class is not None:
            assert step_class == self._get_step_class_name(), (
                "Warning!"
                " you are trying to load parameters of an other WorkflowStep"
            )
        self.out_cols = out_cols
        self.duplicate_policy = duplicate_policy
        self.notes = notes
        self.version = __version__
        self._name = (
            name
            if name is not None
            else str(self.__class__).split(".")[-1][:-2]
        )

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

    @property
    def name(self):
        return self._name
        # return str(self.__class__).split(".")[-1][:-2]

    @property
    def metadata(self):
        """generate a dictionary with all relevant metadata

        Returns:
            dictionary containing metadata
        """
        d = {"name": self.name}
        d.update(
            {
                n: getattr(self, n)
                for n in self.__init__.__code__.co_varnames
                if hasattr(self, n)
            },
        )
        d["step_class"] = self._get_step_class_name()
        return d

    def _get_step_class_name(self):
        return str(self.__class__).split("'")[1]

    @staticmethod
    def from_dict(
        wf_dict: dict,
    ) -> PreProcessingStep:  # TODO: move to workflow class...
        """Load parameters from a dict-like structure

        Args:
            wf_dict: _description_

        Returns:
            _description_
        """
        dict_ = deepcopy(wf_dict)
        step_class_tree = dict_["step_class"].split(".")
        step_class = sed
        for next in step_class_tree[1:]:
            step_class = getattr(step_class, next)
        return step_class(**dict_)

    def __call__(
        self,
        dd,
    ) -> Any:  # Alternative use format, maybe less intuitive
        """Allows the usage of this class as a function

        alternative application method, maybe less intuitive than "apply_to"
        """
        return self.apply_to(dd)

    def __repr__(self) -> str:
        s = f"{str(self.__class__).split('.')[-1][:-2]}("
        for k, v in self.metadata.items():
            if isinstance(v, str):
                v = f"'{v}'"
            s += f"{k}={v}, "
        return s[:-2] + ")"

    def __str__(self) -> str:
        s = f"{self.name} | "
        for k, v in self.metadata.items():
            s += f"{k}: {v}, "
        return s

    def _repr_html_(self) -> str:
        s = f"Workflow step: <strong>{self.name}</strong><br>"
        s += "<table>"
        s += "<tr><th>Parameter</th><th>Value</th></tr>"
        for k, v in self.metadata.items():
            s += f"<tr><td>{k}</td><td>{v}</td></tr>"
        s += "</table>"
        return s

    def to_json(self) -> dict:
        """summarize the workflow step as a string

        Intended for json serializing the workflow step.

        Returns:
            _description_
        """
        return self.__repr__()

    def apply_to(self, dd, return_=True) -> None:  # TODO: add inplace option?
        """Map the main function self.func on a dataframe.

        Args:
            dd: the dataframe on which to map the function

        Raises:
            TypeError: if the dataframe is of an unsupported format.
        """
        if isinstance(dd, ddf.DataFrame):
            dd[self.out_cols] = dd.map_partitions(
                self.func,
                # *self.args,
                # **self.kwargs,
            )
        elif isinstance(dd, pd.DataFrame):
            dd[self.out_cols] = dd.map(self.func)
        else:
            raise TypeError("Only Dask or Pandas DataFrames are supported")
        if return_:
            return dd


class SumColumns(PreProcessingStep):
    def __init__(
        self,
        col_a: str,
        col_b: str,
        factor: int = 1,
        out_cols: str | Sequence[str] = None,
        duplicate_policy: str = "raise",
        notes: str = "",
        **kwargs,
    ) -> None:
        """Sum values in two columns

        follows the equation out = a + factor * b

        Args:
            col_a: left column
            col_b: right column
            factor: factor to apply to right column. Defaults to 1.
            out_cols: _description_. Defaults to None.
            duplicate_policy: _description_. Defaults to "raise".
            notes: _description_. Defaults to "".
        """
        if out_cols is None:
            out_cols = col_a
        super().__init__(out_cols, duplicate_policy, notes, **kwargs)
        self.col_a = col_a
        self.col_b = col_b
        self.factor = factor

    def func(self, df):
        assert self.col_a in df.columns
        assert self.col_b in df.columns
        return df[self.col_a] + self.factor * df[self.col_b]


class MultiplyColumns(PreProcessingStep):
    def __init__(
        self,
        col_a: str,
        col_b: str,
        out_cols: str | Sequence[str],
        duplicate_policy: str = "raise",
        notes: str = "",
        **kwargs,
    ) -> None:
        super().__init__(out_cols, duplicate_policy, notes, **kwargs)
        """ Multiplies values in two columns

        follows the equation out = a * b

        Args:
            col_a: left column
            col_b: right column
            factor: factor to apply to right column. Defaults to 1.
            out_cols: _description_. Defaults to None.
            duplicate_policy: _description_. Defaults to "raise".
            notes: _description_. Defaults to "".
        """
        if out_cols is None:
            out_cols = col_a
        super().__init__(out_cols, duplicate_policy, notes)
        self.col_a = col_a
        self.col_b = col_b

    def func(self, df):
        assert self.col_a in df.columns
        assert self.col_b in df.columns
        return df[self.col_a] * df[self.col_b]


class DivideColumns(PreProcessingStep):
    def __init__(
        self,
        col_a: str,
        col_b: str,
        out_cols: str | Sequence[str],
        duplicate_policy: str = "raise",
        notes: str = "",
        **kwargs,
    ) -> None:
        super().__init__(out_cols, duplicate_policy, notes, **kwargs)
        """ Divides values in a column by the values in an other column

        follows the equation out = a / b

        Args:
            col_a: left column
            col_b: right column
            factor: factor to apply to right column. Defaults to 1.
            out_cols: _description_. Defaults to None.
            duplicate_policy: _description_. Defaults to "raise".
            notes: _description_. Defaults to "".
        """
        if out_cols is None:
            out_cols = col_a
        super().__init__(out_cols, duplicate_policy, notes)
        self.col_a = col_a
        self.col_b = col_b

    def func(self, df):
        assert self.col_a in df.columns
        assert self.col_b in df.columns
        return df[self.col_a] / df[self.col_b]


class AddJitter(PreProcessingStep):
    def __init__(
        self,
        column: str,
        amplitude: float = 0.5,
        jitter_type: str = "uniform",
        inplace: bool = True,
        out_cols: str | Sequence[str] = None,
        duplicate_policy: str = "append",
        **kwargs,
    ) -> None:
        if out_cols is None:
            out_cols = column if inplace else f"{column}_jit"
        super().__init__(out_cols, duplicate_policy, **kwargs)
        assert jitter_type in [
            "uniform",
            "normal",
        ], f"jitter type must be one of 'uniform' and 'normal', not {jitter_type}"
        self.jitter_type = jitter_type
        self.amplitude = amplitude
        self.column = column

    @property
    def name(self):
        return f"AddJitter_{self.column}"

    @property
    def jitter_array(self):
        if self.jitter_type == "uniform":
            # Uniform Jitter distribution
            return np.random.uniform(low=-1, high=1, size=self._col_size)
        elif self.jitter_type == "normal":
            # Normal Jitter distribution works better for non-linear transformations and
            # jitter sizes that don't match the original bin sizes
            return np.random.standard_normal(size=self._col_size)

    def func(self, df: ddf.DataFrame) -> ddf.DataFrame:
        self._col_size = df.shape[0]
        # print(self._col_size,df)

        return df[self.column] + self.jitter_array * self.amplitude
