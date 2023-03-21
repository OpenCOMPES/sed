from typing import Sequence
from typing import Union

import dask.dataframe as ddf
import numpy as np

from .workflow import PreProcessingStep


class SumColumns(PreProcessingStep):
    def __init__(
        self,
        col_a: str,
        col_b: str,
        factor: int = 1,
        out_cols: Union[str, Sequence[str]] = None,
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
        out_cols: Union[str, Sequence[str]],
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
        out_cols: Union[str, Sequence[str]],
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
        out_cols: Union[str, Sequence[str]] = None,
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
