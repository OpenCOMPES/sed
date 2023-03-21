from typing import Sequence
from typing import Union

import dask.array as dda
import dask.dataframe as ddf
import numpy as np

from ..core.workflow import WorkflowStep


class Tof2Energy(WorkflowStep):
    def __init__(
        self,
        tof_column: str,
        tof_offset: float,
        tof_distance: float,
        energy_offset: float = 0,
        sign: float = 1.0,
        out_cols="energy",
        duplicate_policy="raise",
        **kwargs,
    ) -> None:
        """Convert time of flight to energy.

        Args:
            tof_column: name of the column containing tof data
            tof_offset: time of flight offset
            tof_distance: path length for the time of flight
            energy_offset: shift to apply to the energy axis. Defaults to 0.
            sign: sign to apply to the energy axis, +1 for kinetic energy,
                -1 for binding energy
            out_cols: _description_. Defaults to 'energy'.
            duplicate_policy: _description_. Defaults to 'raise'.
        """
        self.tof_column = tof_column
        self.tof_offset = tof_offset
        self.tof_distance = tof_distance
        self.energy_offset = energy_offset
        self.sign = sign
        super().__init__(
            out_cols=out_cols,
            duplicate_policy=duplicate_policy,
            **kwargs,
        )

    def func(self, df: ddf.DataFrame) -> ddf.DataFrame:
        k = self.sign * 0.5 * 1e18 * 9.10938e-31 / 1.602177e-19
        return (
            k
            * np.power(
                self.tof_distance / ((df[self.tof_column]) - self.tof_offset),
                2.0,
            )
            - self.energy_offset
        )


class DLDSectorCorrection(WorkflowStep):
    def __init__(
        self,
        tof_column: str,
        sector_id_column: str,
        sector_shifts: Union[list, dda.Array],
        out_cols: Union[str, Sequence[str]] = None,
        duplicate_policy: str = "raise",
        notes: str = "",
        **kwargs,
    ) -> None:
        """Correct the shift in tof on each sector a the dld detector

        Args:
            tof_column: _description_
            sector_id_column: _description_
            sector_shifts: list of shift values to account for at each segment of the
                detector
            out_cols: _description_
            duplicate_policy: _description_. Defaults to "raise".
            notes: _description_. Defaults to "".
        """
        if out_cols is None:
            out_cols = tof_column
        super().__init__(out_cols, duplicate_policy, notes, **kwargs)
        self.tof_column = tof_column
        self.sector_id_column = sector_id_column
        if not isinstance(sector_shifts, dda.Array):
            self.sector_shifts = dda.from_array(sector_shifts)

    def func(self, df: ddf.DataFrame) -> ddf.DataFrame:

        return (
            df[self.tof_column]
            - self.sector_shifts[df[self.sector_id_column].values.astype(int)]
        )


class SumColumns(WorkflowStep):
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


class MultiplyColumns(WorkflowStep):
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


class DivideColumns(WorkflowStep):
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


class AddJitter(WorkflowStep):
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
