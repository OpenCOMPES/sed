from typing import Sequence
from typing import Union

import dask.array as dda
import dask.dataframe as ddf
import numpy as np

from ..core.workflow import PreProcessingStep


class Tof2Energy(PreProcessingStep):
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


class DLDSectorCorrection(PreProcessingStep):
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
