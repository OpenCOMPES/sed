from __future__ import annotations

from typing import TypeVar

import dask.dataframe as ddf
import numpy as np
from pandas import DataFrame
from pandas import MultiIndex
from pandas import Series

DFType = TypeVar("DFType", DataFrame, ddf.DataFrame)


class MultiIndexCreator:
    def reset_multi_index(self) -> None:
        """Resets the index per pulse and electron"""
        self.index_per_electron = None
        self.index_per_pulse = None

    def create_multi_index_per_electron(self, train_id, np_array, ubid_offset) -> None:
        """
        Creates an index per electron using pulseId for usage with the electron
            resolved pandas DataFrame.

        Args:
            train_id (Series): The train ID Series.
            np_array (np.ndarray): The numpy array containing the pulseId data.

        Notes:
            - This method relies on the 'pulseId' channel to determine
                the macrobunch IDs.
            - It creates a MultiIndex with trainId, pulseId, and electronId
                as the index levels.
        """

        # Create a series with the macrobunches as index and
        # microbunches as values
        macrobunches = (
            Series(
                (np_array[i] for i in train_id.index),
                name="pulseId",
                index=train_id,
            )
            - ubid_offset
        )

        # Explode dataframe to get all microbunch vales per macrobunch,
        # remove NaN values and convert to type int
        microbunches = macrobunches.explode().dropna().astype(int)

        # Create temporary index values
        index_temp = MultiIndex.from_arrays(
            (microbunches.index, microbunches.values),
            names=["trainId", "pulseId"],
        )

        # Calculate the electron counts per pulseId unique preserves the order of appearance
        electron_counts = index_temp.value_counts()[index_temp.unique()].values

        # Series object for indexing with electrons
        electrons = (
            Series(
                [np.arange(electron_counts[i]) for i in range(electron_counts.size)],
            )
            .explode()
            .astype(int)
        )

        # Create a pandas MultiIndex using the exploded datasets
        self.index_per_electron = MultiIndex.from_arrays(
            (microbunches.index, microbunches.values, electrons),
            names=self.multi_index,
        )

    def create_multi_index_per_pulse(
        self,
        train_id: Series,
        np_array: np.ndarray,
    ) -> None:
        """
        Creates an index per pulse using a pulse resolved channel's macrobunch ID, for usage with
        the pulse resolved pandas DataFrame.

        Args:
            train_id (Series): The train ID Series.
            np_array (np.ndarray): The numpy array containing the pulse resolved data.

        Notes:
            - This method creates a MultiIndex with trainId and pulseId as the index levels.
        """

        # Create a pandas MultiIndex, useful for comparing electron and
        # pulse resolved dataframes
        self.index_per_pulse = MultiIndex.from_product(
            (train_id, np.arange(0, np_array.shape[1])),
            names=["trainId", "pulseId"],
        )
