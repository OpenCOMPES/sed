from __future__ import annotations

import numpy as np
from pandas import concat
from pandas import DataFrame
from pandas import MultiIndex
from pandas import Series

from sed.loader.fel.dataframe import DataFrameCreator
from sed.loader.fel.utils import get_channels


class FlashDataFrameCreator(DataFrameCreator):
    def pulse_index(self, offset: int) -> tuple[MultiIndex, slice | np.ndarray]:
        """
        Computes the index for the 'per_electron' data.

        Args:
            offset (int): The offset value.

        Returns:
            tuple[MultiIndex, np.ndarray]: A tuple containing the computed MultiIndex and
            the indexer.
        """
        # Get the pulseId and the index_train
        index_train, dataset_pulse = self.get_dataset_array("pulseId", slice_=True)
        # Repeat the index_train by the number of pulses
        index_train_repeat = np.repeat(index_train, dataset_pulse.shape[1])
        # Explode the pulse dataset and subtract by the ubid_offset
        pulse_ravel = dataset_pulse.ravel() - offset
        # Create a MultiIndex with the index_train and the pulse
        microbunches = MultiIndex.from_arrays((index_train_repeat, pulse_ravel)).dropna()

        # Only sort if necessary
        indexer = slice(None)
        if not microbunches.is_monotonic_increasing:
            microbunches, indexer = microbunches.sort_values(return_indexer=True)

        # Count the number of electrons per microbunch and create an array of electrons
        electron_counts = microbunches.value_counts(sort=False).values
        electrons = np.concatenate([np.arange(count) for count in electron_counts])

        # Final index constructed here
        index = MultiIndex.from_arrays(
            (
                microbunches.get_level_values(0),
                microbunches.get_level_values(1).astype(int),
                electrons,
            ),
            names=self.multi_index,
        )
        return index, indexer

    @property
    def df_electron(self) -> DataFrame:
        """
        Returns a pandas DataFrame for a given channel name of type [per electron].

        Returns:
            DataFrame: The pandas DataFrame for the 'per_electron' channel's data.
        """
        channels = get_channels(self._config.channels, "per_electron")
        if not channels:
            return DataFrame()

        offset = self._config.ubid_offset
        # Index
        index, indexer = self.pulse_index(offset)

        # Data logic
        slice_index = [self._config.channels.get(channel).slice for channel in channels]

        # First checking if dataset keys are the same for all channels
        dataset_keys = [self._config.channels.get(channel).dataset_key for channel in channels]
        all_keys_same = all(key == dataset_keys[0] for key in dataset_keys)

        # If all dataset keys are the same, we can directly use the ndarray to create frame
        if all_keys_same:
            _, dataset = self.get_dataset_array(channels[0])
            data_dict = {
                channel: dataset[:, slice_, :].ravel()
                for channel, slice_ in zip(channels, slice_index)
            }
            dataframe = DataFrame(data_dict)
        # Otherwise, we need to create a Series for each channel and concatenate them
        else:
            series = {
                channel: Series(self.get_dataset_array(channel, slice_=True)[1].ravel())
                for channel in channels
            }
            dataframe = concat(series, axis=1)

        drop_vals = np.arange(-offset, 0)

        # Few things happen here:
        # Drop all NaN values like while creating the multiindex
        # if necessary, the data is sorted with [indexer]
        # MultiIndex is set
        # Finally, the offset values are dropped
        return (
            dataframe.dropna()[indexer]
            .set_index(index)
            .drop(index=drop_vals, level="pulseId", errors="ignore")
        )
