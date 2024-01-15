from __future__ import annotations

import numpy as np
from pandas import concat
from pandas import DataFrame
from pandas import Index
from pandas import MultiIndex
from pandas import Series

from sed.loader.fel.dataframe import DataFrameCreator
from sed.loader.fel.utils import get_channels


class SXPDataFrameCreator(DataFrameCreator):
    def get_dataset_array(
        self,
        channel: str,
        slice_: bool = False,
    ) -> tuple[Index, np.ndarray]:
        """
        Returns a numpy array for a given channel name for a given file.

        Args:
            h5_file (h5py.File): The h5py file object.
            channel (str): The name of the channel.

        Returns:
            Tuple[Index, np.ndarray]: A tuple containing the train ID Series and the numpy array
            for the channel's data.

        """
        # Get the data from the necessary h5 file and channel
        channel_dict = self._config.channels.get(channel)
        index_key = channel_dict.index_key
        dataset_key = channel_dict.dataset_key

        key = Index(self.h5_file[index_key], name="trainId")

        # unpacks the data into np.ndarray
        np_array = self.h5_file[dataset_key][()]
        if len(np_array.shape) == 2 and channel_dict.max_hits:
            np_array = np_array[:, : channel_dict.max_hits]

        if channel_dict.scale:
            np_array = np_array / float(channel_dict.scale)

        # If np_array is size zero, fill with NaNs
        if len(np_array.shape) == 0:
            # Fill the np_array with NaN values of the same shape as train_id
            np_array = np.full_like(key, np.nan, dtype=np.double)

        return key, np_array

    def pulse_index(self) -> MultiIndex:
        """
        Computes the index for the 'per_electron' data.

        Args:
            offset (int): The offset value.

        Returns:
            MultiIndex: computed MultiIndex
        """
        train_id, mab_array = self.get_dataset_array("trainId")
        train_id, mib_array = self.get_dataset_array("pulseId")

        macrobunch_index = []
        microbunch_ids = []
        macrobunch_indices = []

        for i, _ in enumerate(train_id):
            num_trains = self._config.num_trains
            if num_trains:
                try:
                    num_valid_hits = np.where(np.diff(mib_array[i].astype(np.int32)) < 0)[0][
                        num_trains - 1
                    ]
                    mab_array[i, num_valid_hits:] = 0
                    mib_array[i, num_valid_hits:] = 0
                except IndexError:
                    pass

            train_ends = np.where(np.diff(mib_array[i].astype(np.int32)) < -1)[0]
            indices = []
            index = 0
            for train, train_end in enumerate(train_ends):
                macrobunch_index.append(train_id[i] + np.uint(train))
                microbunch_ids.append(mib_array[i, index:train_end])
                indices.append(slice(index, train_end))
                index = train_end + 1
            macrobunch_indices.append(indices)

        # Create a series with the macrobunches as index and
        # microbunches as values
        macrobunches = (
            Series(
                (microbunch_ids[i] for i in range(len(macrobunch_index))),
                name="pulseId",
                index=macrobunch_index,
            )
            - self._config.ubid_offset
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
        index = MultiIndex.from_arrays(
            (microbunches.index, microbunches.values, electrons),
            names=self.multi_index,
        )
        return macrobunch_indices, index

    @property
    def df_electron(self) -> DataFrame:
        """
        Returns a pandas DataFrame for 'per_electron' data.

        Returns:
            DataFrame: The pandas DataFrame for the 'per_electron' data.
        """

        # Get the channels for the dataframe
        channels = get_channels(self._config.channels, "per_electron")
        if not channels:
            return DataFrame()

        series = []
        for channel in channels:
            array_indices, index = self.pulse_index()
            _, np_array = self.get_dataset_array(channel)
            if array_indices is None or len(array_indices) != np_array.shape[0]:
                raise RuntimeError(
                    "macrobunch_indices not set correctly, internal inconstency detected.",
                )
            train_data = []
            for i, _ in enumerate(array_indices):
                for indices in array_indices[i]:
                    train_data.append(np_array[i, indices])

            drop_vals = np.arange(-self._config.ubid_offset, 0)
            series.append(
                Series((train for train in train_data), name=channel)
                .explode()
                .dropna()
                .to_frame()
                .set_index(index)
                .drop(
                    index=drop_vals,
                    level=1,
                    errors="ignore",
                ),
            )
            print(series)

        return concat(series, axis=1)
