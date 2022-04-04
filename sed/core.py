from typing import Any

import pandas as pd
import psutil
import xarray as xr

from .metadata import MetaHandler

N_CPU = psutil.cpu_count()


class SedProcessor:
    """[summary]"""

    def __init__(self):

        self._dataframe = None

        self._dimensions = []
        self._coordinates = {}
        self._attributes = MetaHandler()

    def __repr__(self):
        if self._dataframe is None:
            df_str = "Data Frame: No Data loaded"
        else:
            df_str = self._dataframe.__repr__()
        coordinates_str = f"Coordinates: {self._coordinates}"
        dimensions_str = f"Dimensions: {self._dimensions}"
        s = df_str + "\n" + coordinates_str + "\n" + dimensions_str
        return s

    def __getitem__(self, val: Any) -> pd.DataFrame:
        """Accessor to the underlying data structure.

        Args:
            val: [description]

        Raises:
            ValueError: [description]

        Returns:
            [description]
        """
        return self._dataframe[val]

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims):
        assert isinstance(dims, list)
        self._dimensions = dims

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords):
        assert isinstance(coords, dict)
        self._coordinates = {}
        for k, v in coords.items():
            self._coordinates[k] = xr.DataArray(v)

    def load(self, data: pd.DataFrame) -> None:
        """Load tabular data of Single Events

        Args:
            data: data in tabular format. Accepts anything which
                can be interpreted by pd.DataFrame as an input

        Returns:
            None
        """
        self._dataframe = pd.DataFrame(data)

    def compute(
        self,
        mode: str = "numba",
        bin_dict: dict = None,
        axes: list = None,
        n_bins: int = None,
        ranges: list = None,
        pbar: bool = True,
        jittered: bool = True,
        n_cores: int = N_CPU,
        pbenv: str = "classic",
        **kwds,
    ) -> xr.DataArray:
        """Compute the histogram along the given dimensions.

        Args:
            mode: Binning method, choose between numba,
                fast, lean and legacy (Y. Acremann's method).
            n_cores: [description].
            axes: [description].
            n_bins: [description].
            ranges: [description].
            bin_dict: [description].
            pbar: [description].
            jittered: [description].
            pbenv: [description].

        Returns:
            [description]
        """
        pass

    def add_dimension(self, name, range):
        if name in self._coordinates:
            raise ValueError(f"Axis {name} already exists")
        else:
            self.axis[name] = self.make_axis(range)
