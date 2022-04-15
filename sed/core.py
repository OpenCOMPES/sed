from typing import Any
from typing import Union
from typing import Tuple
from typing import Sequence
from typing import List

import pandas as pd
import psutil
import xarray as xr

from .metadata import MetaHandler
from .diagnostic import grid_histogram

N_CPU = psutil.cpu_count()


class SedProcessor:
    """[summary]"""

    def __init__(self):

        self._dataframe = None

        self._config={}
        self._config["default_bins"] = [80, 80, 80, 80]
        self._config["default_axes"] = ['X', 'Y', 't', 'ADC']
        self._config["default_ranges"] = [(0, 1800), (0, 1800), (68000, 74000), (0, 500)]

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
        self._dataframe = data

    def compute(
        self,
        mode: str = "numba",
        binDict: dict = None,
        axes: list = None,
        nbins: int = None,
        ranges: list = None,
        pbar: bool = True,
        jittered: bool = True,
        ncores: int = N_CPU,
        pbenv: str = "classic",
        **kwds,
    ) -> xr.DataArray:
        """Compute the histogram along the given dimensions.

        Args:
            mode: Binning method, choose between numba,
                fast, lean and legacy (Y. Acremann's method).
            ncores: [description].
            axes: [description].
            nbins: [description].
            ranges: [description].
            binDict: [description].
            pbar: [description].
            jittered: [description].
            pbenv: [description].

        Returns:
            [description]
        """
        pass

    def viewEventHistogram(self, dfpid: int, ncol: int=2, bins: Sequence[int]=None, 
                axes:Sequence[str]=None,
                ranges:Sequence[Tuple[float, float]]=None, jittered:bool = True,
                backend:str='bokeh', legend:bool=True, histkwds:dict={}, legkwds:dict={}, **kwds:Any):
        """
        Plot individual histograms of specified dimensions (axes) from a substituent dataframe partition.
        **Parameters**\n
        dfpid: int
            Number of the data frame partition to look at.
        ncol: int
            Number of columns in the plot grid.
        axes: list/tuple
            Name of the axes to view.
        bins: list/tuple
            Bin values of all speicified axes.
        ranges: list
            Value ranges of all specified axes.
        backend: str | 'matplotlib'
            Backend of the plotting library ('matplotlib' or 'bokeh').
        legend: bool | True
            Option to include a legend in the histogram plots.
        histkwds, legkwds, **kwds: dict, dict, keyword arguments
            Extra keyword arguments passed to ``mpes.visualization.grid_histogram()``.
        """
        if bins is None:
            bins = self._config["default_bins"]
        if axes is None:
            axes = self._config["default_axes"]
        if ranges is None:
            ranges = self._config["default_ranges"]

        input_types = map(type, [axes, bins, ranges])
        allowed_types = [list, tuple]

        df = self._dataframe

        if jittered:
            assert self._dataframe_jittered is not None, (
                "jittered dataframe needs to be generated first, "
                "use SedProcessor.gen_jittered_df(cols)!"
            )
            df = self._dataframe_jittered

        if set(input_types).issubset(allowed_types):

            # Read out the values for the specified groups
            group_dict = {}
            dfpart = df.get_partition(dfpid)
            cols = dfpart.columns
            for ax in axes:
                group_dict[ax] = dfpart.values[:, cols.get_loc(ax)].compute()

            # Plot multiple histograms in a grid
            grid_histogram(group_dict, ncol=ncol, rvs=axes, rvbins=bins, rvranges=ranges,
                    backend=backend, legend=legend, histkwds=histkwds, legkwds=legkwds, **kwds)

        else:
            raise TypeError('Inputs of axes, bins, ranges need to be list or tuple!')

    def add_dimension(self, name, range):
        if name in self._coordinates:
            raise ValueError(f"Axis {name} already exists")
        else:
            self.axis[name] = self.make_axis(range)
