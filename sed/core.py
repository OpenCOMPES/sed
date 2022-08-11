from pathlib import Path
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import dask.dataframe
import numpy as np
import pandas as pd
import psutil
import xarray as xr

from .binning import bin_dataframe
from .dfops import apply_jitter
from .diagnostic import grid_histogram
from .metadata import MetaHandler

N_CPU = psutil.cpu_count()


class SedProcessor:
    """[summary]"""

    def __init__(
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame] = None,
        metadata: dict = {},
        config: Union[dict, Path, str] = {},
    ):

        # TODO: handle/load config dict/file
        self._config = config
        if not isinstance(self._config, dict):
            self._config = {}
        if "hist_mode" not in self._config.keys():
            self._config["hist_mode"] = "numba"
        if "mode" not in self._config.keys():
            self._config["mode"] = "fast"
        if "pbar" not in self._config.keys():
            self._config["pbar"] = True
        if "num_cores" not in self._config.keys():
            self._config["num_cores"] = N_CPU - 1
        if "threads_per_worker" not in self._config.keys():
            self._config["threads_per_worker"] = 4
        if "threadpool_API" not in self._config.keys():
            self._config["threadpool_API"] = "blas"

        self._dataframe = df
        self._dataframe_jittered = None

        self._config["default_bins"] = [80, 80, 80, 80]
        self._config["default_axes"] = ["X", "Y", "t", "ADC"]
        self._config["default_ranges"] = [
            (0, 1800),
            (0, 1800),
            (68000, 74000),
            (0, 500),
        ]

        self._dimensions = []
        self._coordinates = {}
        self._attributes = MetaHandler(meta=metadata)

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

    def load(
        self,
        data: Union[pd.DataFrame, dask.dataframe.DataFrame],
    ) -> None:
        """Load tabular data of Single Events

        Args:
            data: data in tabular format. Accepts anything which
                can be interpreted by pd.DataFrame as an input

        Returns:
            None
        """
        self._dataframe = data

    def add_jitter(self, cols: Sequence[str] = None) -> None:
        """Add jitter to the selected dataframe columns.


        Args:
            cols: the colums onto which to apply jitter. If omitted,
            the comlums are taken from the config.

        Returns:
            None
        """
        if cols is None:
            try:
                cols = self._config["jitter_cols"]
            except KeyError:
                cols = self._dataframe.columns  # jitter all columns

        self._dataframe = self._dataframe.map_partitions(
            apply_jitter,
            cols=cols,
            cols_jittered=cols,
        )

    def compute(
        self,
        bins: Union[
            int,
            dict,
            tuple,
            List[int],
            List[np.ndarray],
            List[tuple],
        ] = 100,
        axes: Union[str, Sequence[str]] = None,
        ranges: Sequence[Tuple[float, float]] = None,
        **kwds,
    ) -> xr.DataArray:
        """Compute the histogram along the given dimensions.

        Args:
            bins: Definition of the bins. Can  be any of the following cases:
                - an integer describing the number of bins in on all dimensions
                - a tuple of 3 numbers describing start, end and step of the binning
                  range
                - a np.arrays defining the binning edges
                - a list (NOT a tuple) of any of the above (int, tuple or np.ndarray)
                - a dictionary made of the axes as keys and any of the above as values.
                This takes priority over the axes and range arguments.
            axes: The names of the axes (columns) on which to calculate the histogram.
                The order will be the order of the dimensions in the resulting array.
            ranges: list of tuples containing the start and end point of the binning
                    range.
            kwds: Keywords argument passed to bin_dataframe.

        Raises:
            AssertError: Rises when no dataframe has been loaded.

        Returns:
            The result of the n-dimensional binning represented in an
                xarray object, combining the data with the axes.
        """

        assert (
            self._dataframe is not None
        ), "dataframe needs to be loaded first!"

        hist_mode = kwds.pop("hist_mode", self._config["hist_mode"])
        mode = kwds.pop("mode", self._config["mode"])
        pbar = kwds.pop("pbar", self._config["pbar"])
        num_cores = kwds.pop("num_cores", self._config["num_cores"])
        threads_per_worker = kwds.pop(
            "threads_per_worker",
            self._config["threads_per_worker"],
        )
        threadpool_API = kwds.pop(
            "threadpool_API",
            self._config["threadpool_API"],
        )

        self._binned = bin_dataframe(
            df=self._dataframe,
            bins=bins,
            axes=axes,
            ranges=ranges,
            histMode=hist_mode,
            mode=mode,
            pbar=pbar,
            nCores=num_cores,
            nThreadsPerWorker=threads_per_worker,
            threadpoolAPI=threadpool_API,
            **kwds,
        )
        return self._binned

    def viewEventHistogram(
        self,
        dfpid: int,
        ncol: int = 2,
        bins: Sequence[int] = None,
        axes: Sequence[str] = None,
        ranges: Sequence[Tuple[float, float]] = None,
        jittered: bool = False,
        backend: str = "bokeh",
        legend: bool = True,
        histkwds: dict = {},
        legkwds: dict = {},
        **kwds: Any,
    ):
        """
        Plot individual histograms of specified dimensions (axes) from a substituent
        dataframe partition.

        Args:
            dfpid: Number of the data frame partition to look at.
            ncol: Number of columns in the plot grid.
            bins: Number of bins to use for the speicified axes.
            axes: Name of the axes to display.
            ranges: Value ranges of all specified axes.
            jittered: Option to use the jittered dataframe.
            backend: Backend of the plotting library ('matplotlib' or 'bokeh').
            legend: Option to include a legend in the histogram plots.
            histkwds, legkwds, **kwds: Extra keyword arguments passed to
            ``sed.diagnostics.grid_histogram()``.

        Raises:
            AssertError if Jittering is requested, but the jittered dataframe
            has not been created.
            TypeError: Raises when the input values are not of the correct type.
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
            grid_histogram(
                group_dict,
                ncol=ncol,
                rvs=axes,
                rvbins=bins,
                rvranges=ranges,
                backend=backend,
                legend=legend,
                histkwds=histkwds,
                legkwds=legkwds,
                **kwds,
            )

        else:
            raise TypeError(
                "Inputs of axes, bins, ranges need to be list or tuple!",
            )

    def add_dimension(self, name, range):
        if name in self._coordinates:
            raise ValueError(f"Axis {name} already exists")
        else:
            self.axis[name] = self.make_axis(range)
