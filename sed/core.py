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

from sed.binning import bin_dataframe
from sed.dfops import apply_jitter
from sed.metadata import MetaHandler
from sed.settings import parse_config

N_CPU = psutil.cpu_count()


class SedProcessor:
    """[summary]"""

    def __init__(
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame] = None,
        metadata: dict = {},
        config: Union[dict, Path, str] = {},
    ):

        self._config = parse_config(config)
        if "num_cores" in self._config.keys():
            if self._config["num_cores"] >= N_CPU:
                self._config["num_cores"] = N_CPU - 1
        else:
            self._config["num_cores"] = N_CPU - 1

        self._dataframe = df

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

    def add_dimension(self, name, range):
        if name in self._coordinates:
            raise ValueError(f"Axis {name} already exists")
        else:
            self.axis[name] = self.make_axis(range)
