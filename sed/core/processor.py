"""This module contains the core class for the sed package

"""
from typing import Any
from typing import Dict
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
from sed.config.settings import parse_config
from sed.core.dfops import apply_jitter
from sed.core.metadata import MetaHandler
from sed.diagnostics import grid_histogram

N_CPU = psutil.cpu_count()


class SedProcessor:
    """[summary]"""

    def __init__(
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame] = None,
        metadata: dict = None,
        config: Union[dict, str] = None,
    ):

        self._config = parse_config(config)
        if "num_cores" in self._config["binning"].keys():
            if self._config["binning"]["num_cores"] >= N_CPU:
                self._config["binning"]["num_cores"] = N_CPU - 1
        else:
            self._config["binning"]["num_cores"] = N_CPU - 1

        self._dataframe = df

        self._binned: xr.DataArray = None

        self._dimensions: List[str] = []
        self._coordinates: Dict[Any, Any] = {}
        self.axis: Dict[Any, Any] = {}
        self._attributes = MetaHandler(meta=metadata)

    def __repr__(self):
        if self._dataframe is None:
            df_str = "Data Frame: No Data loaded"
        else:
            df_str = self._dataframe.__repr__()
        coordinates_str = f"Coordinates: {self._coordinates}"
        dimensions_str = f"Dimensions: {self._dimensions}"
        pretty_str = df_str + "\n" + coordinates_str + "\n" + dimensions_str
        return pretty_str

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
    def config(self) -> Dict[Any, Any]:
        """Getter attribute for the config dictionary

        Returns:
            Dict: The config dictionary.
        """
        return self._config

    @config.setter
    def config(self, config: Union[dict, str]):
        """Setter function for the config dictionary.

        Args:
            config (Union[dict, str]): Config dictionary or path of config file
            to load.
        """
        self._config = parse_config(config)
        if "num_cores" in self._config["binning"].keys():
            if self._config["binning"]["num_cores"] >= N_CPU:
                self._config["binning"]["num_cores"] = N_CPU - 1
        else:
            self._config["binning"]["num_cores"] = N_CPU - 1

    @property
    def dimensions(self) -> list:
        """Getter attribute for the dimensions.

        Returns:
            list: List of dimensions.
        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims: list):
        """Setter function for the dimensions.

        Args:
            dims (list): List of dimensions to set.
        """
        assert isinstance(dims, list)
        self._dimensions = dims

    @property
    def coordinates(self) -> dict:
        """Getter attribute for the coordinates dict.

        Returns:
            dict: Dictionary of coordinates.
        """
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords: dict):
        """Setter function for the coordinates dict

        Args:
            coords (dict): Dictionary of coordinates.
        """
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

        hist_mode = kwds.pop("hist_mode", self._config["binning"]["hist_mode"])
        mode = kwds.pop("mode", self._config["binning"]["mode"])
        pbar = kwds.pop("pbar", self._config["binning"]["pbar"])
        num_cores = kwds.pop("num_cores", self._config["binning"]["num_cores"])
        threads_per_worker = kwds.pop(
            "threads_per_worker",
            self._config["binning"]["threads_per_worker"],
        )
        threadpool_api = kwds.pop(
            "threadpool_API",
            self._config["binning"]["threadpool_API"],
        )

        self._binned = bin_dataframe(
            df=self._dataframe,
            bins=bins,
            axes=axes,
            ranges=ranges,
            hist_mode=hist_mode,
            mode=mode,
            pbar=pbar,
            n_cores=num_cores,
            threads_per_worker=threads_per_worker,
            threadpool_api=threadpool_api,
            **kwds,
        )
        return self._binned

    def view_event_histogram(
        self,
        dfpid: int,
        ncol: int = 2,
        bins: Sequence[int] = None,
        axes: Sequence[str] = None,
        ranges: Sequence[Tuple[float, float]] = None,
        backend: str = "bokeh",
        legend: bool = True,
        histkwds: dict = None,
        legkwds: dict = None,
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
            bins = self._config["histogram"]["bins"]
        if axes is None:
            axes = self._config["histogram"]["axes"]
        if ranges is None:
            ranges = self._config["histogram"]["ranges"]

        input_types = map(type, [axes, bins, ranges])
        allowed_types = [list, tuple]

        df = self._dataframe

        if not set(input_types).issubset(allowed_types):
            raise TypeError(
                "Inputs of axes, bins, ranges need to be list or tuple!",
            )

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

    def add_dimension(self, name: str, axis_range: Tuple):
        """Add a dimension axis.

        Args:
            name (str): name of the axis
            axis_range (Tuple): range for the axis.

        Raises:
            ValueError: Raised if an axis with that name already exists.
        """
        if name in self._coordinates:
            raise ValueError(f"Axis {name} already exists")

        self.axis[name] = self.make_axis(axis_range)

    def make_axis(self, axis_range: Tuple) -> np.ndarray:
        """Function to make an axis.

        Args:
            axis_range (Tuple): range for the new axis.
        """

        # TODO: What shall this function do?
        return np.arange(*axis_range)
