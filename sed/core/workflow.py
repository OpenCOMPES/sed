from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import dask.dataframe as ddf
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from ..binning import bin_dataframe
from ..config.settings import parse_config
from .metadata import MetaHandler


__version__ = "0.0.1_alpha"  # TODO: infer from sed package version

# TODO: add save to parquet option


class WorkflowStep(ABC):
    """A generic worflow step class intended to be subclassed by any workflow step"""

    def __init__(
        self,
        out_cols: Union[str, Sequence[str]],
        duplicate_policy: str = "raise",  # TODO implement duplicate policy
        notes: str = "",
    ) -> None:
        assert isinstance(out_cols, (str, list, tuple)), (
            "New columns defined in out_cols"
            " must be a string or list of strings"
        )

        self.out_cols = out_cols
        self.duplicate_policy = duplicate_policy
        self.notes = notes
        self.version = __version__
        self.name = str(self.__class__).split(".")[-1]

    @property
    def metadata(self):
        """generate a dictionary with all relevant metadata

        Returns:
            dictionary containing metadata
        """
        d = {
            n: getattr(self, n)
            for n in self.__init__.__code__.co_varnames
            if hasattr(self, n)
        }
        d["name"] = str(self.__class__).split("'")[-2].split(".")[-1]
        return d

    def apply_to(self, dd, return_=True) -> None:  # TODO: add inplace option?
        """Map the main function self.func on a dataframe.

        Args:
            dd: the dataframe on which to map the function

        Raises:
            TypeError: if the dataframe is of an unsupported format.
        """
        if isinstance(dd, ddf.DataFrame):
            dd[self.out_cols] = dd.map_partitions(
                self.func,
            )  # ,**self._kwargs)
        elif isinstance(dd, pd.DataFrame):
            dd[self.out_cols] = dd.map(self.func)  # ,**self._kwargs)
        else:
            raise TypeError("Only Dask or Pandas DataFrames are supported")
        if return_:
            return dd

    def __call__(
        self,
        dd,
    ) -> None:  # Alternative use format, maybe less intuitive
        """Allows the usage of this class as a function

        alternative application method, maybe less intuitive than "apply_to"
        """
        return self.apply_to(dd)

    @abstractmethod
    def func(
        self,
        x,
    ) -> ddf.DataFrame:
        """The main function to map on the dataframe.
        Args:
            x: the input column(s)

        Returns:
            the generated series or dataframe (column or columns)
        """
        pass

    def __repr__(self):
        s = f"Workflow step: {self.__class__}\n"
        s += "Parameters:\n"
        for k, v in self.metadata.items():
            s += f" - {k}: {v}\n"
        return s


class WorkflowManager:
    """Single event dataframe workflow manager

    Allows to apply serial transformations to the single event dataframe keeping track
    of functions and parameters called as metadata
    """

    def __init__(
        self,
        dataframe: Union[pd.DataFrame, ddf.DataFrame] = None,
        metadata: dict = None,
        config: Union[dict, str] = None,
        workflow: Union[Sequence, None] = None,
    ) -> None:
        """_summary_

        Args:
            dataframe: _description_. Defaults to None.
            metadata: _description_. Defaults to None.
            config: _description_. Defaults to None.
            steps: _description_. Defaults to None.
        """

        self._metadata: MetaHandler = metadata
        self._config = parse_config(config)
        self._dataframe = dataframe

        if workflow is None:
            self._workflow_queue = []
        elif isinstance(workflow, (list, tuple)):
            self._workflow_queue = workflow
        elif isinstance(workflow, WorkflowStep):
            self._workflow_queue = [workflow]

    def add_step(
        self,
        step: Union[WorkflowStep, Sequence[WorkflowStep]],
        **kwargs,
    ):
        """Add a workflow step to the queue

        Args:
            step: _description_
        """
        self._workflow_queue.append(step(**kwargs))

    def apply_workflow(self):
        """Run the workflow steps on the dataframe"""
        for step in self._workflow_queue:
            self._dataframe = step(self._dataframe)
            self._metadata.add(step.metadata)

    def __repr__(self):
        s = "Workflow Manager:\n"
        s += f"data:\n{self._dataframe.__repr__()}\n"
        s += "Workflow:\n"
        for step in self._workflow_queue:
            s += f"\t {step.__repr__()}"
        return s

    def add_binning(
        self,
    ) -> None:
        raise NotImplementedError

    def fast_binning(
        self,
    ) -> xr.DataArray:
        raise NotImplementedError

    def compute_dataframe(self) -> ddf.DataFrame:
        """compute the dask dataframe and store it in memory."""
        with ProgressBar():
            return self._dataframe.compute()

    def compute_binning(
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
        df_partitions = kwds.pop("df_partitions", None)
        if df_partitions is not None:
            dataframe = self._dataframe.partitions[
                0 : min(df_partitions, self._dataframe.npartitions)
            ]
        else:
            dataframe = self._dataframe

        self._binned = bin_dataframe(
            df=dataframe,
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
        from ..diagnostics import grid_histogram

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
