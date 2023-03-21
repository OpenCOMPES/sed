from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Sequence

import dask.dataframe as ddf
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

import sed
from ..binning import bin_dataframe
from ..config.settings import parse_config
from .metadata import MetadataManager

__version__ = "0.0.1"  # TODO: infer from sed package version
# TODO: add save to parquet option


class ConfigManager(dict):
    """Dummy class for config manager.

    *** This should be moved to the config section of SED. ***
    """

    def __init__(self) -> None:
        super().__init__()


class PreProcessingStep(ABC):
    """A generic worflow step class intended to be subclassed by any workflow step"""

    def __init__(
        self,
        out_cols: str | Sequence[str],
        duplicate_policy: str = "raise",  # TODO implement duplicate policy
        notes: str = "",
        name: str | None = None,
        step_class: str | None = None,
        **kwargs,
    ) -> None:
        assert isinstance(out_cols, (str, list, tuple)), (
            "New columns defined in out_cols"
            " must be a string or list of strings"
        )
        if step_class is not None:
            assert step_class == self._get_step_class_name(), (
                "Warning!"
                " you are trying to load parameters of an other WorkflowStep"
            )
        self.out_cols = out_cols
        self.duplicate_policy = duplicate_policy
        self.notes = notes
        self.version = __version__
        self._name = (
            name
            if name is not None
            else str(self.__class__).split(".")[-1][:-2]
        )

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

    @property
    def name(self):
        return self._name
        # return str(self.__class__).split(".")[-1][:-2]

    @property
    def metadata(self):
        """generate a dictionary with all relevant metadata

        Returns:
            dictionary containing metadata
        """
        d = {"name": self.name}
        d.update(
            {
                n: getattr(self, n)
                for n in self.__init__.__code__.co_varnames
                if hasattr(self, n)
            },
        )
        d["step_class"] = self._get_step_class_name()
        return d

    def _get_step_class_name(self):
        return str(self.__class__).split("'")[1]

    @staticmethod
    def from_dict(
        wf_dict: dict,
    ) -> PreProcessingStep:  # TODO: move to workflow class...
        """Load parameters from a dict-like structure

        Args:
            wf_dict: _description_

        Returns:
            _description_
        """
        dict_ = deepcopy(wf_dict)
        step_class_tree = dict_["step_class"].split(".")
        step_class = sed
        for next in step_class_tree[1:]:
            step_class = getattr(step_class, next)
        return step_class(**dict_)

    def __call__(
        self,
        dd,
    ) -> Any:  # Alternative use format, maybe less intuitive
        """Allows the usage of this class as a function

        alternative application method, maybe less intuitive than "apply_to"
        """
        return self.apply_to(dd)

    def __repr__(self) -> str:
        s = f"{str(self.__class__).split('.')[-1][:-2]}("
        for k, v in self.metadata.items():
            if isinstance(v, str):
                v = f"'{v}'"
            s += f"{k}={v}, "
        return s[:-2] + ")"

    def __str__(self) -> str:
        s = f"{self.name} | "
        for k, v in self.metadata.items():
            s += f"{k}: {v}, "
        return s

    def _repr_html_(self) -> str:
        s = f"Workflow step: <strong>{self.name}</strong><br>"
        s += "<table>"
        s += "<tr><th>Parameter</th><th>Value</th></tr>"
        for k, v in self.metadata.items():
            s += f"<tr><td>{k}</td><td>{v}</td></tr>"
        s += "</table>"
        return s

    def to_json(self) -> dict:
        """summarize the workflow step as a string

        Intended for json serializing the workflow step.

        Returns:
            _description_
        """
        return self.__repr__()

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
                # *self.args,
                # **self.kwargs,
            )
        elif isinstance(dd, pd.DataFrame):
            dd[self.out_cols] = dd.map(self.func)
        else:
            raise TypeError("Only Dask or Pandas DataFrames are supported")
        if return_:
            return dd


class ParameterGenerator(ABC):
    """Class template to generate parameters for a preprocessing step

    Args:
        ABC: _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _description_
    """

    @abstractmethod
    def generate_parameters(self) -> dict:
        """Core method to be defined, which generates and returns the parameters

        Returns:
            _description_
        """
        pass

    def get_preprocessing_step(self):
        """Return an instance of the associated preprocessing step with the
        defined parameters
        """
        assert isinstance(self._preprocessing_step, PreProcessingStep)
        return self._preprocessing_step(**self.parameters)

    def metadata(self) -> dict:
        raise NotImplementedError


class PostProcessingStep(ABC):
    pass


class WorkflowManager:
    """Single event dataframe workflow manager, from loading to binning

    Allows to apply serial transformations to the single event dataframe keeping track
    of functions and parameters called as metadata
    """

    def __init__(
        self,
        dataframe: pd.DataFrame | ddf.DataFrame = None,
        preprocessing_pipeline: Sequence[PreProcessingStep]
        | str
        | Path
        | None = None,
        metadata: MetadataManager | dict = None,
        config: dict | ConfigManager | str | Path | None = None,
    ) -> None:
        self._metadata: MetadataManager = metadata
        self._config: ConfigManager = parse_config(config)
        self._dataframe: pd.DataFrame | ddf.DataFrame = dataframe

        if preprocessing_pipeline is None:
            self._preprocessing_pipeline = []
        elif isinstance(preprocessing_pipeline, (list, tuple)):
            self._preprocessing_pipeline = preprocessing_pipeline
        elif isinstance(preprocessing_pipeline, PreProcessingStep):
            self._preprocessing_pipeline = [preprocessing_pipeline]

    def __repr__(self):
        s = "Workflow Manager:\n"
        s += f"data:\n{self._dataframe.__repr__()}\n"
        s += "Workflow:\n"
        for i, step in enumerate(self._preprocessing_pipeline):
            s += f"{i+1}. {str(step)}\n"
        return s

    # loading

    def load_datarame(self, source) -> None:
        raise NotImplementedError

    # pre-processing

    def add_preprocessing_step(
        self,
        steps: PreProcessingStep | Sequence[PreProcessingStep],
    ):
        """Add one or a list of workflow step to the queue

        Args:
            step: Workflow steps to add
        """
        if not isinstance(steps, list):
            steps = [steps]
        for s in steps:
            if isinstance(s, (PreProcessingStep, ParameterGenerator)):
                self._preprocessing_pipeline.append(s)
            else:
                raise TypeError(f"{s} is not a valid pre-processing step")

    def preprocess(self):
        """Run the workflow steps on the dataframe"""
        for step in self._preprocessing_pipeline:
            if isinstance(step, ParameterGenerator):
                step.get = (
                    self._dataframe
                )  # give the dataframe to the paramter generator

                parameter_dict = (
                    step.make_parameter_dict()
                )  # this is an interactive gui which when complete returns the params
                step = step.get_preprocessing_step(
                    **parameter_dict,
                )  # this is the function for which the parameters are being generated
            if isinstance(step, PreProcessingStep):
                self._dataframe = step(self._dataframe)
                self._metadata.add(step.metadata)
            else:
                raise AttributeError(
                    f"{step} is not a valid preprocessing step or parameter generator.",
                )

    def fast_binning(
        self,
    ) -> xr.DataArray:
        raise NotImplementedError

    # binning and output

    def compute_dataframe(self) -> ddf.DataFrame:
        """compute the dask dataframe and store it in memory."""
        with ProgressBar():
            return self._dataframe.compute()

    def add_binning_dimension(self) -> None:
        raise NotImplementedError

    def compute_binning(
        self,
        bins: (
            int | dict | tuple | list[int] | list[np.ndarray] | list[tuple]
        ) = 100,
        axes: str | Sequence[str] = None,
        ranges: Sequence[tuple[float, float]] = None,
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

    # post-processing
    # for example the per-pulse normalizations

    def add_postprocessing_step(self) -> None:
        raise NotImplementedError

    def postprocess(self) -> None:
        if self._binned is None:
            raise RuntimeError("Must bin first!")
        raise NotImplementedError

    # I/O

    def to_json(self, fname: str | Path = None, key: str = "") -> dict:
        """Save the current workflow to a json file

        Args:
            fname: file where to write the workflow. If none, no file is saved.
                Defaults to None
            key: key in the file where to write. Defaults to ''

        Raises:
            NotImplementedError: # TODO: make this

        Returns:
            dictionary with the workflow steps
        """
        raise NotImplementedError

    def to_parquet(
        self,
        fname: str | Path,
        run_workflow: bool = True,
    ) -> None:
        """Save the dataframe as parquet

        Args:
            fname: path where to save the dataframe
            run_workflow: if true, it computes the workflow before saving, else
                it saves the current version of the dataframe.

        Raises:
            NotImplementedError: # TODO: make this
        """
        raise NotImplementedError

    def to_nexus(self) -> Path:
        """creates a nexus file from the binned data

        Returns:
            path to the file generated
        """
        raise NotImplementedError
