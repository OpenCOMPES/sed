from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union, List, Dict, Any, Tuple

import dask.dataframe as ddf
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from .preprocessing import PreProcessingPipeline, PreProcessingStep
from .metadata import MetaHandler
from ..binning import bin_dataframe
from .config import parse_config 


class WorkflowManager:
    """ Single event dataframe workflow manager, from loading to binning"""

    def __init__(
            self,
            dataframe:ddf.DataFrame=None,
            pre_processing: Union[PreProcessingPipeline,list[PreProcessingStep]]=None,
            metadata: MetaHandler | dict | None = None,
            config: dict | None = None,
        ) -> None:
        if not isinstance(metadata, MetaHandler):
            metadata = MetaHandler(metadata)
        self._metadata: MetaHandler = metadata
        if pre_processing is None:
            pre_processing = []
        self._pre_pipe: PreProcessingPipeline = PreProcessingPipeline(
            pre_processing
        )
        # config management is inconsistent with he rest of the code. We should 
        # move it to sed.core.config and use a ConfigManager class
        self._config: dict = parse_config(config)
        if isinstance(dataframe,(ddf.DataFrame,pd.DataFrame)):
            self._dataframe: pd.DataFrame | ddf.DataFrame = dataframe


        self._dimensions: List[str] = []
        self._coordinates: Dict[Any, Any] = {}

        self._binned_array: xr.DataArray | None = None

    @property
    def binned_array(self) -> xr.DataArray | None:
        
        if self._binned_array is not None:
            if (self._binned_array.dims == self._dimensions) or \
                (self._binned_array.coords == self._coordinates):
                Warning(
                    "Binned array dimensions and coordinates do not match dataframe. "
                    "Recompute binning."
                )
            return self._binned_array
        else:
            raise ValueError("Binned array not yet computed. Run bin_dataframe() first")

    @property
    def metadata(self) -> MetaHandler:
        return self._metadata
    
    @property
    def config(self) -> dict:
        return self._config
    
    @property
    def dataframe(self) -> pd.DataFrame | ddf.DataFrame:
        return self._dataframe
    
    @property
    def dims(self) -> List[str]:
        return self._dimensions
    
    @dims.setter
    def dims(self, dims: List[str]) -> None:
        self._dimensions = dims
    
    @property
    def coords(self) -> Dict[Any, Any]:
        if len(self._dimensions) != len(self._coordinates):
            raise ValueError("Number of dimensions and coordinates do not match")
        return self._coordinates
    
    @coords.setter
    def coords(self, coords: Dict[Any, Any]) -> None:
        if any([dim not in self._dimensions for dim in coords.keys()]):
            raise ValueError("Not all coordinates are defined as dimensions")
        self._coordinates = coords
    
    @property
    def attrs(self) -> Dict[Any, Any]:
        return self._metadata
    
    @attrs.setter
    def attrs(self, attrs: Dict[Any, Any]) -> None:
        self._metadata = attrs
    
    @property
    def _bins(self) -> dict:
        return self._config["bins"]
    
    @property
    def _axes(self) -> list:
        return self.dims
    
    @property
    def _ranges(self) -> List(Tuple[float, float]):
        return [(min(x), max(x)) for x in self._coordinates.values()]

    def __repr__(self):
        return f"WorkflowManager(dataframe={self._dataframe}, pre_processing={self._pre_processing_pipeline}, metadata={self._metadata}, config={self._config})"
    
    def __str__(self):
        s = "Workflow Manager:\n"
        s += f"data:\n{self._dataframe.__repr__()}\n"
        s += "Workflow:\n"
        for i, step in enumerate(self._preprocessing_pipeline):
            s += f"{i+1}. {str(step)}\n"
        return s

    def __getitem__(self, key):
        return self._dataframe[key]
    
    def _repr_html_(self):
        """ html representation of the __str__  method"""
        return self.__str__().replace("\n", "<br>")
    
    def add_dimension(self,dim,coord=None):
        """Add a dimension to the dataframe"""
        if dim not in self._dimensions:
            self._dimensions.append(dim)
        if coord is not None:
            self._coordinates[dim] = coord
    
    def add_axis(self,axis,coord=None):
        """Add an axis to the dataframe"""
        if axis not in self.axis:
            self.axis[axis] = coord

    def pre_process(self, dataframe: pd.DataFrame | ddf.DataFrame | None = None) -> pd.DataFrame | ddf.DataFrame:
        """Apply preprocessing pipeline to dataframe"""
        dataframe = dataframe or self._dataframe
        self._pre_pipe.map(dataframe, self._metadata)
        return dataframe
    
    def from_parquet(self, path: str | Path, **kwargs) -> ddf.DataFrame:
        """Load dataframe from path
        
        Args:
            path: path to dataframe
            **kwargs: keyword arguments passed to dask.dataframe.read_parquet
            
        Returns:
            dask.dataframe.DataFrame
        """
        self._dataframe = ddf.read_parquet(path, **kwargs)
        return self._dataframe
    
    def to_parquet(self, path: str | Path, **kwargs) -> None:
        """Save dataframe to path
        
        Args:
            path: path to dataframe
            **kwargs: keyword arguments passed to dask.dataframe.DataFrame.to_parquet
        """
        self._dataframe.to_parquet(path, **kwargs)
    
    def from_dataframe(self, dataframe: pd.DataFrame | ddf.DataFrame, npartitions=2) -> None:
        """Load dataframe
        
        Args:
            dataframe: dataframe to load
            npartitions: number of partitions to use for dask dataframe
        
        Raises:
            ValueError: if dataframe is not a pandas or dask dataframe
        
        """
        if isinstance(dataframe, pd.DataFrame):
            d = ddf.from_pandas(dataframe, npartitions=npartitions)
        elif isinstance(dataframe, ddf.DataFrame):
            d = dataframe
        else:
            raise ValueError("dataframe must be a pandas or dask dataframe")
        self._dataframe = dataframe

    def bin(
            self, 
            bins: (
                int | dict | tuple | list[int] | list[np.ndarray] | list[tuple]
            ) = 100,
            axes: str | Sequence[str] = None,
            ranges: Sequence[tuple[float, float]] = None,**kwargs
        ) -> xr.DataArray:
        """Bin dataframe
        
        Args:
            bins: number of bins or bin edges. If a list of arrays is given,
                each array is used as bin edges for the corresponding axis.
                If a list of tuples is given, each tuple is used as (bins, range)
                for the corresponding axis. If a dict is given, it must be of the
                form {axis: bins} or {axis: (bins, range)}
            axes: The names of the axes (columns) on which to calculate the histogram.
                The order will be the order of the dimensions in the resulting array.
                If None, the axes are inferred from bins.
            ranges: list of tuples containing the start and end point of the binning
                    range. If None, the ranges are inferred from bins.
            **kwargs: additional arguments passed to the binning function 
                (see  :func:`bin_dataframe`)
                
        Returns:
            binned dataframe
        """
        bins = bins or self._bins
        axes = axes or self._axes
        ranges = ranges or self._ranges
        # TODO: check config file for binning parameters
        self._binned_array = bin_dataframe(
            self._dataframe, 
            bins=bins,
            axes=axes,
            ranges=ranges,
            **kwargs)
        return bin_dataframe(self._dataframe, **kwargs)
    
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
