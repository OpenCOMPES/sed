
from typing import Any, Dict, Sequence
import psutil

import pandas as pd
import numpy as np
import xarray as xr


from .metadata import MetaHandler
from .binning import bin_dataframe#binDataframe, binDataframe_fast,binDataframe_lean,binDataframe_numba

N_CPU = psutil.cpu_count()

class SedProcessor:
    """[summary]
    """

    def __init__(self):

        self._dataframe = None
        
        self._dimensions = []
        self._coordinates = {}
        self._attributes = MetaHandler()

    def __repr__(self):
        if self._dataframe is None:
            df_str = 'Data Frame: No Data loaded'
        else:
            df_str = self._dataframe.__repr__()
        coordinates_str = f'Coordinates: {self._coordinates}'
        dimensions_str = f'Dimensions: {self._dimensions}'
        s = df_str + '\n' + coordinates_str + '\n' + dimensions_str
        return s

    def __getitem__(self,val: Any) -> pd.DataFrame:
        """ Accessor to the underlying data structure.

        Args:
            val (Any): [description]

        Raises:
            ValueError: [description]

        Returns:
            pd.DataFrame: [description]
        """
        return self._dataframe[val]

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self,dims):
        assert isinstance(dims,list)
        self._dimensions = dims

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self,coords):
        assert isinstance(coords,dict)
        self._coordinates = {}
        for k,v in coords.items():
            self._coordinates[k] = xr.DataArray(v)

    def load(self, data: pd.DataFrame) -> None:
        """ Load tabular data of Single Events

        Args:
            data (TabularType): data in tabular format. Accepts anything which 
                can be interpreted by pd.DataFrame as an input

        Returns:
            None
        """
        self._dataframe = pd.DataFrame(data)

    def compute(self,
        mode: str='numba',
        binDict: dict=None, 
        axes: list=None, 
        nbins: int=None,
        ranges: list=None,
        pbar: bool=True, 
        jittered: bool=True, 
        ncores: int=N_CPU, 
        pbenv: str='classic', 
        **kwds) -> xr.DataArray:
        """ Compute the histogram along the given dimensions.

        Args:
            mode (str, optional): Binning method, choose between numba, 
                fast, lean and legacy (Y. Acremann's method). Defaults to 'numba'.
            ncores (int, optional): [description]. Defaults to N_CPU.
            axes ([type], optional): [description]. Defaults to None.
            nbins (int, optional): [description]. Defaults to None.
            ranges (list, optional): [description]. Defaults to None.
            binDict (dict, optional): [description]. Defaults to None.
            pbar (bool, optional): [description]. Defaults to True.
            jittered (bool, optional): [description]. Defaults to True.
            pbenv (str, optional): [description]. Defaults to 'classic'.

        Returns:
            xr.DataArray: [description]
        """
        pass

    def add_dimension(self,name,range):
        if name in self._coordinates:
            raise ValueError(f'Axis {name} already exists')
        else:
            self.axis[name] = self.make_axis(range)
