from typing import Any
from pathlib import Path
from typing import Union
from typing import Tuple
from typing import Sequence
from typing import List

import numpy as np
import pandas as pd
import psutil
import xarray as xr
import dask

from .metadata import MetaHandler
from .dfops import apply_jitter
from .binning import bin_dataframe

N_CPU = psutil.cpu_count()


class SedProcessor:
    """[summary]"""

    def __init__(self, df: pd.DataFrame = None, metadata: dict = {}, config: Union[dict, Path, str] = {}):

        # TODO: handle/load config dict/file
        self._config = config
        if not isinstance(self._config, dict):
            self._config = {}
        if 'num_cores' not in self._config.keys():
            self._config['num_cores'] = N_CPU-1

        self._dataframe = df
        self._dataframe_jittered = None

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

    def load(self, data: pd.DataFrame) -> None:
        """Load tabular data of Single Events

        Args:
            data: data in tabular format. Accepts anything which
                can be interpreted by pd.DataFrame as an input

        Returns:
            None
        """
        self._dataframe = pd.DataFrame(data)

    def gen_jittered_df(self, cols: Sequence[str] = None) -> None:
        """Generate a a second dataframe, where the selected dataframe columes have jitter applied

        Args:
            cols: the colums onto which to apply jitter. If omitted, the comlums are taken from the config.

        Returns:
            None
        """
        if cols is None:
            try: 
                cols = self._config['jitter_cols']
            except KeyError:
                cols = self._dataframe.columns #jitter all columns
            
        self._dataframe_jittered = self._dataframe.map_partitions(apply_jitter, cols=cols, cols_jittered=cols)


    def compute(
        self,
        bins: Union[
        int,
        dict,
        tuple,
        List[int],
        List[np.ndarray],
        List[tuple],] = 100,
        axes: Union[str, Sequence[str]] = None,
        ranges: Sequence[Tuple[float, float]] = None,
        histMode: str = "numba",
        mode: str = "fast",
        jittered: bool = True,
        pbar: bool = True,
        nCores: int = None,
        nThreadsPerWorker: int = 4,
        threadpoolAPI: str = "blas",
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
            pbar: [description].
            jittered: [description].
            pbenv: [description].

        Returns:
            [description]
        """
        assert self._dataframe is not None, 'dataframe needs to be loaded first!'

        if nCores is None:
            nCores = self._config['num_cores']

        df = self._dataframe

        if jittered:
            assert self._dataframe_jittered is not None, 'jittered dataframe needs to be generated first, use SedProcessor.gen_jittered_df()!'
            df = self._dataframe_jittered
        
        self._binned = bin_dataframe(df=df, bins=bins, axes=axes, ranges=ranges, histMode=histMode, mode=mode, pbar=pbar, nCores=nCores, nThreadsPerWorker=nThreadsPerWorker, threadpoolAPI=threadpoolAPI, **kwds)
        return self._binned


    def add_dimension(self, name, range):
        if name in self._coordinates:
            raise ValueError(f"Axis {name} already exists")
        else:
            self.axis[name] = self.make_axis(range)
