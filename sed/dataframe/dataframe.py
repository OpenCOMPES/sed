from typing import Any
from typing import List
from typing import Union

import dask
import dask.dataframe as ddf
import numpy as np
import pandas as pd

COLUMN_ALIASES = {
    "ANALYSER_ENERGY": ["energy", "ev"],
    "TIME_OF_FLIGHT": ["tof", "dldTime", "dldTof"],
    "DELAY_STAGE_TIME": ["delayStage"],
    "PUMP_PROBE_TIME": ["pumpProbeTime"],
    "POSITION_X": ["dldPosX"],
    "POSITION_Y": ["dldPosY"],
}


class MetadataHandler:
    pass


class Columns:
    def __init__(self, names, units=None, alias_dict=None):
        self._names = names
        self._units = units
        self._alias_dict = alias_dict

    def _check_consistency(self, df):
        return all(self.has_column(c) for c in df) and self.ncols == df.ncols


class Channel:
    def __init__(
        self,
        data: Any = None,
        index: Any = None,  # should be an instance of Channel...
        name: str = None,
        alias: str = None,
    ) -> None:
        self.data = data
        self.index = index
        self.name = name
        self.alias = alias

    def values(self):
        return pd.Series(data=self._data, index=self.index, name=self.alias)

    def from_h5(self, h5addr):
        raise NotImplementedError

    def from_array(self, arr, index=None):
        if index is not None:
            if len(arr) != len(index):
                raise ValueError(
                    "The array and index provided have different sizes.",
                )
        self._data = arr

    def __len__(self):
        return len(self.data)


class DataFrame:
    def __init__(
        self,
        df: Any = None,
        columns: Union[Columns, List[str], None] = None,
        metadata: MetadataHandler = None,
        **kwargs,
    ) -> None:
        self._df = df  # main dataframe
        self._meta = metadata  # static data and metadata
        if isinstance(columns, Columns):
            self.columns = columns
        else:
            self.columns = Columns(columns)

        if isinstance(df, pd.DataFrame):
            self.from_pandas(df, **kwargs)
        elif df is None:
            pass
        else:
            self.from_array(df, self.columns, **kwargs)

    def from_pandas(self, df, **kwargs):
        col_names = kwargs.get("columns", df.columns)
        col_alias_dict = kwargs.get("alias_dict", None)
        col_units = kwargs.get("units", None)
        self._df = ddf.from_pandas(data=df, columns=col_names, **kwargs)
        self._columns = Columns(col_names, col_units, col_alias_dict)

    def from_array(self, df, columns, **kwargs):
        col_alias_dict = kwargs.get("alias_dict", None)
        col_units = kwargs.get("units", None)
        self._df = ddf.from_array(df, columns=columns, **kwargs)
        self._columns = Columns(columns, col_units, col_alias_dict)

    @property
    def df(self):
        assert self.columns._check_consistency(self._df)
        return self._df

    def __getitem__(self, key: str):
        return self._df[key]

    # def __setitem__(self, key:str, item:Any):
    #     self._df[key] = item

    def add_column(self, key, values, alias=None):
        self._df[key] = values

    def __repr__(self):
        return repr(self._df)

    def __len__(self):
        return len(self._df)

    def copy(self):
        # TODO: expand to create a full copy of the instance
        return self._df.copy()

    def has_column(self, key: str):
        return key in self._df.columns

    def columns(self):
        return self._df.columns()

    def compute(self):
        self._df = dask.compute(self._df)

    def multifilter(self, inplace=True, **kwargs):
        """Apply filtering along multiple columns

        Kwargs:
            col[str]: (lower_bound[float],upper_bound[float])
                express no bound by passing np.inf or -np.inf in upper and lower bounds
                respectively.
        Raises:
            AttributeError: keyword name is not a column in the dataframe
        """
        cols_not_found = [k for k in kwargs.keys() if not self.has_column(k)]
        if len(cols_not_found) > 0:
            raise AttributeError(
                f"columns {cols_not_found} not found in the dataframe.",
            )
        for col, bounds in kwargs.items():
            lower, upper = bounds
            self.filter(col=col, lower_bound=lower, upper_bound=upper)

    def filter(
        self,
        col: str,
        lower_bound: float = -np.inf,
        upper_bound=np.inf,
        inplace=True,
    ) -> None:
        out_df = self._df[
            (self._df[col] > lower_bound) & (self._df[col] < upper_bound)
        ]
        if inplace:
            self._df = out_df
        else:
            return out_df
