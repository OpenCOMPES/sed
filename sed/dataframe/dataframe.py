from __future__ import annotations

import copy
from typing import Any
from typing import Sequence
from warnings import warn

import dask.array as dda
import dask.dataframe as ddf
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar


class Column:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get("name", None)
        self.alias = kwargs.get("alias", None)
        self.unit = kwargs.get("unit", None)
        self.metadata = kwargs.get("metadata", None)


class DataFrame:
    """Single event dataframe, expanding functionalities on top of pandas/dask.

    Incoroporates the ability of using aliases for accessing columns.
    """

    def __init__(
        self,
        df: pd.DataFrame | ddf.DataFrame | np.ndarray = None,
        columns: Sequence = None,
        alias_dict: dict = None,
        chunksize: int = None,
        npartitions: int = None,
        **kwargs,
    ) -> None:
        self._df = None
        self._columns = {
            "col1": {
                "unit": "fs",
                "name": "delayStage",
                "alias": "OPTICAL_DELAY_STAGE",
                "col_number": 2,
            },
        }

        if isinstance(df, pd.DataFrame):
            self.from_pandas(df, chunksize, npartitions, **kwargs)
        elif isinstance(df, np.ndarray):
            self.from_array(df, columns=columns, chunksize=chunksize, **kwargs)
        elif isinstance(df, dda.Array):
            self.from_dask_array(df, columns=columns, **kwargs)
        elif isinstance(df, ddf.DataFrame):
            self.from_dask_dataframe(df)
        self._alias = alias_dict if alias_dict is not None else {}

    def from_pandas(
        self,
        df: pd.DataFrame,
        chunksize: int = None,
        npartitions: int = None,
        name: str = None,
        **kwargs,
    ) -> None:
        """Load from a pandas dataframe

        if any of chunksize or npartitions are provided, a dask dataframe is generated.
        Otherwise an in-memory pandas dataframe.

        Args:
            df: pandas dataframe
            chunksize: number of rows per each partition. Defaults to None.
            npartitions: number of partitions. Defaults to None.
            kwargs: other arguments passed to dask.dataframe.from_pandas
        """

        if chunksize is not None or npartitions is not None:
            self._df = ddf.from_pandas(
                data=df,
                npartitions=npartitions,
                chunksize=chunksize,
                name=name,
                **kwargs,
            )
        else:
            self._df = df
            if name is not None:
                df.name = name

    def from_dask_dataframe(
        self,
        df: ddf.DataFrame,
    ) -> None:
        """Load from a dask dataframe

        Args:
            df: dask dataframe
        """
        self._df = df

    def from_array(
        self,
        arr: np.ndarray,
        columns: Sequence = None,
        chunksize: int = None,
        index: np.ndarray = None,
        **kwargs,
    ) -> None:
        """Generate dataframe from a numpy array.

        if any of chunksize or npartitions are provided, a dask dataframe is generated.
        Otherwise, an in-memory pandas dataframe is.

        Args:
            arr: 2D NxM numpy array
            columns: name of the M columns
            chunksize: number of rows per each partition. Defaults to None.
            npartitions: number of partitions. Defaults to None.
            kwargs: other arguments passed to `dask.dataframe.from_array`.

        Raises:
            ValueError: if the array passed is not 2D.
        """
        if columns is None:
            raise ValueError(
                "Must provide column names if loading a numpy array",
            )
        if arr.ndim != 2:
            raise ValueError(
                "The array must be 2 dimensional to generate a table.",
            )
        if chunksize is not None:
            self._df = ddf.from_array(arr, chunksize=chunksize, **kwargs)
        else:
            self._df = pd.DataFrame(
                data=arr,
                columns=columns,
                index=index,
                **kwargs,
            )

    def from_dask_array(
        self,
        arr: np.ndarray | dda.Array,
        columns: Sequence = None,
        **kwargs,
    ) -> None:
        """Generate dataframe from a dask array.

        if any of chunksize or npartitions are provided, a dask dataframe is generated.
        Otherwise an in-memory pandas dataframe.

        Args:
            arr: 2D dask array
            kwargs: other arguments passed to `dask.dataframe.from_dask_array`.

        Raises:
            ValueError: if the array passed is not 2D.
        """
        if columns is None:
            raise ValueError(
                "Must provide column names if loading a dask array",
            )
        self._df = ddf.from_dask_array(arr, **kwargs)

    @property
    def columns(self) -> list:
        return self._df.columns

    @property
    def alias(self) -> dict:
        return self._alias

    @property
    def alias_inv(self) -> dict:
        inv = {v: k for k, v in self.alias.items()}
        return inv

    def set_alias(self, column: str, alias: str) -> None:
        if column in self.alias.values:
            raise ValueError(
                f"{column} already has an alias as {self.alias_inv[column]}",
            )
        # two aliases for a single column is ok? maybe...
        if alias in self.alias.keys():
            raise ValueError(
                f"{alias} is already an alias for the column {self.alias[alias]}",
            )
        else:
            self._alias[alias] = column

    def remove_alias(self, name: str) -> None:
        """remove an alias pair given alias or column name

        Args:
            name: column name or alias
        """
        if name in self.alias.keys():
            _ = self._alias.pop(name)
        elif name in self.alias_inv.keys():
            _ = self._alias.pop(self.alias_inv[name])

    def __getitem__(self, key: str):
        """get a column by its name or by its alias"""
        if isinstance(key, str):
            key = [key]
        columns = [self._get_col_from_alias_or_col(k) for k in key]
        df = self._df[columns]
        return DataFrame(
            df,
            alias_dict={k: v for k, v in self.alias.items() if v in columns},
        )

    def _get_col_from_alias_or_col(self, key: str) -> str:
        if key in self.columns:
            return key
        elif key in self.alias:
            return self.alias[key]
        else:
            raise KeyError(f"{key} is not a column nor an alias for one.")

    def __setitem__(self, key: str, item: Any):
        self._df[key] = item

    def __repr__(self) -> str:
        return repr(self._df)

    def _repr_html_(self) -> str:
        try:
            return self._df._repr_html_()
        except AttributeError:
            return repr(self._df)

    def __str__(self) -> str:
        return str(self._df)

    def __len__(self) -> int:
        return len(self._df)

    def __eq__(self, other):
        if not isinstance(other, DataFrame):
            return False
        if not all(self._df == other._df):
            return False
        if self.alias != other.alias:
            return False
        else:
            return True

    @property
    def ncols(self) -> int:
        return len(self._df.columns)

    @property
    def values(self) -> np.ndarray:
        return self._df.values

    def compute(self, inplace=True, ret=False, pbar=True) -> None:
        if isinstance(self._df, ddf.DataFrame):
            if pbar:
                with ProgressBar():
                    df = self._df.compute()
            else:
                df = self._df.compute()
        else:
            df = self._df
            warn("Dataframe is not lazy, no need to compute it.")
        if inplace:
            self._df = df
        if ret:
            return DataFrame(
                df,
                alias_dict={
                    k: v for k, v in self.alias.items() if v in df.columns
                },
            )

    def copy(self) -> DataFrame:
        return copy.copy(self)

    def deepcopy(self) -> DataFrame:
        return copy.deepcopy(self)

    def multifilter(self, inplace=True, **kwargs) -> None:
        """Apply filtering along multiple columns

        Kwargs:
            col[str]: (lower_bound[float],upper_bound[float])
                express no bound by passing np.inf or -np.inf in upper and lower bounds
                respectively.
        Raises:
            AttributeError: keyword name is not a column in the dataframe
        """
        # TODO: add testing
        cols_not_found = [k for k in kwargs.keys() if not self.has_column(k)]
        if len(cols_not_found) > 0:
            raise AttributeError(
                f"columns {cols_not_found} not found in the dataframe.",
            )
        for col, bounds in kwargs.items():
            lower, upper = bounds
            self.filter(
                col=col,
                lower_bound=lower,
                upper_bound=upper,
                inplace=inplace,
            )

    def filter(
        self,
        col: str,
        lower_bound: float = -np.inf,
        upper_bound=np.inf,
        inplace=True,
    ) -> None:
        # TODO: add testing
        out_df = self[(self[col] > lower_bound) & (self[col] < upper_bound)]
        if inplace:
            self._df = out_df
        return out_df
