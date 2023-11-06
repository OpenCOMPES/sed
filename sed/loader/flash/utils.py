"""Helper functions for the flash loader."""
from typing import Union

import dask.dataframe
import numpy as np
import pandas as pd

from ..utils import split_channel_bitwise


def split_dld_time_from_sector_id(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    tof_column: str = None,
    sector_id_column: str = None,
    sector_id_reserved_bits: int = None,
    config: dict = None,
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """Converts the 8s time in steps to time in steps and sectorID.

    The 8s detector encodes the dldSectorID in the 3 least significant bits of the
    dldTimeSteps channel.

    Args:
        df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to use.
        tof_column (str, optional): Name of the column containing the
            time-of-flight steps. Defaults to config["dataframe"]["tof_column"].
        sector_id_column (str, optional): Name of the column containing the
            sectorID. Defaults to config["dataframe"]["sector_id_column"].
        sector_id_reserved_bits (int, optional): Number of bits reserved for the
        config (dict, optional): Configuration dictionary. Defaults to None.

    Returns:
        Union[pd.DataFrame, dask.dataframe.DataFrame]: Dataframe with the new columns.
    """
    if tof_column is None:
        if config is None:
            raise ValueError("Either tof_column or config must be given.")
        tof_column = config["dataframe"]["tof_column"]
    if sector_id_column is None:
        if config is None:
            raise ValueError("Either sector_id_column or config must be given.")
        sector_id_column = config["dataframe"]["sector_id_column"]
    if sector_id_reserved_bits is None:
        if config is None:
            raise ValueError("Either sector_id_reserved_bits or config must be given.")
        sector_id_reserved_bits = config["dataframe"].get("sector_id_reserved_bits", None)
        if sector_id_reserved_bits is None:
            raise ValueError('No value for "sector_id_reserved_bits" found in config.')

    if sector_id_column in df.columns:
        raise ValueError(
            f"Column {sector_id_column} already in dataframe. This function is not idempotent.",
        )
    df = split_channel_bitwise(
        df=df,
        input_column=tof_column,
        output_columns=[sector_id_column, tof_column],
        bit_mask=sector_id_reserved_bits,
        overwrite=True,
        types=[np.int8, np.int32],
    )
    return df
