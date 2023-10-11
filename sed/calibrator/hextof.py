"""sed.calibrator.hextof module. Code for handling hextof specific transformations and
calibrations.
"""
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import dask.dataframe


def unravel_8s_detector_time_channel(
    df: dask.dataframe.DataFrame,
    tof_column: str = None,
    sector_id_column: str = None,
    config: dict = None,
) -> dask.dataframe.DataFrame:
    """Converts the 8s time in steps to time in steps and sectorID.

    The 8s detector encodes the dldSectorID in the 3 least significant bits of the
    dldTimeSteps channel.

    Args:
        sector_delays (Sequece[float], optional): Sector delays for the 8s time.
            Defaults to config["dataframe"]["sector_delays"].
    """
    if tof_column is None:
        if config is None:
            raise ValueError("Either tof_column or config must be given.")
        tof_column = config["dataframe"]["tof_column"]
    if sector_id_column is None:
        if config is None:
            raise ValueError("Either sector_id_column or config must be given.")
        sector_id_column = config["dataframe"]["sector_id_column"]

    if sector_id_column in df.columns:
        raise ValueError(f"Column {sector_id_column} already in dataframe. "
                         "This function is not idempotent.")
    df[sector_id_column] = (df[tof_column] % 8).astype(np.int8)
    df[tof_column] = (df[tof_column] // 8).astype(np.int32)
    return df


def align_dld_sectors(
        df: dask.dataframe.DataFrame,
        sector_delays: Sequence[float] = None,
        sector_id_column: str = None,
        tof_column: str = None,
        config: dict = None,
) -> Tuple[Union[pd.DataFrame, dask.dataframe.DataFrame], dict]:
    """Aligns the 8s sectors to the first sector.

    Args:
        sector_delays (Sequece[float], optional): Sector delays for the 8s time.
        in units of step. Calibration should be done with binning along dldTimeSteps.
        Defaults to config["dataframe"]["sector_delays"].
    """
    if sector_delays is None:
        if config is None:
            raise ValueError("Either sector_delays or config must be given.")
        sector_delays = config["dataframe"]["sector_delays"]
    if sector_id_column is None:
        if config is None:
            raise ValueError("Either sector_id_column or config must be given.")
        sector_id_column = config["dataframe"]["sector_id_column"]
    if tof_column is None:
        if config is None:
            raise ValueError("Either tof_column or config must be given.")
        tof_column = config["dataframe"]["tof_column"]
    # align the 8s sectors
    sector_delays_arr = dask.array.from_array(sector_delays)

    def align_sector(x):
        val = x[tof_column] - sector_delays_arr[x[sector_id_column].values.astype(int)]
        return val.astype(np.float32)
    df[tof_column] = df.map_partitions(
        align_sector, meta=(tof_column, np.float32)
    )

    metadata = {}
    metadata["applied"] = True
    metadata["sector_delays"] = sector_delays

    return df, metadata


def dld_time_to_ns(
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        tof_ns_column: str = None,
        tof_binwidth: float = None,
        tof_column: str = None,
        tof_binning: int = None,
        config: dict = None,
) -> Tuple[Union[pd.DataFrame, dask.dataframe.DataFrame], dict]:
    """Converts the 8s time in steps to time in ns.

    Args:
        tof_binwidth (float, optional): Time step size in nanoseconds.
            Defaults to config["dataframe"]["tof_binwidth"].
        tof_column (str, optional): Name of the column containing the
            time-of-flight steps. Defaults to config["dataframe"]["tof_column"].
        tof_column (str, optional): Name of the column containing the
            time-of-flight. Defaults to config["dataframe"]["tof_column"].
        tof_binning (int, optional): Binning of the time-of-flight steps.
    """
    if tof_binwidth is None:
        if config is None:
            raise ValueError("Either tof_binwidth or config must be given.")
        tof_binwidth: float = config["dataframe"]["tof_binwidth"]
    if tof_column is None:
        if config is None:
            raise ValueError("Either tof_column or config must be given.")
        tof_column: str = config["dataframe"]["tof_column"]
    if tof_binning is None:
        if config is None:
            raise ValueError("Either tof_binning or config must be given.")
        tof_binning: int = config["dataframe"]["tof_binning"]
    if tof_ns_column is None:
        if config is None:
            raise ValueError("Either tof_ns_column or config must be given.")
        tof_ns_column: str = config["dataframe"]["tof_ns_column"]

    def convert_to_ns(x):
        val = x[tof_column] * tof_binwidth * 2**tof_binning
        return val.astype(np.float32)
    df[tof_ns_column] = df.map_partitions(
        convert_to_ns, meta=(tof_column, np.float32)
    )
    metadata = {}
    metadata["applied"] = True
    metadata["tof_binwidth"] = tof_binwidth

    return df, metadata
