"""sed.calibrator.hextof module. Code for handling hextof specific transformations and
calibrations.
"""
from typing import Any
from typing import Dict
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
        tof_column (str, optional): Name of the column containing the
            time-of-flight steps. Defaults to config["dataframe"]["tof_column"].
        sector_id_column (str, optional): Name of the column containing the
            sectorID. Defaults to config["dataframe"]["sector_id_column"].
        config (dict, optional): Configuration dictionary. Defaults to None.

    Returns:
        dask.dataframe.DataFrame: Dataframe with the new columns.
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
        sector_id_column (str, optional): Name of the column containing the
            sectorID. Defaults to config["dataframe"]["sector_id_column"].
        tof_column (str, optional): Name of the column containing the
            time-of-flight. Defaults to config["dataframe"]["tof_column"].
        config (dict, optional): Configuration dictionary. Defaults to None.
    
    Returns:
        dask.dataframe.DataFrame: Dataframe with the new columns.
        dict: Metadata dictionary.
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
    metadata: Dict[str,Any] = {
        "applied": True,
        "sector_delays": sector_delays,
    }
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
    
    Returns:
        dask.dataframe.DataFrame: Dataframe with the new columns.
        dict: Metadata dictionary.
    """
    if tof_binwidth is None:
        if config is None:
            raise ValueError("Either tof_binwidth or config must be given.")
        tof_binwidth = config["dataframe"]["tof_binwidth"]
    if tof_column is None:
        if config is None:
            raise ValueError("Either tof_column or config must be given.")
        tof_column = config["dataframe"]["tof_column"]
    if tof_binning is None:
        if config is None:
            raise ValueError("Either tof_binning or config must be given.")
        tof_binning = config["dataframe"]["tof_binning"]
    if tof_ns_column is None:
        if config is None:
            raise ValueError("Either tof_ns_column or config must be given.")
        tof_ns_column = config["dataframe"]["tof_ns_column"]

    def convert_to_ns(x):
        val = x[tof_column] * tof_binwidth * 2**tof_binning
        return val.astype(np.float32)
    df[tof_ns_column] = df.map_partitions(
        convert_to_ns, meta=(tof_column, np.float32)
    )
    metadata: Dict[str,Any] = {
        "applied": True,
        "tof_binwidth": tof_binwidth
    }
    return df, metadata


def calibrate_k_division_model(
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        warp_params: Sequence[float] = None,
        x_column: str = None,
        y_column: str = None,
        kx_column: str = None,
        ky_column: str = None,
        config: dict = None,
) -> Tuple[Union[pd.DataFrame, dask.dataframe.DataFrame], dict]:
    """ K calibration based on the division model
    
    This function returns the distorted coordinates given the undistorted ones
    a little complicated by the fact that (warp_params[6],warp_params[7]) needs to 
    go to (0,0)
    it uses a radial distortion model called division model 
    (https://en.wikipedia.org/wiki/Distortion_(optics)#Software_correction)
    commonly used to correct for lens artifacts.

    Args:
        warp_params (Sequence[float], optional): warping parameters.
            warp_params[0],warp_params[1] center of distortion in px
            warp_params[6],warp_params[7] normal emission (Gamma) in px
            warp_params[2],warp_params[3], warp_params[4] K_n; rk = rpx/(K_0 + K_1*rpx^2 + K_2*rpx^4)
            warp_params[5] rotation in rad
            Defaults to config["dataframe"]["warp_params"].
        x_column (str, optional): Name of the column containing the
            x steps. Defaults to config["dataframe"]["x_column"].
        y_column (str, optional): Name of the column containing the
            y steps. Defaults to config["dataframe"]["y_column"].
        kx_column (str, optional): Name of the column containing the
            x steps. Defaults to config["dataframe"]["kx_column"].
        ky_column (str, optional): Name of the column containing the
            y steps. Defaults to config["dataframe"]["ky_column"].
    """
    if warp_params is None:
        if config is None:
            raise ValueError("Either warp_params or config must be given.")
        warp_params: float = config["dataframe"]["warp_params"]
    if x_column is None:
        if config is None:
            raise ValueError("Either x_column or config must be given.")
        x_column: str = config["dataframe"]["x_column"]
    if kx_column is None:
        if config is None:
            raise ValueError("Either kx_column or config must be given.")
        kx_column: str = config["dataframe"]["kx_column"]
    if y_column is None:
        if config is None:
            raise ValueError("Either y_column or config must be given.")
        y_column: str = config["dataframe"]["y_column"]
    if ky_column is None:
        if config is None:
            raise ValueError("Either ky_column or config must be given.")
        ky_column: str = config["dataframe"]["ky_column"]

    wp = warp_params
     def convert_to_kx(x):
        """Converts the x steps to kx."""
        x_diff = x[x_column] - wp[0]
        y_diff = x[y_column] - wp[1]
        dist = np.sqrt(x_diff**2 + y_diff**2)
        den = wp[2] + wp[3]*dist**2 + wp[4]*dist**4
        angle = np.arctan2(y_diff, x_diff) - wp[5]
        warp_diff = np.sqrt((wp[6] - wp[0])**2 + (wp[7] - wp[1])**2)
        warp_den = wp[2] + wp[3]*(wp[6] - wp[0])**2 + wp[4]*(wp[7] - wp[1])**2
        warp_angle = np.arctan2(wp[7] - wp[1], wp[6] - wp[0]) - wp[5]
        return (dist/den)*np.cos(angle) - (warp_diff/warp_den)*np.cos(warp_angle)

    def convert_to_ky(x):
        x_diff = x[x_column] - wp[0]
        y_diff = x[y_column] - wp[1]
        dist = np.sqrt(x_diff**2 + y_diff**2)
        den = wp[2] + wp[3]*dist**2 + wp[4]*dist**4
        angle = np.arctan2(y_diff, x_diff) - wp[5]
        warp_diff = np.sqrt((wp[6] - wp[0])**2 + (wp[7] - wp[1])**2)
        warp_den = wp[2] + wp[3]*(wp[6] - wp[0])**2 + wp[4]*(wp[7] - wp[1])**2
        warp_angle = np.arctan2(wp[7] - wp[1], wp[6] - wp[0]) - wp[5]
        return (dist/den)*np.sin(angle) - (warp_diff/warp_den)*np.sin(warp_angle)
    
    df[kx_column] = df.map_partitions(
        convert_to_kx, meta=(kx_column, np.float64)
    )
    df[ky_column] = df.map_partitions(
        convert_to_ky, meta=(ky_column, np.float64)
    )

    metadata = {
        "applied": True,
        "warp_params": warp_params,
    }
    return df, metadata
