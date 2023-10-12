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
from dask.diagnostics import ProgressBar


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


    df[tof_ns_column] = df.map_partitions(
        step2ns, meta=(tof_column, np.float64)
    )
    metadata: Dict[str,Any] = {
        "applied": True,
        "tof_binwidth": tof_binwidth
    }
    return df, metadata


def rolling_average_on_acquisition_time(
        df: Union[pd.DataFrame, dask.dataframe.DataFrame], 
        rolling_group_channel: str,
        columns: str = None, 
        window: float = None, 
        sigma: float = 2,
        config: dict = None,

) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """ Perform a rolling average with a gaussian weighted window.
    
    In order to preserve the number of points, the first and last "widnow"
    number of points are substituted with the original signal.
    # TODO: this is currently very slow, and could do with a remake.

    Args:
        df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to use.
        group_channel: (str): Name of the column on which to group the data
        cols (str): Name of the column on which to perform the rolling average
        window (float): Size of the rolling average window
        sigma (float): number of standard deviations for the gaussian weighting of the window. 
            a value of 2 corresponds to a gaussian with sigma equal to half the window size.
            Smaller values reduce the weighting in the window frame.

    Returns:
        Union[pd.DataFrame, dask.dataframe.DataFrame]: Dataframe with the new columns.
    """
    if rolling_group_channel is None:
        if config is None:
            raise ValueError("Either group_channel or config must be given.")
        rolling_group_channel = config["dataframe"]["rolling_group_channel"]        
    with ProgressBar():
        print(f'rolling average over {columns}...')
        if isinstance(columns,str):
            columns=[columns]
        df_ = df.groupby(rolling_group_channel).agg({c:'mean' for c in columns}).compute()
        df_['dt'] = pd.to_datetime(df_.index, unit='s')
        df_['ts'] = df_.index
        for c in columns:
            df_[c+'_rolled'] = df_[c].interpolate(
                method='nearest'
            ).rolling(
                window,center=True,win_type='gaussian'
            ).mean(
                std=window/sigma
            ).fillna(df_[c])
            df_ = df_.drop(c, axis=1)
            if c+'_rolled' in df.columns:
                df = df.drop(c+'_rolled',axis=1)
    return df.merge(df_,left_on='timeStamp',right_on='ts').drop(['ts','dt'], axis=1)


def shift_energy_axis(
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        columns: Union[str,Sequence[str]],
        signs: Union[int,Sequence[int]],
        energy_column: str = None,
        mode: Union[str,Sequence[str]] = "direct",
        window: float = None,
        sigma: float = 2,
        rolling_group_channel: str = None,
        config: dict = None, 
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """ Apply an energy shift to the given column(s).
    
    Args: 
        df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to use.
        energy_column (str): Name of the column containing the energy values.
        column_name (Union[str,Sequence[str]]): Name of the column(s) to apply the shift to.
        sign (Union[int,Sequence[int]]): Sign of the shift to apply. (+1 or -1)
        mode (str): The mode of the shift. One of 'direct', 'average' or rolled.
            if rolled, window and sigma must be given.
        config (dict): Configuration dictionary.
        **kwargs: Additional arguments for the rolling average function.
    """
    if energy_column is None:
        if config is None:
            raise ValueError("Either energy_column or config must be given.")
        energy_column = config["dataframe"]["energy_column"]
    
    if isinstance(columns,str):
        columns=[columns]
    if isinstance(signs,int):
        signs=[signs]
    if isinstance(mode,str):
        mode = [mode] * len(columns)
    if len(columns) != len(signs):
        raise ValueError("column_name and sign must have the same length.")
    with ProgressBar(minimum=5,):
        if mode == "rolled":
            if window is None:
                if config is None:
                    raise ValueError("Either window or config must be given.")
                window = config["dataframe"]["rolling_window"]
            if sigma is None:
                if config is None:
                    raise ValueError("Either sigma or config must be given.")
                sigma = config["dataframe"]["rolling_sigma"]
            if rolling_group_channel is None:
                if config is None:
                    raise ValueError("Either rolling_group_channel or config must be given.")
                rolling_group_channel = config["dataframe"].get("rolling_group_channel",None)
            if rolling_group_channel is None:
                raise ValueError("T f mode is 'rolled', rolling_group_channel must be"
                                 "given or present in config.")
            print('rolling averages...')
            df = rolling_average_on_acquisition_time(
                df,
                rolling_group_channel=rolling_group_channel,
                columns=columns,
                window=window,
                sigma=sigma,
            )
        for col, s, m in zip(columns,signs, mode):
            s = s/np.abs(s) # enusre s is either +1 or -1
            if m == "rolled":
                col = col + '_rolled'
            if m == "direct" or m == "rolled":
                df[col] = df.map_partitions(
                    lambda x: x[col] + s * x[energy_column], meta=(col, np.float32)
                )
            elif m == 'mean':
                print('computing means...')
                col_mean = df[col].mean()
                df[col] = df.map_partitions(
                    lambda x: x[col] + s * (x[energy_column] - col_mean), meta=(col, np.float32)
                )
            else:
                raise ValueError(f"mode must be one of 'direct', 'mean' or 'rolled'. Got {m}.")
    metadata: dict[str,Any] = {
        "applied": True,
        "energy_column": energy_column,
        "column_name": columns,
        "sign": signs,
    }    
    return df, metadata


def step2ns(
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        tof_column: str, 
        tof_binwidth: float, 
        tof_binning: int,
        dtype: type = np.float64, 
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """ Converts the time-of-flight steps to time-of-flight in nanoseconds.
    
    designed for use with dask.dataframe.DataFrame.map_partitions.

    Args:
        df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to convert.
        tof_column (str): Name of the column containing the time-of-flight steps.
        tof_binwidth (float): Time step size in nanoseconds.
        tof_binning (int): Binning of the time-of-flight steps.
        
    Returns:
        Union[pd.DataFrame, dask.dataframe.DataFrame]: Dataframe with the new column.
    """
    val = df[tof_column].astype(dtype) * tof_binwidth * 2**tof_binning
    return val.astype(dtype)
