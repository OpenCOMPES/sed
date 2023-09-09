"""sed.calibrator.delay module. Code for delay calibration.
"""
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import dask.dataframe
import h5py
import numpy as np
import pandas as pd


class DelayCalibrator:
    """
    Pump-Probe delay calibration methods.
    Initialization of the DelayCalibrator class passes the config.

    Args:
        config (dict, optional): Config dictionary. Defaults to None.
    """

    def __init__(
        self,
        config: dict = None,
    ):
        """Initialization of the DelayCalibrator class passes the config.

        Args:
            config (dict, optional): Config dictionary. Defaults to None.
        """
        if config is None:
            config = {}

        self._config = config

        self.adc_column = self._config["dataframe"]["adc_column"]
        self.delay_column = self._config["dataframe"]["delay_column"]
        self.calibration: Dict[Any, Any] = {}

    def append_delay_axis(
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        adc_column: str = None,
        delay_column: str = None,
        calibration: dict = None,
        adc_range: Union[Tuple, List, np.ndarray] = None,
        delay_range: Union[Tuple, List, np.ndarray] = None,
        time0: float = None,
        delay_range_mm: Union[Tuple, List, np.ndarray] = None,
        datafile: str = None,
        p1_key: str = None,
        p2_key: str = None,
        t0_key: str = None,
    ) -> Tuple[Union[pd.DataFrame, dask.dataframe.DataFrame], dict]:
        """Calculate and append the delay axis to the events dataframe, by converting
        values from an analog-digital-converter (ADC).

        Args:
            df (Union[pd.DataFrame, dask.dataframe.DataFrame]): The dataframe where
                to apply the delay calibration to.
            adc_column (str, optional): Source column for delay calibration.
                Defaults to config["dataframe"]["adc_column"].
            delay_column (str, optional): Destination column for delay calibration.
                Defaults to config["dataframe"]["delay_column"].
            calibration (dict, optional): Calibration dictionary with parameters for
                delay calibration.
            adc_range (Union[Tuple, List, np.ndarray], optional): The range of used
                ADC values. Defaults to config["delay"]["adc_range"].
            delay_range (Union[Tuple, List, np.ndarray], optional): Range of scanned
                delay values in ps. If omitted, the range is calculated from the
                delay_range_mm and t0 values.
            time0 (float, optional): Pump-Probe overlap value of the delay coordinate.
                If omitted, it is searched for in the data files.
            delay_range_mm (Union[Tuple, List, np.ndarray], optional): Range of scanned
                delay stage in mm. If omitted, it is searched for in the data files.
            datafile (str, optional): Datafile in which delay parameters are searched
                for. Defaults to None.
            p1_key (str, optional): hdf5 key for delay_range_mm start value.
                Defaults to config["delay"]["p1_key"]
            p2_key (str, optional): hdf5 key for delay_range_mm end value.
                Defaults to config["delay"]["p2_key"]
            t0_key (str, optional): hdf5 key for t0 value (mm).
                Defaults to config["delay"]["t0_key"]

        Raises:
            ValueError: Raised if delay parameters are not found in the file.
            NotImplementedError: Raised if no sufficient information passed.

        Returns:
            Union[pd.DataFrame, dask.dataframe.DataFrame]: dataframe with added column
            and delay calibration metdata dictionary.
        """
        # pylint: disable=duplicate-code
        if calibration is None:
            if self.calibration:
                calibration = deepcopy(self.calibration)
            else:
                calibration = deepcopy(
                    self._config["delay"].get(
                        "calibration",
                        {},
                    ),
                )

        if adc_range is not None:
            calibration["adc_range"] = adc_range
        if delay_range is not None:
            calibration["delay_range"] = delay_range
        if time0 is not None:
            calibration["time0"] = time0
        if delay_range_mm is not None:
            calibration["delay_range_mm"] = delay_range_mm

        if adc_column is None:
            adc_column = self.adc_column
        if delay_column is None:
            delay_column = self.delay_column
        if p1_key is None:
            p1_key = self._config["delay"].get("p1_key", "")
        if p2_key is None:
            p2_key = self._config["delay"].get("p2_key", "")
        if t0_key is None:
            t0_key = self._config["delay"].get("t0_key", "")

        if "adc_range" not in calibration.keys():
            calibration["adc_range"] = np.asarray(
                self._config["delay"]["adc_range"],
            ) / 2 ** (self._config["dataframe"]["adc_binning"] - 1)

        if "delay_range" not in calibration.keys():
            if "delay_range_mm" not in calibration.keys() or "time0" not in calibration.keys():
                if datafile is not None and p1_key and p2_key and t0_key:
                    try:
                        ret = extract_delay_stage_parameters(
                            datafile,
                            p1_key,
                            p2_key,
                            t0_key,
                        )
                    except KeyError as exc:
                        raise ValueError(
                            "Delay stage values not found in file",
                        ) from exc
                    calibration["datafile"] = datafile
                    calibration["delay_range_mm"] = (ret[0], ret[1])
                    calibration["time0"] = ret[2]
                    print(f"Extract delay range from file '{datafile}'.")
                else:
                    raise NotImplementedError(
                        "Not enough parameters for delay calibration.",
                    )

            calibration["delay_range"] = np.asarray(
                mm_to_ps(
                    np.asarray(calibration["delay_range_mm"]),
                    calibration["time0"],
                ),
            )
            print(f"delay_range (ps) = {calibration['delay_range']}")

        if "delay_range" in calibration.keys():
            df[delay_column] = calibration["delay_range"][0] + (
                df[adc_column] - calibration["adc_range"][0]
            ) * (calibration["delay_range"][1] - calibration["delay_range"][0]) / (
                calibration["adc_range"][1] - calibration["adc_range"][0]
            )
        else:
            raise NotImplementedError

        metadata = {"calibration": calibration}

        return df, metadata


def extract_delay_stage_parameters(
    file: str,
    p1_key: str,
    p2_key: str,
    t0_key: str,
) -> Tuple:
    """
    Read delay stage ranges from hdf5 file

    Parameters:
        file: filename
        p1_key: hdf5 path to the start of the scan range
        p2_key: hdf5 path to the end of the scan range
        t0_key: hdf5 path to the t0 value

    Returns:
        (p1_value, p2_value, t0_value)
    """
    with h5py.File(file, "r") as file_handle:
        values = []
        for key in [p1_key, p2_key, t0_key]:
            if key[0] == "@":
                values.append(file_handle.attrs[key[1:]])
            else:
                values.append(file_handle[p1_key])

        return tuple(values)


def mm_to_ps(
    delay_mm: Union[float, np.ndarray],
    time0_mm: float,
) -> Union[float, np.ndarray]:
    """Converts a delaystage position in mm into a relative delay in picoseconds
    (double pass).

    Args:
        delay_mm (Union[float, Sequence[float]]):
            Delay stage position in mm
        time0_mm (_type_):
            Delay stage position of pump-probe overlap in mm

    Returns:
        Union[float, Sequence[float]]:
            Relative delay in picoseconds
    """
    delay_ps = (delay_mm - time0_mm) / 0.15
    return delay_ps
