"""sed.calibrator.delay module. Code for delay calibration.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime
from typing import Any

import dask.dataframe
import h5py
import numpy as np
import pandas as pd

from sed.core import dfops
from sed.core.logging import set_verbosity
from sed.core.logging import setup_logging

# Configure logging
logger = setup_logging("delay")


class DelayCalibrator:
    """
    Pump-Probe delay calibration methods.
    Initialization of the DelayCalibrator class passes the config.

    Args:
        config (dict, optional): Config dictionary. Defaults to None.
        verbose (bool, optional): Option to print out diagnostic information.
            Defaults to True.
    """

    def __init__(
        self,
        config: dict = None,
        verbose: bool = True,
    ) -> None:
        """Initialization of the DelayCalibrator class passes the config.

        Args:
            config (dict, optional): Config dictionary. Defaults to None.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to True.
        """
        if config is not None:
            self._config = config
        else:
            self._config = {}

        self._verbose = verbose
        set_verbosity(logger, self._verbose)

        self.adc_column: str = config["dataframe"]["columns"]["adc"]
        self.delay_column: str = config["dataframe"]["columns"]["delay"]
        self.corrected_delay_column = self._config["dataframe"]["columns"].get(
            "corrected_delay",
            self.delay_column,
        )
        self.calibration: dict[str, Any] = self._config["delay"].get("calibration", {})
        self.offsets: dict[str, Any] = self._config["delay"].get("offsets", {})

    @property
    def verbose(self) -> bool:
        """Accessor to the verbosity flag.

        Returns:
            bool: Verbosity flag.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """Setter for the verbosity.

        Args:
            verbose (bool): Option to turn on verbose output. Sets loglevel to INFO.
        """
        self._verbose = verbose
        set_verbosity(logger, self._verbose)

    def append_delay_axis(
        self,
        df: pd.DataFrame | dask.dataframe.DataFrame,
        adc_column: str = None,
        delay_column: str = None,
        calibration: dict[str, Any] = None,
        adc_range: tuple | list | np.ndarray = None,
        delay_range: tuple | list | np.ndarray = None,
        time0: float = None,
        delay_range_mm: tuple | list | np.ndarray = None,
        datafile: str = None,
        p1_key: str = None,
        p2_key: str = None,
        t0_key: str = None,
        suppress_output: bool = False,
    ) -> tuple[pd.DataFrame | dask.dataframe.DataFrame, dict]:
        """Calculate and append the delay axis to the events dataframe, by converting
        values from an analog-digital-converter (ADC).

        Args:
            df (pd.DataFrame | dask.dataframe.DataFrame): The dataframe where
                to apply the delay calibration to.
            adc_column (str, optional): Source column for delay calibration.
                Defaults to config["dataframe"]["columns"]["adc"].
            delay_column (str, optional): Destination column for delay calibration.
                Defaults to config["dataframe"]["columns"]["delay"].
            calibration (dict, optional): Calibration dictionary with parameters for
                delay calibration.
            adc_range (tuple | list | np.ndarray, optional): The range of used
                ADC values. Defaults to config["delay"]["adc_range"].
            delay_range (tuple | list | np.ndarray, optional): Range of scanned
                delay values in ps. If omitted, the range is calculated from the
                delay_range_mm and t0 values.
            time0 (float, optional): Pump-Probe overlap value of the delay coordinate.
                If omitted, it is searched for in the data files.
            delay_range_mm (tuple | list | np.ndarray, optional): Range of scanned
                delay stage in mm. If omitted, it is searched for in the data files.
            datafile (str, optional): Datafile in which delay parameters are searched
                for. Defaults to None.
            p1_key (str, optional): hdf5 key for delay_range_mm start value.
                Defaults to config["delay"]["p1_key"]
            p2_key (str, optional): hdf5 key for delay_range_mm end value.
                Defaults to config["delay"]["p2_key"]
            t0_key (str, optional): hdf5 key for t0 value (mm).
                Defaults to config["delay"]["t0_key"]
            suppress_output (bool, optional): Option to suppress log output. Defaults to False.

        Raises:
            ValueError: Raised if delay parameters are not found in the file.
            NotImplementedError: Raised if no sufficient information passed.

        Returns:
            tuple[pd.DataFrame | dask.dataframe.DataFrame, dict]: dataframe with added column
            and delay calibration metadata dictionary.
        """
        # pylint: disable=duplicate-code
        if calibration is None:
            calibration = deepcopy(self.calibration)

        if (
            adc_range is not None
            or delay_range is not None
            or time0 is not None
            or delay_range_mm is not None
            or datafile is not None
        ):
            calibration = {}
            calibration["creation_date"] = datetime.now()
            if adc_range is not None:
                calibration["adc_range"] = adc_range
            if delay_range is not None:
                calibration["delay_range"] = delay_range
            if time0 is not None:
                calibration["time0"] = time0
            if delay_range_mm is not None:
                calibration["delay_range_mm"] = delay_range_mm
        else:
            # report usage of loaded parameters
            if "creation_date" in calibration and not suppress_output:
                datestring = calibration["creation_date"].strftime("%m/%d/%Y, %H:%M:%S")
                logger.info(f"Using delay calibration parameters generated on {datestring}")

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
            calibration["adc_range"] = (
                np.asarray(self._config["delay"]["adc_range"])
                / self._config["dataframe"]["adc_binning"]
            )

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
                    if not suppress_output:
                        logger.info(f"Extract delay range from file '{datafile}'.")
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
            if not suppress_output:
                logger.info(f"Converted delay_range (ps) = {calibration['delay_range']}")
            calibration["creation_date"] = datetime.now()

        if "delay_range" in calibration.keys():
            df[delay_column] = calibration["delay_range"][0] + (
                df[adc_column] - calibration["adc_range"][0]
            ) * (calibration["delay_range"][1] - calibration["delay_range"][0]) / (
                calibration["adc_range"][1] - calibration["adc_range"][0]
            )
            self.calibration = deepcopy(calibration)
            if not suppress_output:
                logger.info(
                    "Append delay axis using delay_range = "
                    f"[{calibration['delay_range'][0]}, {calibration['delay_range'][1]}]"
                    " and adc_range = "
                    f"[{calibration['adc_range'][0]}, {calibration['adc_range'][1]}]",
                )
        else:
            raise NotImplementedError

        metadata = {"calibration": calibration}
        return df, metadata

    def add_offsets(
        self,
        df: dask.dataframe.DataFrame,
        offsets: dict[str, Any] = None,
        constant: float = None,
        flip_delay_axis: bool = None,
        columns: str | Sequence[str] = None,
        weights: float | Sequence[float] = 1.0,
        preserve_mean: bool | Sequence[bool] = False,
        reductions: str | Sequence[str] = None,
        delay_column: str = None,
        suppress_output: bool = False,
    ) -> tuple[dask.dataframe.DataFrame, dict]:
        """Apply an offset to the delay column based on a constant or other columns.

        Args:
            df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to use.
            offsets (dict, optional): Dictionary of delay offset parameters.
            constant (float, optional): The constant to shift the delay axis by.
            flip_delay_axis (bool, optional): Whether to flip the time axis. Defaults to False.
            columns (str | Sequence[str]): Name of the column(s) to apply the shift from.
            weights (float | Sequence[float]): weights to apply to the columns.
                Can also be used to flip the sign (e.g. -1). Defaults to 1.
            preserve_mean (bool | Sequence[bool]): Whether to subtract the mean of the column
                before applying the shift. Defaults to False.
            reductions (str | Sequence[str]): The reduction to apply to the column. Should be an
                available method of dask.dataframe.Series. For example "mean". In this case the
                function is applied to the column to generate a single value for the whole dataset.
                If None, the shift is applied per-dataframe-row. Defaults to None. Currently only
                "mean" is supported.
            delay_column (str, optional): Name of the column containing the delay values.
            suppress_output (bool, optional): Option to suppress log output. Defaults to False.

        Returns:
            tuple[dask.dataframe.DataFrame, dict]: Dataframe with the shifted delay axis and
            Metadata dictionary.
        """
        if offsets is None:
            offsets = deepcopy(self.offsets)

        if delay_column is None:
            delay_column = self.delay_column

        metadata: dict[str, Any] = {
            "applied": True,
        }

        if columns is not None or constant is not None or flip_delay_axis:
            # pylint:disable=duplicate-code
            # use passed parameters, overwrite config
            offsets = {}
            offsets["creation_date"] = datetime.now()
            # column-based offsets
            if columns is not None:
                offsets["columns"] = {}
                if weights is None:
                    weights = 1
                if isinstance(weights, (int, float, np.integer, np.floating)):
                    weights = [weights]
                if len(weights) == 1:
                    weights = [weights[0]] * len(columns)
                if not isinstance(weights, Sequence):
                    raise TypeError(
                        f"Invalid type for weights: {type(weights)}. Must be a number or sequence",
                    )
                if not all(isinstance(s, (int, float, np.integer, np.floating)) for s in weights):
                    raise TypeError(
                        f"Invalid type for weights: {type(weights)}. Must be a number or sequence",
                    )

                if isinstance(columns, str):
                    columns = [columns]
                if isinstance(preserve_mean, bool):
                    preserve_mean = [preserve_mean] * len(columns)
                if not isinstance(reductions, Sequence):
                    reductions = [reductions]
                if len(reductions) == 1:
                    reductions = [reductions[0]] * len(columns)

                # store in offsets dictionary
                for col, weight, pmean, red in zip(columns, weights, preserve_mean, reductions):
                    offsets["columns"][col] = {
                        "weight": weight,
                        "preserve_mean": pmean,
                        "reduction": red,
                    }

            # constant offset
            if isinstance(constant, (int, float, np.integer, np.floating)):
                offsets["constant"] = constant
            elif constant is not None:
                raise TypeError(f"Invalid type for constant: {type(constant)}")
            # flip the time direction
            if flip_delay_axis:
                offsets["flip_delay_axis"] = flip_delay_axis

        elif "creation_date" in offsets and not suppress_output:
            datestring = offsets["creation_date"].strftime("%m/%d/%Y, %H:%M:%S")
            logger.info(f"Using delay offset parameters generated on {datestring}")

        if len(offsets) > 0:
            # unpack dictionary
            columns = []
            weights = []
            preserve_mean = []
            reductions = []
            log_str = "Delay offset parameters:"
            for k, v in offsets.items():
                if k == "creation_date":
                    continue
                if k == "constant":
                    constant = v
                    log_str += f"\n   Constant: {constant}"
                elif k == "flip_delay_axis":
                    fda = str(v)
                    if fda.lower() in ["true", "1"]:
                        flip_delay_axis = True
                    elif fda.lower() in ["false", "0"]:
                        flip_delay_axis = False
                    else:
                        raise ValueError(
                            f"Invalid value for flip_delay_axis in config: {flip_delay_axis}.",
                        )
                    log_str += f"\n   Flip delay axis: {flip_delay_axis}"
                elif k == "columns":
                    for column_name, column_dict in offsets["columns"].items():
                        columns.append(column_name)
                        weight = column_dict.get("weight", 1)
                        if not isinstance(weight, (int, float, np.integer, np.floating)):
                            raise TypeError(
                                f"Invalid type for weight of column {column_name}: {type(weight)}",
                            )
                        weights.append(weight)
                        pm = column_dict.get("preserve_mean", False)
                        preserve_mean.append(pm)
                        red = column_dict.get("reduction", None)
                        reductions.append(red)
                        log_str += (
                            f"\n   Column[{column_name}]: Weight={weight}, Preserve Mean: {pm}, "
                            f"Reductions: {red}."
                        )

            if not suppress_output:
                logger.info(log_str)

            if len(columns) > 0:
                df = dfops.offset_by_other_columns(
                    df=df,
                    target_column=delay_column,
                    offset_columns=columns,
                    weights=weights,
                    preserve_mean=preserve_mean,
                    reductions=reductions,
                )

            if constant:
                df[delay_column] = df.map_partitions(
                    lambda x: x[delay_column] + constant,
                    meta=(delay_column, np.float64),
                )

            if flip_delay_axis:
                df[delay_column] = -df[delay_column]

            self.offsets = offsets
            metadata["offsets"] = offsets

        return df, metadata


def extract_delay_stage_parameters(
    file: str,
    p1_key: str,
    p2_key: str,
    t0_key: str,
) -> tuple:
    """
    Read delay stage ranges from hdf5 file

    Parameters:
        file (str): filename
        p1_key (str): hdf5 path to the start of the scan range
        p2_key (str): hdf5 path to the end of the scan range
        t0_key (str): hdf5 path to the t0 value

    Returns:
        tuple: (p1_value, p2_value, t0_value)
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
    delay_mm: float | np.ndarray,
    time0_mm: float,
) -> float | np.ndarray:
    """Converts a delay stage position in mm into a relative delay in picoseconds
    (double pass).

    Args:
        delay_mm (float | np.ndarray): Delay stage position in mm
        time0_mm (float): Delay stage position of pump-probe overlap in mm

    Returns:
        float | np.ndarray: Relative delay in picoseconds
    """
    delay_ps = (delay_mm - time0_mm) / 0.15
    return delay_ps
