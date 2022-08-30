import copy
import gc
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


class EnergyCalibrator:
    def __init__(
        self,
        df: pd.DataFrame = None,
        tof_axis_name: str = "dldTime",
        tof_axis_units: str = "step",
        sector_correction=True,
        jitter=True,
        functional_shift=False,
        parameters: Union[dict, str, Path] = {},
        **kwargs,
    ):
        self._df = df
        self.tof_axis_name = tof_axis_name
        self.tof_axis_units = tof_axis_units
        self._parameters = self.load_parameters(parameters, **kwargs)
        self._cached_parameters = {}

    def load_parameters(self, pars, **kwargs):
        """Load parameters from file, dictionary or explicit arguments

        Args:
            pars: dictionary or file containing the required parameters
            kwargs: any other argument passed is added as a parameter

        Raises:
            AttributeError: _description_

        Returns:
            the merged dictionary containing all parameters
        """
        if isinstance(pars, (str, Path)):
            pars = self._parameters_from_file(pars)
        elif not isinstance(pars, dict):
            raise AttributeError(
                "paramters must be a dictionary or a json/yaml file.",
            )
        return {**pars, **kwargs}

    def _init_calibration_dataframe(self):
        self._calib_df = copy.deepcopy(self._df[self.tof_axis_name])
        if self.tof_axis_units == "step":
            self._calib_df["tof_step"] = self._calib_df[self.tof_axis_name]
            self._calib_df["tof_ns"] = self._calib_df["tof_step"]

    @property
    def parameters(self):
        return {**self._parameters, **self._cached_parameters}

    def clear_cache(self):
        """clear the cached datasets and collect garbage"""
        del self._cached_parameters
        gc.collect()
        self._cached_parameters = {}

    def apply_sector_correction(self):
        raise NotImplementedError

    def apply_jitter(self):
        raise NotImplementedError

    def apply_functional_shift(self):
        raise NotImplementedError

    @staticmethod
    def tof_to_energy(
        tof_ns: Union[np.array, pd.Series],
        time_offset: float,
        tof_length: float,
        energy_offset: float = 0.0,
        tof_axis_name: str = "tof",
    ) -> pd.DataFrame:
        """Convert time of flight to energy.

        The energy axis generated is not yet neither kinetic nor binding energy. Its
        only linear with energy

        Args:
            df: _description_
            time_offset: _description_
            energy_offset: _description_
            tof_length: _description_
            tof_axis_name: _description_. Defaults to 'tof'.

        Returns:
            the values of tof_ns converted to energy
        """
        k = 0.5 * 1e18 * 9.10938e-31 / 1.602177e-19
        return (
            k * np.power(tof_length / (tof_ns - time_offset), 2.0)
            - energy_offset
        )

    def kinetic_energy(self, **kwargs):
        pars = {**self.parameters, **kwargs}

        return self._new_df.map_partitions(
            self.tof_to_energy,
            pars["tof_axis_name"],
            pars["time_offset"],
            pars["energy_offset"],
            pars["tof_path_length"],
        )
