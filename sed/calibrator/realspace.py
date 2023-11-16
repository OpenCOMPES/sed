""" sed.calibrator.realspace module. code for calibration of real space runs."""
from typing import Any
from typing import Dict
from typing import Tuple

import dask


class RealSpaceCalibrator:
    """Class for calibrating real space runs."""

    def __init__(
        self,
        config: dict = None,
    ) -> None:
        """constructor of the RealSpaceCalbirator class"""
        if config is None:
            config = {}
        self._config: dict = config
        self.x_column: str = self._config["dataframe"]["x_column"]
        self.y_column: str = self._config["dataframe"]["y_column"]
        self.x_position_column: str = self._config["dataframe"].get("x_position_column", None)
        self.y_position_column: str = self._config["dataframe"].get("y_position_column", None)
        self.center: Tuple[float, float] = self._config["dataframe"].get("center", (0.0, 0.0))

    def append_position_axis(
        self,
        df: dask.dataframe.DataFrame,
        px_to_um: float = None,
        center: Tuple[float, float] = None,
    ) -> Tuple[dask.dataframe.DataFrame, dict]:
        """append position axis to the dataframe"""
        if px_to_um is None:
            px_to_um = self._config["dataframe"].get("px_to_um", None)
        if px_to_um is None:
            raise ValueError("px_to_um is not defined")
        if center is None:
            center = self.center

        def fn(df) -> Any:
            df[self.x_position_column] = df[self.x_column] - self.center[0] * px_to_um
            df[self.y_position_column] = df[self.y_column] - self.center[1] * px_to_um
            return df

        df = df.map_partitions(fn)
        metadata: Dict[str, Any] = {"px_to_um": px_to_um, "center": self.center, "applied": True}
        return df, metadata
