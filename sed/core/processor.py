"""This module contains the core class for the sed package

"""
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import dask.dataframe as ddf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import xarray as xr

from sed.binning import bin_dataframe
from sed.calibrator.delay import DelayCalibrator
from sed.calibrator.energy import EnergyCalibrator
from sed.calibrator.momentum import MomentumCorrector
from sed.config.settings import parse_config
from sed.core.dfops import apply_jitter
from sed.core.metadata import MetaHandler
from sed.core.workflow_recorder import CallTracker
from sed.diagnostics import grid_histogram
from sed.loader.loader_interface import get_loader
from sed.loader.mirrorutil import CopyTool

N_CPU = psutil.cpu_count()


class SedProcessor:
    """[summary]"""

    def __init__(
        self,
        metadata: dict = None,
        config: Union[dict, str] = None,
        dataframe: Union[pd.DataFrame, ddf.DataFrame] = None,
        files: List[str] = None,
        folder: str = None,
        **kwds,
    ):

        self._config = parse_config(config)
        num_cores = self._config.get("binning", {}).get("num_cores", N_CPU - 1)
        if num_cores >= N_CPU:
            num_cores = N_CPU - 1
        self._config["binning"]["num_cores"] = num_cores

        self._dataframe: Union[pd.DataFrame, ddf.DataFrame] = None
        self._files: List[str] = []

        self._binned: xr.DataArray = None
        self._pre_binned: xr.DataArray = None

        self._dimensions: List[str] = []
        self._coordinates: Dict[Any, Any] = {}
        self.axis: Dict[Any, Any] = {}
        self._attributes = MetaHandler(meta=metadata)

        loader_name = self._config["core"]["loader"]
        self.loader = get_loader(
            loader_name=loader_name,
            config=self._config,
        )

        self.ec = EnergyCalibrator(
            loader=self.loader,
            config=self._config,
            tracker=self._call_tracker,
        )

        self.mc = MomentumCorrector(
            config=self._config,
            tracker=self._call_tracker,
        )

        self.dc = DelayCalibrator(
            config=self._config,
            tracker=self._call_tracker,
        )

        self.use_copy_tool = self._config.get("core", {}).get(
            "use_copy_tool",
            False,
        )
        if self.use_copy_tool:
            try:
                self.ct = CopyTool(
                    source=self._config["core"]["copy_tool_source"],
                    dest=self._config["core"]["copy_tool_dest"],
                    **self._config["core"].get("copy_tool_kwds", {}),
                )
            except KeyError:
                self.use_copy_tool = False

        # Load data if provided:
        if dataframe is not None or files is not None or folder is not None:
            self.load(dataframe=dataframe, files=files, folder=folder, **kwds)

    def __repr__(self):
        if self._dataframe is None:
            df_str = "Data Frame: No Data loaded"
        else:
            df_str = self._dataframe.__repr__()
        coordinates_str = f"Coordinates: {self._coordinates}"
        dimensions_str = f"Dimensions: {self._dimensions}"
        pretty_str = df_str + "\n" + coordinates_str + "\n" + dimensions_str
        return pretty_str

    def __getitem__(self, val: Any) -> pd.DataFrame:
        """Accessor to the underlying data structure.

        Args:
            val: [description]

        Raises:
            ValueError: [description]

        Returns:
            [description]
        """
        return self._dataframe[val]

    @property
    def config(self) -> Dict[Any, Any]:
        """Getter attribute for the config dictionary

        Returns:
            Dict: The config dictionary.
        """
        return self._config

    @config.setter
    def config(self, config: Union[dict, str]):
        """Setter function for the config dictionary.

        Args:
            config (Union[dict, str]): Config dictionary or path of config file
            to load.
        """
        self._config = parse_config(config)
        num_cores = self._config.get("binning", {}).get("num_cores", N_CPU - 1)
        if num_cores >= N_CPU:
            num_cores = N_CPU - 1
        self._config["binning"]["num_cores"] = num_cores

    @property
    def call_tracker(self) -> List[MethodCall]:
        """List of tracked function calls."""

        return self._call_tracker

    @property
    def dimensions(self) -> list:
        """Getter attribute for the dimensions.

        Returns:
            list: List of dimensions.
        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims: list):
        """Setter function for the dimensions.

        Args:
            dims (list): List of dimensions to set.
        """
        assert isinstance(dims, list)
        self._dimensions = dims

    @property
    def coordinates(self) -> dict:
        """Getter attribute for the coordinates dict.

        Returns:
            dict: Dictionary of coordinates.
        """
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords: dict):
        """Setter function for the coordinates dict

        Args:
            coords (dict): Dictionary of coordinates.
        """
        assert isinstance(coords, dict)
        self._coordinates = {}
        for k, v in coords.items():
            self._coordinates[k] = xr.DataArray(v)

    def cpy(self, path: Union[str, List[str]]) -> Union[str, List[str]]:
        """Returns either the original or the copied path to the given path.

        Args:
            path (Union[str, List[str]]): Source path or path list

        Returns:
            Union[str, List[str]]: Source or destination path or path list.
        """
        if self.use_copy_tool:
            if isinstance(path, list):
                path_out = []
                for file in path:
                    path_out.append(self.ct.copy(file))
                return path_out

            return self.ct.copy(path)

        if isinstance(path, list):
            return path

        return path

    @CallTracker
    def load(
        self,
        dataframe: Union[pd.DataFrame, ddf.DataFrame] = None,
        files: List[str] = None,
        folder: str = None,
        **kwds,
    ):
        """Load tabular data of Single Events

        Args:
            dataframe: data in tabular format. Accepts anything which
                can be interpreted by pd.DataFrame as an input
        """
        if dataframe is not None:
            self._dataframe = dataframe
        elif folder is not None:
            # pylint: disable=unused-variable
            dataframe, metadata = self.loader.read_dataframe(
                folder=cast(str, self.cpy(folder)),
                **kwds,
            )
            self._dataframe = dataframe
            # TODO: Implement metadata treatment
            # self._attributes.add(metadata)
            self._files = self.loader.files
        elif files is not None:
            # pylint: disable=unused-variable
            dataframe, metadata = self.loader.read_dataframe(
                files=cast(List[str], self.cpy(files)),
                **kwds,
            )
            self._dataframe = dataframe
            # TODO: Implement metadata treatment
            # self._attributes.add(metadata)
            self._files = self.loader.files
        else:
            raise ValueError(
                "Either 'dataframe', 'files' or 'folder' needs to be privided!",
            )

    # Momentum calibration workflow
    # 1. Bin raw detector data for distortion correction
    def bin_and_load_momentum_calibration(
        self,
        df_partitions: int = 100,
        rotation_symmetry: int = 6,
        axes: List[str] = None,
        bins: List[int] = None,
        ranges: Sequence[Tuple[float, float]] = None,
        plane: int = 0,
        width: int = 5,
        apply: bool = False,
        **kwds,
    ):
        """Function to do an initial binning of the dataframe loaded to the class,
        slice a plane from it using an interactive view, and load it into the
        momentum corrector class.

        Args:
            df_partitions (int, optional):
                Number of dataframe partitions to use for the initial binning.
                Defaults to 100.
            rotation_symmetry (int, optional):
                Number of rotational symmetry axes. Defaults to 6.
            axes (List[str], optional):
                Axes to bin. Defaults to _config["momentum"]["axes"].
            bins (List[int], optional):
                Bin numbers to use for binning.
                Defaults to _config["momentum"]["bins"].
            ranges (List[Tuple], optional):
                Ranges to use for binning. Defaults to _config["momentum"]["ranges"].
            plane (int, optional):
                Initial value for the plane slider. Defaults to 0.
            width (int, optional):
                Initial value for the width slider. Defaults to 5.
            apply (bool, optional):
                Option to directly apply the values and select the slice.
                Defaults to False.
            **kwds:
                Keyword argument passed to the pre_binning function.
        """

        self._pre_binned = self.pre_binning(
            df_partitions=df_partitions,
            axes=axes,
            bins=bins,
            ranges=ranges,
            **kwds,
        )

        self.mc.load_data(data=self._pre_binned, rotsym=rotation_symmetry)
        self.mc.select_slicer(plane=plane, width=width, apply=apply)

    # 2. Generate the spline warp correction from momentum features.
    # Either autoselect features, or input features from view above.
    def generate_splinewarp(
        self,
        features: np.ndarray = None,
        auto_detect: bool = False,
        include_center: bool = True,
        **kwds,
    ):
        """Detects or assigns the provided feature points, and generates the
        splinewarp correction.

        Args:
            features (np.ndarray, optional):
                np.ndarray of features. Defaults to None.
            auto_detect (bool, optional):
                Whether to auto-detect the features. Defaults to False.
            include_center (bool, optional):
                Option to fix the position of the center point for the correction.
                Defaults to True.
        """
        if auto_detect:  # automatic feature selection
            sigma = kwds.pop(
                "sigma",
                self._config.get("momentum", {}).get("sigma", 5),
            )
            fwhm = kwds.pop(
                "fwhm",
                self._config.get("momentum", {}).get("fwhm", 8),
            )
            sigma_radius = kwds.pop(
                "sigma_radius",
                self._config.get("momentum", {}).get("sigma_radius", 1),
            )
            self.mc.feature_extract(
                sigma=sigma,
                fwhm=fwhm,
                sigma_radius=sigma_radius,
                **kwds,
            )
        else:  # Manual feature selection
            assert features is not None
            self.mc.add_features(features, **kwds)

        print("Original slice with reference features")
        self.mc.view(annotated=True, backend="bokeh", crosshair=True)

        self.mc.spline_warp_estimate(include_center=include_center, **kwds)

        print("Corrected slice with target features")
        self.mc.view(
            image=self.mc.slice_corrected,
            annotated=True,
            points={"feats": self.mc.ptargs},
            backend="bokeh",
            crosshair=True,
        )

        print("Original slice with target features")
        self.mc.view(
            image=self.mc.slice,
            points={"feats": self.mc.ptargs},
            annotated=True,
            backend="bokeh",
        )

    # 3. Pose corrections. Provide interactive interface for correcting
    # scaling, shift and rotation
    def pose_adjustment(
        self,
        scale: float = 1,
        xtrans: float = 0,
        ytrans: float = 0,
        angle: float = 0,
        apply: bool = False,
    ):
        """Interactive panel to adjust transformations that are applied to the image.
        Applies first a scaling, next a x/y translation, and last a rotation around
        the center of the image (pixel 256/256).

        Args:
            scale (float, optional):
                Initial value of the scaling slider. Defaults to 1.
            xtrans (float, optional):
                Initial value of the xtrans slider. Defaults to 0.
            ytrans (float, optional):
                Initial value of the ytrans slider. Defaults to 0.
            angle (float, optional):
                Initial value of the angle slider. Defaults to 0.
            apply (bool, optional):
                Option to directly apply the provided transformations.
                Defaults to False.
        """
        self.mc.pose_adjustment(
            scale=scale,
            xtrans=xtrans,
            ytrans=ytrans,
            angle=angle,
            apply=apply,
        )

    # 4. Calculate momentum calibration and apply correction and calibration
    # to the dataframe
    def calibrate_momentum_axes(
        self,
        point_a: Union[np.ndarray, List[int]],
        point_b: Union[np.ndarray, List[int]] = None,
        k_distance: float = None,
        k_coord_a: Union[np.ndarray, List[float]] = None,
        k_coord_b: Union[np.ndarray, List[float]] = np.array([0.0, 0.0]),
        equiscale: bool = True,
        apply=True,
    ):
        """Calibrate momentum axes and apply distortion correction and
        momentum calibration to dataframe. One can either provide pixel coordinates
        of a high-symmetry point and its distance to the BZ center, or the
        k-coordinates of two points in the BZ.

        Args:
            point_a (Union[np.ndarray, List[int]]):
                Pixel coordinates of the first point used for momentum calibration.
            point_b (Union[np.ndarray, List[int]], optional):
                Pixel coordinates of the second point used for momentum calibration.
                Defaults to the center pixel, _config["momentum"]["center_pixel".
            k_distance (float, optional):
                Momentum distance between point a and b. Needs to be provided if no
                specific k-koordinates for the two points are given. Defaults to None.
            k_coord_a (Union[np.ndarray, List[float]], optional):
                Momentum coordinate of the first point used for calibration.
                Used if equiscale is False. Defaults to None.
            k_coord_b (Union[np.ndarray, List[float]], optional):
                Momentum coordinate of the second point used for calibration.
                Defaults to [0.0, 0.0].
            equiscale (bool, optional):
                Option to apply different scales to kx and ky. If True, the distance
                between points a and b, and the absolute position of point a are used
                for defining the scale. If False, the scale is calculated from the k-
                positions of both points a and b. Defaults to True.
            apply (bool, optional):
                Option to apply the Distortion correction and momentum calibration to
                the dataframe. Defaults to True.
        """
        if point_b is None:
            point_b = self._config.get("momentum", {}).get(
                "center_pixel",
                [256, 256],
            )

        calibration = self.mc.calibrate(
            point_a=point_a,
            point_b=point_b,
            k_distance=k_distance,
            k_coord_a=k_coord_a,
            k_coord_b=k_coord_b,
            equiscale=equiscale,
        )

        self.mc.view(
            image=self.mc.slice_transformed,
            imkwds={"extent": calibration["extent"]},
        )
        plt.title("Momentum calibrated data")
        plt.xlabel("$k_x$", fontsize=15)
        plt.ylabel("$k_y$", fontsize=15)
        plt.axhline(0)
        plt.axvline(0)
        plt.show()

        if apply and self._dataframe is not None:
            print("Adding corrected X/Y columns to dataframe:")
            self._dataframe = self.mc.apply_distortion_correction(
                self._dataframe,
            )
            print("Adding kx/ky columns to dataframe:")
            self._dataframe = self.mc.append_k_axis(self._dataframe)
            print(self._dataframe.head(10))

    # Energy correction workflow
    # 1. Adjust the energy correction parameters
    def adjust_energy_correction(
        self,
        correction_type: str = None,
        amplitude: float = None,
        center: Tuple[float, float] = None,
        apply=False,
        **kwds,
    ):
        """Present an interactive plot to adjust the parameters for the TOF/energy
        correction. Also pre-bins the data if they are not present yet.

        Args:
            correction_type (str, optional):
                Type of correction to use. Possible values are:
                "sperical", "Lorentzian", "Gaussian", "Lorentzian_asymmetric".
                Defaults to _config["energy"]["correction"]["correction_type"].
            amplitude (float, optional):
                Amplitude of the correction.
                Defaults to _config["energy"]["correction"]["amplitude"].
            center (Tuple[float, float], optional):
                Center X/Y coordinates for the correction.
                Defaults to _config["energy"]["correction"]["center"].
            apply (bool, optional):
                Option to directly apply the provided or default correction
                parameters. Defaults to False.
        """
        if self._pre_binned is None:
            print(
                "Pre-binned data not present, binning using defaults from config...",
            )
            self._pre_binned = self.pre_binning()

        self.ec.adjust_energy_correction(
            self._pre_binned,
            correction_type=correction_type,
            amplitude=amplitude,
            center=center,
            apply=apply,
            **kwds,
        )

    # 2. Apply energy correction to dataframe
    def apply_energy_correction(self):
        """Apply the enery correction parameters stored in the class to the
        dataframe. Per default it is directly applied to the TOF column.
        """
        if self._dataframe is not None:
            print("Applying energy correction to dataframe...")
            self._dataframe = self.ec.apply_energy_correction(self._dataframe)

    # Energy calibrator workflow
    # 1. Load and normalize data
    def load_bias_series(
        self,
        data_files: List[str],
        axes: List[str] = None,
        bins: List = None,
        ranges: Sequence[Tuple[float, float]] = None,
        biases: np.ndarray = None,
        bias_key: str = None,
        normalize: bool = None,
        span: int = None,
        order: int = None,
    ):
        """Load and bin data from single-event files

        Parameters:
            data_files: list of file names to bin
            axes: bin axes | _config["dataframe"]["tof_column"]
            bins: number of bins | _config["energy"]["bins"]
            ranges: bin ranges | _config["energy"]["ranges"]
            biases: Bias voltages used
            bias_key: hdf5 path where bias values are stored.
                    | _config["energy"]["bias_key"]
        """

        self.ec.bin_data(
            data_files=cast(List[str], self.cpy(data_files)),
            axes=axes,
            bins=bins,
            ranges=ranges,
            biases=biases,
            bias_key=bias_key,
        )
        if (normalize is not None and normalize is True) or (
            normalize is None
            and self._config.get("energy", {}).get("normalize", True)
        ):
            if span is None:
                span = self._config.get("energy", {}).get("normalize_span", 7)
            if order is None:
                order = self._config.get("energy", {}).get(
                    "normalize_order",
                    1,
                )
            self.ec.normalize(smooth=True, span=span, order=order)
        self.ec.view(
            traces=self.ec.traces_normed,
            xaxis=self.ec.tof,
            backend="bokeh",
        )

    # 2. extract ranges and get peak positions
    def find_bias_peaks(
        self,
        ranges: Union[List[Tuple], Tuple],
        ref_id: int = 0,
        infer_others: bool = True,
        mode: str = "replace",
        radius: int = None,
        peak_window: int = None,
    ):
        """find a peak within a given range for the indicated reference trace,
        and tries to find the same peak for all other traces. Uses fast_dtw to
        align curves, which might not be too good if the shape of curves changes
        qualitatively. Ideally, choose a reference trace in the middle of the set,
        and don't choose the range too narrow around the peak.
        Alternatively, a list of ranges for all traces can be given.

        Args:
            ranges (Union[List[Tuple], Tuple]):
                Tuple of TOF values indicating a range. Alternatively, a list of
                ranges for all traces can be given.
            refid (int, optional):
                The id of the trace the range refers to. Defaults to 0.

            infer_others (bool, optional):
                Whether to determine the range for the other traces. Defaults to True.
            mode (str, optional):
                whether to "add" or "replace" existing ranges. Defaults to "replace".
            radius (int, optional):
                Radius parameter for fast_dtw. Defaults to
                _config["energy"]["fastdtw_radius"].
            peak_window (int, optional):
                peak_window parameter for the peak detection algorthm. amount of points
                that have to have to behave monotoneously around a peak. Defaults to
                _config["energy"]["peak_window"].
        """
        if radius is None:
            radius = self._config.get("energy", {}).get("fastdtw_radius", 2)
        self.ec.add_features(
            ranges=ranges,
            ref_id=ref_id,
            infer_others=infer_others,
            mode=mode,
            radius=radius,
        )
        self.ec.view(
            traces=self.ec.traces_normed,
            segs=self.ec.featranges,
            xaxis=self.ec.tof,
            backend="bokeh",
        )
        print(self.ec.featranges)
        if peak_window is None:
            peak_window = self._config.get("energy", {}).get("peak_window", 7)
        try:
            self.ec.feature_extract(peak_window=peak_window)
            self.ec.view(
                traces=self.ec.traces_normed,
                peaks=self.ec.peaks,
                backend="bokeh",
            )
        except IndexError:
            print("Could not determine all peaks!")
            raise

    # 3. Fit the energy calibration reation, and apply it to the data frame
    def calibrate_energy_axis(
        self,
        ref_id: int,
        ref_energy: float,
        method: str = None,
        energy_scale: str = None,
        apply: bool = True,
        **kwds,
    ):
        """Calculate the calibration function for the energy axis,
        and apply it to the dataframe. Two approximations are implemented,
        a (normally 3rd order) polynomial approximation, and a d^2/(t-t0)^2
        relation.

        Args:
            ref_id (int):
                id of the trace at the bias where the reference energy is given.
            ref_energy (float):
                absolute energy of the detected feature at the bias of ref_id
            method (str, optional):
                The calibration method to use. Possible values are
                "lmfit", "lstsq", "lsqr".
                Defaults to _config["energy"]["calibration_method"]
            energy_scale (str, optional):
                which energy scale to use. Possible values are
                "kinetic" and "binding"
                Defaults to _config["energy"]["energy_scale"]
            apply (bool, optional):
                Whether to apply the calibration to the dataframe. Defaults to True.
        """
        if method is None:
            method = self._config.get("energy", {}).get(
                "calibration_method",
                "lmfit",
            )

        if energy_scale is None:
            energy_scale = self._config.get("energy", {}).get(
                "energy_scale",
                "kinetic",
            )

        self.ec.calibrate(
            ref_id=ref_id,
            ref_energy=ref_energy,
            method=method,
            energy_scale=energy_scale,
            **kwds,
        )
        print("Quality of Calibration:")
        self.ec.view(
            traces=self.ec.traces_normed,
            xaxis=self.ec.calibration["axis"],
            align=True,
            energy_scale=energy_scale,
            backend="bokeh",
        )
        print("E/TOF relationship:")
        self.ec.view(
            traces=self.ec.calibration["axis"][None, :],
            xaxis=self.ec.tof,
            backend="matplotlib",
            show_legend=False,
        )
        if energy_scale == "kinetic":
            plt.scatter(
                self.ec.peaks[:, 0],
                -(self.ec.biases - self.ec.biases[ref_id]) + ref_energy,
                s=50,
                c="k",
            )
        elif energy_scale == "binding":
            plt.scatter(
                self.ec.peaks[:, 0],
                self.ec.biases - self.ec.biases[ref_id] + ref_energy,
                s=50,
                c="k",
            )
        else:
            raise ValueError(
                'energy_scale needs to be either "binding" or "kinetic"',
                f", got {energy_scale}.",
            )
        plt.xlabel("Time-of-flight", fontsize=15)
        plt.ylabel("Energy (eV)", fontsize=15)
        plt.show()

        if apply and self._dataframe is not None:
            print("Adding energy column to dataframe:")
            self._dataframe = self.ec.append_energy_axis(self._dataframe)
            print(self._dataframe.head(10))

    # Delay calibration function
    def calibrate_delay_axis(
        self,
        delay_range: Tuple[float, float] = None,
        datafile: str = None,
        **kwds,
    ):
        """Append delay column to dataframe. Either provide delay ranges, or read
        them from a file.

        Args:
            delay_range (Tuple[float, float], optional):
                The scanned delay range in picoseconds. Defaults to None.
            datafile (str, optional):
                The file from which to read the delay ranges. Defaults to None.
            **kwds:
                Keyword args passed to DelayCalibrator.append_delay_axis.
        """
        if self._dataframe is not None:
            print("Adding delay column to dataframe:")

            if delay_range is not None:
                self._dataframe = self.dc.append_delay_axis(
                    self._dataframe,
                    delay_range=delay_range,
                    **kwds,
                )
            else:
                if datafile is None:
                    try:
                        datafile = self._files[0]
                    except IndexError:
                        print(
                            "No datafile available, specify eihter",
                            " 'datafile' or 'delay_range'",
                        )
                        raise

                self._dataframe = self.dc.append_delay_axis(
                    self._dataframe,
                    datafile=datafile,
                    **kwds,
                )

            print(self._dataframe.head(10))

    @track_call
    def add_jitter(self, cols: Sequence[str] = None) -> None:
        """Add jitter to the selected dataframe columns.


        Args:
            cols: the colums onto which to apply jitter. If omitted,
            the comlums are taken from the config.

        Returns:
            None
        """
        if cols is None:
            cols = self._config.get("dataframe", {}).get(
                "jitter_cols",
                self._dataframe.columns,
            )  # jitter all columns

        self._dataframe = self._dataframe.map_partitions(
            apply_jitter,
            cols=cols,
            cols_jittered=cols,
        )

    def pre_binning(
        self,
        df_partitions: int = 100,
        axes: List[str] = None,
        bins: List[int] = None,
        ranges: Sequence[Tuple[float, float]] = None,
        **kwds,
    ) -> xr.DataArray:
        """Function to do an initial binning of the dataframe loaded to the class.

        Args:
            df_partitions (int, optional):
                Number of dataframe partitions to use for the initial binning.
                Defaults to 100.
            axes (List[str], optional):
                Axes to bin. Defaults to _config["momentum"]["axes"].
            bins (List[int], optional):
                Bin numbers to use for binning.
                Defaults to _config["momentum"]["bins"].
            ranges (List[Tuple], optional):
                Ranges to use for binning. Defaults to _config["momentum"]["ranges"].
            **kwds:
                Keyword argument passed to the compute function.
        """
        if axes is None:
            axes = self._config.get("momentum", {}).get(
                "axes",
                ["@x_column, @y_column, @tof_column"],
            )
        for loc, axis in enumerate(axes):
            if axis.startswith("@"):
                axes[loc] = self._config.get("dataframe").get(axis.strip("@"))

        if bins is None:
            bins = self._config.get("momentum", {}).get(
                "bins",
                [512, 512, 300],
            )
        if ranges is None:
            ranges_ = self._config.get("momentum", {}).get(
                "ranges",
                [[-256, 1792], [-256, 1792], [128000, 138000]],
            )
            ranges = [cast(Tuple[float, float], tuple(v)) for v in ranges_]

        assert (
            self._dataframe is not None
        ), "dataframe needs to be loaded first!"

        return self.compute(
            bins=bins,
            axes=axes,
            ranges=ranges,
            df_partitions=df_partitions,
            **kwds,
        )

    @track_call
    def compute(
        self,
        bins: Union[
            int,
            dict,
            tuple,
            List[int],
            List[np.ndarray],
            List[tuple],
        ] = 100,
        axes: Union[str, Sequence[str]] = None,
        ranges: Sequence[Tuple[float, float]] = None,
        **kwds,
    ) -> xr.DataArray:
        """Compute the histogram along the given dimensions.

        Args:
            bins: Definition of the bins. Can  be any of the following cases:
                - an integer describing the number of bins in on all dimensions
                - a tuple of 3 numbers describing start, end and step of the binning
                  range
                - a np.arrays defining the binning edges
                - a list (NOT a tuple) of any of the above (int, tuple or np.ndarray)
                - a dictionary made of the axes as keys and any of the above as values.
                This takes priority over the axes and range arguments.
            axes: The names of the axes (columns) on which to calculate the histogram.
                The order will be the order of the dimensions in the resulting array.
            ranges: list of tuples containing the start and end point of the binning
                    range.
            kwds: Keywords argument passed to bin_dataframe.

        Raises:
            AssertError: Rises when no dataframe has been loaded.

        Returns:
            The result of the n-dimensional binning represented in an
                xarray object, combining the data with the axes.
        """

        assert (
            self._dataframe is not None
        ), "dataframe needs to be loaded first!"

        hist_mode = kwds.pop("hist_mode", self._config["binning"]["hist_mode"])
        mode = kwds.pop("mode", self._config["binning"]["mode"])
        pbar = kwds.pop("pbar", self._config["binning"]["pbar"])
        num_cores = kwds.pop("num_cores", self._config["binning"]["num_cores"])
        threads_per_worker = kwds.pop(
            "threads_per_worker",
            self._config["binning"]["threads_per_worker"],
        )
        threadpool_api = kwds.pop(
            "threadpool_API",
            self._config["binning"]["threadpool_API"],
        )
        df_partitions = kwds.pop("df_partitions", None)
        if df_partitions is not None:
            dataframe = self._dataframe.partitions[
                0 : min(df_partitions, self._dataframe.npartitions)
            ]
        else:
            dataframe = self._dataframe

        self._binned = bin_dataframe(
            df=dataframe,
            bins=bins,
            axes=axes,
            ranges=ranges,
            hist_mode=hist_mode,
            mode=mode,
            pbar=pbar,
            n_cores=num_cores,
            threads_per_worker=threads_per_worker,
            threadpool_api=threadpool_api,
            **kwds,
        )
        return self._binned

    def view_event_histogram(
        self,
        dfpid: int,
        ncol: int = 2,
        bins: Sequence[int] = None,
        axes: Sequence[str] = None,
        ranges: Sequence[Tuple[float, float]] = None,
        backend: str = "bokeh",
        legend: bool = True,
        histkwds: dict = None,
        legkwds: dict = None,
        **kwds: Any,
    ):
        """
        Plot individual histograms of specified dimensions (axes) from a substituent
        dataframe partition.

        Args:
            dfpid: Number of the data frame partition to look at.
            ncol: Number of columns in the plot grid.
            bins: Number of bins to use for the speicified axes.
            axes: Name of the axes to display.
            ranges: Value ranges of all specified axes.
            jittered: Option to use the jittered dataframe.
            backend: Backend of the plotting library ('matplotlib' or 'bokeh').
            legend: Option to include a legend in the histogram plots.
            histkwds, legkwds, **kwds: Extra keyword arguments passed to
            ``sed.diagnostics.grid_histogram()``.

        Raises:
            AssertError if Jittering is requested, but the jittered dataframe
            has not been created.
            TypeError: Raises when the input values are not of the correct type.
        """
        if bins is None:
            bins = self._config["histogram"]["bins"]
        if axes is None:
            axes = self._config["histogram"]["axes"]
        if ranges is None:
            ranges = self._config["histogram"]["ranges"]

        input_types = map(type, [axes, bins, ranges])
        allowed_types = [list, tuple]

        df = self._dataframe

        if not set(input_types).issubset(allowed_types):
            raise TypeError(
                "Inputs of axes, bins, ranges need to be list or tuple!",
            )

        # Read out the values for the specified groups
        group_dict = {}
        dfpart = df.get_partition(dfpid)
        cols = dfpart.columns
        for ax in axes:
            group_dict[ax] = dfpart.values[:, cols.get_loc(ax)].compute()

        # Plot multiple histograms in a grid
        grid_histogram(
            group_dict,
            ncol=ncol,
            rvs=axes,
            rvbins=bins,
            rvranges=ranges,
            backend=backend,
            legend=legend,
            histkwds=histkwds,
            legkwds=legkwds,
            **kwds,
        )

    def add_dimension(self, name: str, axis_range: Tuple):
        """Add a dimension axis.

        Args:
            name (str): name of the axis
            axis_range (Tuple): range for the axis.

        Raises:
            ValueError: Raised if an axis with that name already exists.
        """
        if name in self._coordinates:
            raise ValueError(f"Axis {name} already exists")

        self.axis[name] = self.make_axis(axis_range)

    def make_axis(self, axis_range: Tuple) -> np.ndarray:
        """Function to make an axis.

        Args:
            axis_range (Tuple): range for the new axis.
        """

        # TODO: What shall this function do?
        return np.arange(*axis_range)
