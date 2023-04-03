"""This module contains the core class for the sed package

"""
import pathlib
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
from sed.binning.utils import bin_centers_to_bin_edges
from sed.calibrator import DelayCalibrator
from sed.calibrator import EnergyCalibrator
from sed.calibrator import MomentumCorrector
from sed.config import parse_config
from sed.core.dfops import apply_jitter
from sed.core.metadata import MetaHandler
from sed.diagnostics import grid_histogram
from sed.io import to_h5
from sed.io import to_nexus
from sed.io import to_tiff
from sed.loader import CopyTool
from sed.loader import get_loader

N_CPU = psutil.cpu_count()


class SedProcessor:
    """Processor class of sed. Contains wrapper functions defining a work flow for data
    correction, calibration and binning.

    Args:
        metadata (dict, optional): Dict of external Metadata. Defaults to None.
        config (Union[dict, str], optional): Config dictionary or config file name.
            Defaults to None.
        dataframe (Union[pd.DataFrame, ddf.DataFrame], optional): dataframe to load
            into the class. Defaults to None.
        files (List[str], optional): List of files to pass to the loader defined in
            the config. Defaults to None.
        folder (str, optional): Folder containing files to pass to the loader
            defined in the config. Defaults to None.
        collect_metadata (bool): Option to collect metadata from files.
            Defaults to False.
        **kwds: Keyword arguments passed to the reader.
    """

    def __init__(
        self,
        metadata: dict = None,
        config: Union[dict, str] = None,
        dataframe: Union[pd.DataFrame, ddf.DataFrame] = None,
        files: List[str] = None,
        folder: str = None,
        collect_metadata: bool = False,
        **kwds,
    ):
        """Processor class of sed. Contains wrapper functions defining a work flow
        for data correction, calibration and binning.

        Args:
            metadata (dict, optional): Dict of external Metadata. Defaults to None.
            config (Union[dict, str], optional): Config dictionary or config file name.
                Defaults to None.
            dataframe (Union[pd.DataFrame, ddf.DataFrame], optional): dataframe to load
                into the class. Defaults to None.
            files (List[str], optional): List of files to pass to the loader defined in
                the config. Defaults to None.
            folder (str, optional): Folder containing files to pass to the loader
                defined in the config. Defaults to None.
            collect_metadata (bool): Option to collect metadata from files.
                Defaults to False.
            **kwds: Keyword arguments passed to the reader.
        """
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
        )

        self.mc = MomentumCorrector(
            config=self._config,
        )

        self.dc = DelayCalibrator(
            config=self._config,
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
            self.load(
                dataframe=dataframe,
                metadata=metadata,
                files=files,
                folder=folder,
                collect_metadata=collect_metadata,
                **kwds,
            )

    def __repr__(self):
        if self._dataframe is None:
            df_str = "Data Frame: No Data loaded"
        else:
            df_str = self._dataframe.__repr__()
        coordinates_str = f"Coordinates: {self._coordinates}"
        dimensions_str = f"Dimensions: {self._dimensions}"
        pretty_str = df_str + "\n" + coordinates_str + "\n" + dimensions_str
        return pretty_str

    def __getitem__(self, val: str) -> pd.DataFrame:
        """Accessor to the underlying data structure.

        Args:
            val (str): Name of the dataframe column to retrieve.

        Returns:
            pd.DataFrame: Selected dataframe column.
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
        """Function to mirror a list of files or a folder from a network drive to a
        local storage. Returns either the original or the copied path to the given
        path. The option to use this functionality is set by
        config["core"]["use_copy_tool"].

        Args:
            path (Union[str, List[str]]): Source path or path list.

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

    def load(
        self,
        dataframe: Union[pd.DataFrame, ddf.DataFrame] = None,
        metadata: dict = None,
        files: List[str] = None,
        folder: str = None,
        collect_metadata: bool = False,
        **kwds,
    ):
        """Load tabular data of single events into the dataframe object in the class.

        Args:
            dataframe (Union[pd.DataFrame, ddf.DataFrame], optional): data in tabular
                format. Accepts anything which can be interpreted by pd.DataFrame as
                an input. Defaults to None.
            metadata (dict, optional): Dict of external Metadata. Defaults to None.
            files (List[str], optional): List of file paths to pass to the loader.
                Defaults to None.
            folder (str, optional): Folder path to pass to the loader.
                Defaults to None.
            collect_metadata (bool): Option to collect metadata from files.
                Defaults to False.

        Raises:
            ValueError: Raised if no valid input is provided.
        """
        if metadata is None:
            metadata = {}
        if dataframe is not None:
            self._dataframe = dataframe
        elif folder is not None:
            dataframe, metadata = self.loader.read_dataframe(
                folder=cast(str, self.cpy(folder)),
                metadata=metadata,
                collect_metadata=collect_metadata,
                **kwds,
            )
            self._dataframe = dataframe
            self._files = self.loader.files
        elif files is not None:
            dataframe, metadata = self.loader.read_dataframe(
                files=cast(List[str], self.cpy(files)),
                metadata=metadata,
                collect_metadata=collect_metadata,
                **kwds,
            )
            self._dataframe = dataframe
            self._files = self.loader.files
        else:
            raise ValueError(
                "Either 'dataframe', 'files' or 'folder' needs to be privided!",
            )

        for key in metadata:
            self._attributes.add(
                entry=metadata[key],
                name=key,
                duplicate_policy="merge",
            )

    # Momentum calibration workflow
    # 1. Bin raw detector data for distortion correction
    def bin_and_load_momentum_calibration(
        self,
        df_partitions: int = 100,
        axes: List[str] = None,
        bins: List[int] = None,
        ranges: Sequence[Tuple[float, float]] = None,
        plane: int = 0,
        width: int = 5,
        apply: bool = False,
        **kwds,
    ):
        """1st step of momentum correction work flow. Function to do an initial binning
        of the dataframe loaded to the class, slice a plane from it using an
        interactive view, and load it into the momentum corrector class.

        Args:
            df_partitions (int, optional): Number of dataframe partitions to use for
                the initial binning. Defaults to 100.
            axes (List[str], optional): Axes to bin.
                Defaults to config["momentum"]["axes"].
            bins (List[int], optional): Bin numbers to use for binning.
                Defaults to config["momentum"]["bins"].
            ranges (List[Tuple], optional): Ranges to use for binning.
                Defaults to config["momentum"]["ranges"].
            plane (int, optional): Initial value for the plane slider. Defaults to 0.
            width (int, optional): Initial value for the width slider. Defaults to 5.
            apply (bool, optional): Option to directly apply the values and select the
                slice. Defaults to False.
            **kwds: Keyword argument passed to the pre_binning function.
        """
        self._pre_binned = self.pre_binning(
            df_partitions=df_partitions,
            axes=axes,
            bins=bins,
            ranges=ranges,
            **kwds,
        )

        self.mc.load_data(data=self._pre_binned)
        self.mc.select_slicer(plane=plane, width=width, apply=apply)

    # 2. Generate the spline warp correction from momentum features.
    # Either autoselect features, or input features from view above.
    def generate_splinewarp(
        self,
        features: np.ndarray = None,
        rotation_symmetry: int = 6,
        auto_detect: bool = False,
        include_center: bool = True,
        **kwds,
    ):
        """2. Step of the distortion correction workflow: Detect feature points in
        momentum space, or assign the provided feature points, and generate a
        correction function restoring the symmetry in the image using a splinewarp
        algortihm.

        Args:
            features (np.ndarray, optional): np.ndarray of features. Defaults to None.
            rotation_symmetry (int, optional): Number of rotational symmetry axes.
                Defaults to 6.
            auto_detect (bool, optional): Whether to auto-detect the features.
                Defaults to False.
            include_center (bool, optional): Option to fix the position of the center
                point for the correction. Defaults to True.
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
                rotsym=rotation_symmetry,
                **kwds,
            )
        else:  # Manual feature selection
            self.mc.add_features(
                features=features,
                rotsym=rotation_symmetry,
                **kwds,
            )

        self.mc.spline_warp_estimate(include_center=include_center, **kwds)

        if self.mc.slice is not None:
            print("Original slice with reference features")
            self.mc.view(annotated=True, backend="bokeh", crosshair=True)

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
        use_correction: bool = True,
    ):
        """3. step of the distortion correction workflow: Generate an interactive panel
        to adjust affine transformations that are applied to the image. Applies first
        a scaling, next an x/y translation, and last a rotation around the center of
        the image.

        Args:
            scale (float, optional): Initial value of the scaling slider.
                Defaults to 1.
            xtrans (float, optional): Initial value of the xtrans slider.
                Defaults to 0.
            ytrans (float, optional): Initial value of the ytrans slider.
                Defaults to 0.
            angle (float, optional): Initial value of the angle slider.
                Defaults to 0.
            apply (bool, optional): Option to directly apply the provided
                transformations. Defaults to False.
            use_correction (bool, option): Whether to use the spline warp correction
                or not. Defaults to True.
        """
        # Generate homomorphy as default if no distortion correction has been applied
        if self.mc.slice_corrected is None:
            if self.mc.slice is None:
                raise ValueError(
                    "No slice for corrections and transformations loaded!",
                )
            self.mc.slice_corrected = self.mc.slice

        if self.mc.cdeform_field is None or self.mc.rdeform_field is None:
            # Generate default distortion correction
            self.mc.add_features()
            self.mc.spline_warp_estimate()

        if not use_correction:
            self.mc.reset_deformation()

        self.mc.pose_adjustment(
            scale=scale,
            xtrans=xtrans,
            ytrans=ytrans,
            angle=angle,
            apply=apply,
        )

    def apply_momentum_correction(
        self,
        preview: bool = False,
    ):
        """Applies the distortion correction and pose adjustment (optional)
        to the dataframe.

        Args:
            rdeform_field (np.ndarray, optional): Row deformation field.
                Defaults to None.
            cdeform_field (np.ndarray, optional): Column deformation field.
                Defaults to None.
            inv_dfield (np.ndarray, optional): Inverse deformation field.
                Defaults to None.
            preview (bool): Option to preview the first elements of the data frame.
        """
        if self._dataframe is not None:
            print("Adding corrected X/Y columns to dataframe:")
            self._dataframe, metadata = self.mc.apply_corrections(
                df=self._dataframe,
            )
            # Add Metadata
            self._attributes.add(
                metadata,
                "momentum_correction",
                duplicate_policy="merge",
            )
            if preview:
                print(self._dataframe.head(10))
            else:
                print(self._dataframe)

    # 4. Calculate momentum calibration and apply correction and calibration
    # to the dataframe
    def calibrate_momentum_axes(
        self,
        point_a: Union[np.ndarray, List[int]] = None,
        point_b: Union[np.ndarray, List[int]] = None,
        k_distance: float = None,
        k_coord_a: Union[np.ndarray, List[float]] = None,
        k_coord_b: Union[np.ndarray, List[float]] = np.array([0.0, 0.0]),
        equiscale: bool = True,
        apply=False,
    ):
        """4. step of the momentum correction/calibration workflow. Calibrate momentum
        axes using either provided pixel coordinates of a high-symmetry point and its
        distance to the BZ center, or the k-coordinates of two points in the BZ
        (depending on the equiscale option). Opens an interactive panel for selecting
        the points.

        Args:
            point_a (Union[np.ndarray, List[int]]): Pixel coordinates of the first
                point used for momentum calibration.
            point_b (Union[np.ndarray, List[int]], optional): Pixel coordinates of the
                second point used for momentum calibration.
                Defaults to config["momentum"]["center_pixel"].
            k_distance (float, optional): Momentum distance between point a and b.
                Needs to be provided if no specific k-koordinates for the two points
                are given. Defaults to None.
            k_coord_a (Union[np.ndarray, List[float]], optional): Momentum coordinate
                of the first point used for calibration. Used if equiscale is False.
                Defaults to None.
            k_coord_b (Union[np.ndarray, List[float]], optional): Momentum coordinate
                of the second point used for calibration. Defaults to [0.0, 0.0].
            equiscale (bool, optional): Option to apply different scales to kx and ky.
                If True, the distance between points a and b, and the absolute
                position of point a are used for defining the scale. If False, the
                scale is calculated from the k-positions of both points a and b.
                Defaults to True.
            apply (bool, optional): Option to directly store the momentum calibration
                in the class. Defaults to False.
        """
        if point_b is None:
            point_b = self._config.get("momentum", {}).get(
                "center_pixel",
                [256, 256],
            )

        self.mc.select_k_range(
            point_a=point_a,
            point_b=point_b,
            k_distance=k_distance,
            k_coord_a=k_coord_a,
            k_coord_b=k_coord_b,
            equiscale=equiscale,
            apply=apply,
        )

    # 5. Apply correction and calibration to the dataframe
    def apply_momentum_calibration(
        self,
        calibration: dict = None,
        preview: bool = False,
    ):
        """5. step of the momentum calibration/distortion correction work flow: Apply
        any distortion correction and/or pose adjustment stored in the MomentumCorrector
        class and the momentum calibration to the dataframe.

        Args:
            calibration (dict, optional): Optional dictionary with calibration data to
                use. Defaults to None.
            preview (bool): Option to preview the first elements of the data frame.
        """
        if self._dataframe is not None:

            print("Adding kx/ky columns to dataframe:")
            self._dataframe, metadata = self.mc.append_k_axis(
                df=self._dataframe,
                calibration=calibration,
            )

            # Add Metadata
            self._attributes.add(
                metadata,
                "momentum_calibration",
                duplicate_policy="merge",
            )
            if preview:
                print(self._dataframe.head(10))
            else:
                print(self._dataframe)

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
        """1. step of the energy crrection workflow: Opens an interactive plot to
        adjust the parameters for the TOF/energy correction. Also pre-bins the data if
        they are not present yet.

        Args:
            correction_type (str, optional): Type of correction to apply to the TOF
                axis. Valid values are:

                - 'spherical'
                - 'Lorentzian'
                - 'Gaussian'
                - 'Lorentzian_asymmetric'

                Defaults to config["energy"]["correction_type"].
            amplitude (float, optional): Amplitude of the correction.
                Defaults to config["energy"]["correction"]["amplitude"].
            center (Tuple[float, float], optional): Center X/Y coordinates for the
                correction. Defaults to config["energy"]["correction"]["center"].
            apply (bool, optional): Option to directly apply the provided or default
                correction parameters. Defaults to False.
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
    def apply_energy_correction(
        self,
        correction: dict = None,
        preview: bool = False,
        **kwds,
    ):
        """2. step of the energy correction workflow: Apply the enery correction
        parameters stored in the class to the dataframe.

        Args:
            correction (dict, optional): Dictionary containing the correction
                parameters. Defaults to config["energy"]["calibration"].
            preview (bool): Option to preview the first elements of the data frame.
            **kwds:
                Keyword args passed to ``EnergyCalibrator.apply_energy_correction``.
            preview (bool): Option to preview the first elements of the data frame.
            **kwds:
                Keyword args passed to ``EnergyCalibrator.apply_energy_correction``.
        """
        if self._dataframe is not None:
            print("Applying energy correction to dataframe...")
            self._dataframe, metadata = self.ec.apply_energy_correction(
                df=self._dataframe,
                correction=correction,
                **kwds,
            )

            # Add Metadata
            self._attributes.add(
                metadata,
                "energy_correction",
            )
            if preview:
                print(self._dataframe.head(10))
            else:
                print(self._dataframe)

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
        """1. step of the energy calibration workflow: Load and bin data from
        single-event files.

        Args:
            data_files (List[str]): list of file paths to bin
            axes (List[str], optional): bin axes.
                Defaults to config["dataframe"]["tof_column"].
            bins (List, optional): number of bins.
                Defaults to config["energy"]["bins"].
            ranges (Sequence[Tuple[float, float]], optional): bin ranges.
                Defaults to config["energy"]["ranges"].
            biases (np.ndarray, optional): Bias voltages used. If missing, bias
                voltages are extracted from the data files.
            bias_key (str, optional): hdf5 path where bias values are stored.
                Defaults to config["energy"]["bias_key"].
            normalize (bool, optional): Option to normalize traces.
                Defaults to config["energy"]["normalize"].
            span (int, optional): span smoothing parameters of the LOESS method
                (see ``scipy.signal.savgol_filter()``).
                Defaults to config["energy"]["normalize_span"].
            order (int, optional): order smoothing parameters of the LOESS method
                (see ``scipy.signal.savgol_filter()``).
                Defaults to config["energy"]["normalize_order"].
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
        """2. step of the energy calibration workflow: Find a peak within a given range
        for the indicated reference trace, and tries to find the same peak for all
        other traces. Uses fast_dtw to align curves, which might not be too good if the
        shape of curves changes qualitatively. Ideally, choose a reference trace in the
        middle of the set, and don't choose the range too narrow around the peak.
        Alternatively, a list of ranges for all traces can be provided.

        Args:
            ranges (Union[List[Tuple], Tuple]): Tuple of TOF values indicating a range.
                Alternatively, a list of ranges for all traces can be given.
            refid (int, optional): The id of the trace the range refers to.
                Defaults to 0.
            infer_others (bool, optional): Whether to determine the range for the other
                traces. Defaults to True.
            mode (str, optional): Whether to "add" or "replace" existing ranges.
                Defaults to "replace".
            radius (int, optional): Radius parameter for fast_dtw.
                Defaults to config["energy"]["fastdtw_radius"].
            peak_window (int, optional): Peak_window parameter for the peak detection
                algorthm. amount of points that have to have to behave monotoneously
                around a peak. Defaults to config["energy"]["peak_window"].
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

    # 3. Fit the energy calibration relation
    def calibrate_energy_axis(
        self,
        ref_id: int,
        ref_energy: float,
        method: str = None,
        energy_scale: str = None,
        **kwds,
    ):
        """3. Step of the energy calibration workflow: Calculate the calibration
        function for the energy axis, and apply it to the dataframe. Two
        approximations are implemented, a (normally 3rd order) polynomial
        approximation, and a d^2/(t-t0)^2 relation.

        Args:
            ref_id (int): id of the trace at the bias where the reference energy is
                given.
            ref_energy (float): Absolute energy of the detected feature at the bias
                of ref_id
            method (str, optional): Method for determining the energy calibration.

                - **'lmfit'**: Energy calibration using lmfit and 1/t^2 form.
                - **'lstsq'**, **'lsqr'**: Energy calibration using polynomial form.

                Defaults to config["energy"]["calibration_method"]
            energy_scale (str, optional): Direction of increasing energy scale.

                - **'kinetic'**: increasing energy with decreasing TOF.
                - **'binding'**: increasing energy with increasing TOF.

                Defaults to config["energy"]["energy_scale"]
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

    # 4. Apply energy calibration to the dataframe
    def append_energy_axis(
        self,
        calibration: dict = None,
        preview: bool = False,
        **kwds,
    ):
        """4. step of the energy calibration workflow: Apply the calibration function
        to to the dataframe. Two approximations are implemented, a (normally 3rd order)
        polynomial approximation, and a d^2/(t-t0)^2 relation. a calibration dictionary
        can be provided.

        Args:
            calibration (dict, optional): Calibration dict containing calibration
                parameters. Overrides calibration from class or config.
                Defaults to None.
            preview (bool): Option to preview the first elements of the data frame.
            **kwds:
                Keyword args passed to ``EnergyCalibrator.append_energy_axis``.
        """
        if self._dataframe is not None:
            print("Adding energy column to dataframe:")
            self._dataframe, metadata = self.ec.append_energy_axis(
                df=self._dataframe,
                calibration=calibration,
                **kwds,
            )

            # Add Metadata
            self._attributes.add(
                metadata,
                "energy_calibration",
                duplicate_policy="merge",
            )
            if preview:
                print(self._dataframe.head(10))
            else:
                print(self._dataframe)

    # Delay calibration function
    def calibrate_delay_axis(
        self,
        delay_range: Tuple[float, float] = None,
        datafile: str = None,
        preview: bool = False,
        **kwds,
    ):
        """Append delay column to dataframe. Either provide delay ranges, or read
        them from a file.

        Args:
            delay_range (Tuple[float, float], optional): The scanned delay range in
                picoseconds. Defaults to None.
            datafile (str, optional): The file from which to read the delay ranges.
                Defaults to None.
            preview (bool): Option to preview the first elements of the data frame.
            **kwds: Keyword args passed to ``DelayCalibrator.append_delay_axis``.
        """
        if self._dataframe is not None:
            print("Adding delay column to dataframe:")

            if delay_range is not None:
                self._dataframe, metadata = self.dc.append_delay_axis(
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

                self._dataframe, metadata = self.dc.append_delay_axis(
                    self._dataframe,
                    datafile=datafile,
                    **kwds,
                )

            # Add Metadata
            self._attributes.add(
                metadata,
                "delay_calibration",
                duplicate_policy="merge",
            )
            if preview:
                print(self._dataframe.head(10))
            else:
                print(self._dataframe)

    def add_jitter(self, cols: Sequence[str] = None):
        """Add jitter to the selected dataframe columns.

        Args:
            cols (Sequence[str], optional): The colums onto which to apply jitter.
                Defaults to config["dataframe"]["jitter_cols"].
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
        metadata = []
        for col in cols:
            metadata.append(col)
        self._attributes.add(metadata, "jittering", duplicate_policy="append")

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
            df_partitions (int, optional): Number of dataframe partitions to use for
                the initial binning. Defaults to 100.
            axes (List[str], optional): Axes to bin.
                Defaults to config["momentum"]["axes"].
            bins (List[int], optional): Bin numbers to use for binning.
                Defaults to config["momentum"]["bins"].
            ranges (List[Tuple], optional): Ranges to use for binning.
                Defaults to config["momentum"]["ranges"].
            **kwds: Keyword argument passed to ``compute``.

        Returns:
            xr.DataArray: pre-binned data-array.
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
            bins (int, dict, tuple, List[int], List[np.ndarray], List[tuple], optional):
                Definition of the bins. Can be any of the following cases:

                - an integer describing the number of bins in on all dimensions
                - a tuple of 3 numbers describing start, end and step of the binning
                  range
                - a np.arrays defining the binning edges
                - a list (NOT a tuple) of any of the above (int, tuple or np.ndarray)
                - a dictionary made of the axes as keys and any of the above as values.

                This takes priority over the axes and range arguments. Defaults to 100.
            axes (Union[str, Sequence[str]], optional): The names of the axes (columns)
                on which to calculate the histogram. The order will be the order of the
                dimensions in the resulting array. Defaults to None.
            ranges (Sequence[Tuple[float, float]], optional): list of tuples containing
                the start and end point of the binning range. Defaults to None.
            **kwds: Keyword arguments:

                - **hist_mode**: Histogram calculation method. "numpy" or "numba". See
                  ``bin_dataframe`` for details. Defaults to
                  config["binning"]["hist_mode"].
                - **mode**: Defines how the results from each partition are combined.
                  "fast", "lean" or "legacy". See ``bin_dataframe`` for details.
                  Defaults to config["binning"]["mode"].
                - **pbar**: Option to show the tqdm progress bar. Defaults to
                  config["binning"]["pbar"].
                - **n_cores**: Number of CPU cores to use for parallelization.
                  Defaults to config["binning"]["num_cores"] or N_CPU-1.
                - **threads_per_worker**: Limit the number of threads that
                  multiprocessing can spawn per binning thread. Defaults to
                  config["binning"]["threads_per_worker"].
                - **threadpool_api**: The API to use for multiprocessing. "blas",
                  "openmp" or None. See ``threadpool_limit`` for details. Defaults to
                  config["binning"]["threadpool_API"].
                - **df_partitions**: A list of dataframe partitions. Defaults to all
                  partitions.

                Additional kwds are passed to ``bin_dataframe``.

        Raises:
            AssertError: Rises when no dataframe has been loaded.

        Returns:
            xr.DataArray: The result of the n-dimensional binning represented in an
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

        for dim in self._binned.dims:
            try:
                self._binned[dim].attrs["unit"] = self._config["dataframe"][
                    "units"
                ][dim]
            except KeyError:
                pass

        self._binned.attrs["units"] = "counts"
        self._binned.attrs["long_name"] = "photoelectron counts"
        self._binned.attrs["metadata"] = self._attributes.metadata

        return self._binned

    def get_normalization_histogram(
        self,
        axis: str = "delay",
        **kwds,
    ) -> np.ndarray:
        """Generates a normalization histogram from the TimeStamps column of the
        dataframe.

        Args:
            axis (str, optional): The axis for which to compute histogram.
                Defaults to "delay".
            **kwds: Keyword arguments:

                -df_partitions (int, optional): Number of dataframe partitions to use.
                  Defaults to all.

        Raises:
            ValueError: Raised if no data are binned.
            ValueError: Raised if 'axis' not in binned coordinates.
            ValueError: Raised if config["dataframe"]["time_stamp_alias"] not found
                in Dataframe.

        Returns:
            np.ndarray: The computed normalization histogram (in TimeStamp units
            per bin).
        """

        if self._binned is None:
            raise ValueError("Need to bin data first!")
        if axis not in self._binned.coords:
            raise ValueError(f"Axis '{axis}' not found in binned data!")

        if (
            self._config["dataframe"]["time_stamp_alias"]
            not in self._dataframe
        ):
            raise ValueError("TimeStamp data not found in Dataframe!")

        self._dataframe["time_per_electron"] = self._dataframe[
            self._config["dataframe"]["time_stamp_alias"]
        ].diff()

        df_partitions = kwds.pop("df_partitions", None)
        if df_partitions is not None:
            dataframe = self._dataframe.partitions[
                0 : min(df_partitions, self._dataframe.npartitions)
            ]
        else:
            dataframe = self._dataframe

        bins = dataframe[axis].map_partitions(
            pd.cut,
            bins=bin_centers_to_bin_edges(self._binned.coords[axis].values),
        )

        histogram = (
            dataframe["time_per_electron"]
            .groupby([bins])
            .sum()
            .compute()
            .values
        )

        return histogram

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
        **kwds,
    ):
        """Plot individual histograms of specified dimensions (axes) from a substituent
        dataframe partition.

        Args:
            dfpid (int): Number of the data frame partition to look at.
            ncol (int, optional): Number of columns in the plot grid. Defaults to 2.
            bins (Sequence[int], optional): Number of bins to use for the speicified
                axes. Defaults to config["histogram"]["bins"].
            axes (Sequence[str], optional): Names of the axes to display.
                Defaults to config["histogram"]["axes"].
            ranges (Sequence[Tuple[float, float]], optional): Value ranges of all
                specified axes. Defaults toconfig["histogram"]["ranges"].
            backend (str, optional): Backend of the plotting library
                ('matplotlib' or 'bokeh'). Defaults to "bokeh".
            legend (bool, optional): Option to include a legend in the histogram plots.
                Defaults to True.
            histkwds (dict, optional): Keyword arguments for histograms
                (see ``matplotlib.pyplot.hist()``). Defaults to {}.
            legkwds (dict, optional): Keyword arguments for legend
                (see ``matplotlib.pyplot.legend()``). Defaults to {}.
            **kwds: Extra keyword arguments passed to
                ``sed.diagnostics.grid_histogram()``.

        Raises:
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
        group_dict_dd = {}
        dfpart = df.get_partition(dfpid)
        cols = dfpart.columns
        for ax in axes:
            group_dict_dd[ax] = dfpart.values[:, cols.get_loc(ax)]
        group_dict = ddf.compute(group_dict_dd)[0]

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

    def save(
        self,
        faddr: str,
        **kwds,
    ):
        """Saves the binned data to the provided path and filename.

        Args:
            faddr (str): Path and name of the file to write. Its extension determines
                the file type to write. Valid file types are:

                - "*.tiff", "*.tif": Saves a TIFF stack.
                - "*.h5", "*.hdf5": Saves an HDF5 file.
                - "*.nxs", "*.nexus": Saves a NeXus file.

            **kwds: Keyword argumens, which are passed to the writer functions:
                For TIFF writing:

                - **alias_dict**: Dictionary of dimension aliases to use.

                For HDF5 writing:

                - **mode**: hdf5 read/write mode. Defaults to "w".

                For NeXus:

                - **reader**: Name of the nexustools reader to use.
                  Defaults to config["nexus"]["reader"]
                - **definiton**: NeXus application definition to use for saving.
                  Must be supported by the used ``reader``. Defaults to
                  config["nexus"]["definition"]
                - **input_files**: A list of input files to pass to the reader.
                  Defaults to config["nexus"]["input_files"]
        """
        if self._binned is None:
            raise NameError("Need to bin data first!")

        extension = pathlib.Path(faddr).suffix

        if extension in (".tif", ".tiff"):
            to_tiff(
                data=self._binned,
                faddr=faddr,
                **kwds,
            )
        elif extension in (".h5", ".hdf5"):
            to_h5(
                data=self._binned,
                faddr=faddr,
                **kwds,
            )
        elif extension in (".nxs", ".nexus"):
            reader = kwds.pop("reader", self._config["nexus"]["reader"])
            definition = kwds.pop(
                "definition",
                self._config["nexus"]["definition"],
            )
            input_files = kwds.pop(
                "input_files",
                self._config["nexus"]["input_files"],
            )
            if isinstance(input_files, str):
                input_files = [input_files]

            to_nexus(
                data=self._binned,
                faddr=faddr,
                reader=reader,
                definition=definition,
                input_files=input_files,
                **kwds,
            )

        else:
            raise NotImplementedError(
                f"Unrecognized file format: {extension}.",
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
