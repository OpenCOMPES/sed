"""This module contains the core class for the sed package

"""
import pathlib
from datetime import datetime
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
from sed.binning.binning import normalization_histogram_from_timed_dataframe
from sed.binning.binning import normalization_histogram_from_timestamps
from sed.calibrator import DelayCalibrator
from sed.calibrator import EnergyCalibrator
from sed.calibrator import MomentumCorrector
from sed.core.config import parse_config
from sed.core.config import save_config
from sed.core.dfops import add_time_stamped_data
from sed.core.dfops import apply_filter
from sed.core.dfops import apply_jitter
from sed.core.metadata import MetaHandler
from sed.diagnostics import grid_histogram
from sed.io import to_h5
from sed.io import to_nexus
from sed.io import to_tiff
from sed.loader import CopyTool
from sed.loader import get_loader
from sed.loader.mpes.loader import get_archiver_data
from sed.loader.mpes.loader import MpesLoader

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
        runs (Sequence[str], optional): List of run identifiers to pass to the loader
            defined in the config. Defaults to None.
        collect_metadata (bool): Option to collect metadata from files.
            Defaults to False.
        verbose (bool, optional): Option to print out diagnostic information.
            Defaults to config["core"]["verbose"] or False.
        **kwds: Keyword arguments passed to the reader.
    """

    def __init__(
        self,
        metadata: dict = None,
        config: Union[dict, str] = None,
        dataframe: Union[pd.DataFrame, ddf.DataFrame] = None,
        files: List[str] = None,
        folder: str = None,
        runs: Sequence[str] = None,
        collect_metadata: bool = False,
        verbose: bool = None,
        **kwds,
    ):
        """Processor class of sed. Contains wrapper functions defining a work flow
        for data correction, calibration, and binning.

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
            runs (Sequence[str], optional): List of run identifiers to pass to the loader
                defined in the config. Defaults to None.
            collect_metadata (bool, optional): Option to collect metadata from files.
                Defaults to False.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"] or False.
            **kwds: Keyword arguments passed to parse_config and to the reader.
        """
        config_kwds = {
            key: value for key, value in kwds.items() if key in parse_config.__code__.co_varnames
        }
        for key in config_kwds.keys():
            del kwds[key]
        self._config = parse_config(config, **config_kwds)
        num_cores = self._config.get("binning", {}).get("num_cores", N_CPU - 1)
        if num_cores >= N_CPU:
            num_cores = N_CPU - 1
        self._config["binning"]["num_cores"] = num_cores

        if verbose is None:
            self.verbose = self._config["core"].get("verbose", False)
        else:
            self.verbose = verbose

        self._dataframe: Union[pd.DataFrame, ddf.DataFrame] = None
        self._timed_dataframe: Union[pd.DataFrame, ddf.DataFrame] = None
        self._files: List[str] = []

        self._binned: xr.DataArray = None
        self._pre_binned: xr.DataArray = None
        self._normalization_histogram: xr.DataArray = None
        self._normalized: xr.DataArray = None

        self._attributes = MetaHandler(meta=metadata)

        loader_name = self._config["core"]["loader"]
        self.loader = get_loader(
            loader_name=loader_name,
            config=self._config,
        )

        self.ec = EnergyCalibrator(
            loader=get_loader(
                loader_name=loader_name,
                config=self._config,
            ),
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
        if dataframe is not None or files is not None or folder is not None or runs is not None:
            self.load(
                dataframe=dataframe,
                metadata=metadata,
                files=files,
                folder=folder,
                runs=runs,
                collect_metadata=collect_metadata,
                **kwds,
            )

    def __repr__(self):
        if self._dataframe is None:
            df_str = "Data Frame: No Data loaded"
        else:
            df_str = self._dataframe.__repr__()
        attributes_str = f"Metadata: {self._attributes.metadata}"
        pretty_str = df_str + "\n" + attributes_str
        return pretty_str

    @property
    def dataframe(self) -> Union[pd.DataFrame, ddf.DataFrame]:
        """Accessor to the underlying dataframe.

        Returns:
            Union[pd.DataFrame, ddf.DataFrame]: Dataframe object.
        """
        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe: Union[pd.DataFrame, ddf.DataFrame]):
        """Setter for the underlying dataframe.

        Args:
            dataframe (Union[pd.DataFrame, ddf.DataFrame]): The dataframe object to set.
        """
        if not isinstance(dataframe, (pd.DataFrame, ddf.DataFrame)) or not isinstance(
            dataframe,
            self._dataframe.__class__,
        ):
            raise ValueError(
                "'dataframe' has to be a Pandas or Dask dataframe and has to be of the same kind "
                "as the dataframe loaded into the SedProcessor!.\n"
                f"Loaded type: {self._dataframe.__class__}, provided type: {dataframe}.",
            )
        self._dataframe = dataframe

    @property
    def timed_dataframe(self) -> Union[pd.DataFrame, ddf.DataFrame]:
        """Accessor to the underlying timed_dataframe.

        Returns:
            Union[pd.DataFrame, ddf.DataFrame]: Timed Dataframe object.
        """
        return self._timed_dataframe

    @timed_dataframe.setter
    def timed_dataframe(self, timed_dataframe: Union[pd.DataFrame, ddf.DataFrame]):
        """Setter for the underlying timed dataframe.

        Args:
            timed_dataframe (Union[pd.DataFrame, ddf.DataFrame]): The timed dataframe object to set
        """
        if not isinstance(timed_dataframe, (pd.DataFrame, ddf.DataFrame)) or not isinstance(
            timed_dataframe,
            self._timed_dataframe.__class__,
        ):
            raise ValueError(
                "'timed_dataframe' has to be a Pandas or Dask dataframe and has to be of the same "
                "kind as the dataframe loaded into the SedProcessor!.\n"
                f"Loaded type: {self._timed_dataframe.__class__}, "
                f"provided type: {timed_dataframe}.",
            )
        self._timed_dataframe = timed_dataframe

    @property
    def attributes(self) -> dict:
        """Accessor to the metadata dict.

        Returns:
            dict: The metadata dict.
        """
        return self._attributes.metadata

    def add_attribute(self, attributes: dict, name: str, **kwds):
        """Function to add element to the attributes dict.

        Args:
            attributes (dict): The attributes dictionary object to add.
            name (str): Key under which to add the dictionary to the attributes.
        """
        self._attributes.add(
            entry=attributes,
            name=name,
            **kwds,
        )

    @property
    def config(self) -> Dict[Any, Any]:
        """Getter attribute for the config dictionary

        Returns:
            Dict: The config dictionary.
        """
        return self._config

    @property
    def files(self) -> List[str]:
        """Getter attribute for the list of files

        Returns:
            List[str]: The list of loaded files
        """
        return self._files

    @property
    def binned(self) -> xr.DataArray:
        """Getter attribute for the binned data array

        Returns:
            xr.DataArray: The binned data array
        """
        if self._binned is None:
            raise ValueError("No binned data available, need to compute histogram first!")
        return self._binned

    @property
    def normalized(self) -> xr.DataArray:
        """Getter attribute for the normalized data array

        Returns:
            xr.DataArray: The normalized data array
        """
        if self._normalized is None:
            raise ValueError(
                "No normalized data available, compute data with normalization enabled!",
            )
        return self._normalized

    @property
    def normalization_histogram(self) -> xr.DataArray:
        """Getter attribute for the normalization histogram

        Returns:
            xr.DataArray: The normalizazion histogram
        """
        if self._normalization_histogram is None:
            raise ValueError("No normalization histogram available, generate histogram first!")
        return self._normalization_histogram

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
        runs: Sequence[str] = None,
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
            runs (Sequence[str], optional): List of run identifiers to pass to the
                loader. Defaults to None.
            folder (str, optional): Folder path to pass to the loader.
                Defaults to None.
            collect_metadata (bool, optional): Option for collecting metadata in the reader.
            **kwds: Keyword parameters passed to the reader.

        Raises:
            ValueError: Raised if no valid input is provided.
        """
        if metadata is None:
            metadata = {}
        if dataframe is not None:
            timed_dataframe = kwds.pop("timed_dataframe", None)
        elif runs is not None:
            # If runs are provided, we only use the copy tool if also folder is provided.
            # In that case, we copy the whole provided base folder tree, and pass the copied
            # version to the loader as base folder to look for the runs.
            if folder is not None:
                dataframe, timed_dataframe, metadata = self.loader.read_dataframe(
                    folders=cast(str, self.cpy(folder)),
                    runs=runs,
                    metadata=metadata,
                    collect_metadata=collect_metadata,
                    **kwds,
                )
            else:
                dataframe, timed_dataframe, metadata = self.loader.read_dataframe(
                    runs=runs,
                    metadata=metadata,
                    collect_metadata=collect_metadata,
                    **kwds,
                )

        elif folder is not None:
            dataframe, timed_dataframe, metadata = self.loader.read_dataframe(
                folders=cast(str, self.cpy(folder)),
                metadata=metadata,
                collect_metadata=collect_metadata,
                **kwds,
            )
        elif files is not None:
            dataframe, timed_dataframe, metadata = self.loader.read_dataframe(
                files=cast(List[str], self.cpy(files)),
                metadata=metadata,
                collect_metadata=collect_metadata,
                **kwds,
            )
        else:
            raise ValueError(
                "Either 'dataframe', 'files', 'folder', or 'runs' needs to be provided!",
            )

        self._dataframe = dataframe
        self._timed_dataframe = timed_dataframe
        self._files = self.loader.files

        for key in metadata:
            self._attributes.add(
                entry=metadata[key],
                name=key,
                duplicate_policy="merge",
            )

    def filter_column(
        self,
        column: str,
        min_value: float = -np.inf,
        max_value: float = np.inf,
    ) -> None:
        """Filter values in a column which are outside of a given range

        Args:
            column (str): Name of the column to filter
            min_value (float, optional): Minimum value to keep. Defaults to None.
            max_value (float, optional): Maximum value to keep. Defaults to None.
        """
        if column != "index" and column not in self._dataframe.columns:
            raise KeyError(f"Column {column} not found in dataframe!")
        if min_value >= max_value:
            raise ValueError("min_value has to be smaller than max_value!")
        if self._dataframe is not None:
            self._dataframe = apply_filter(
                self._dataframe,
                col=column,
                lower_bound=min_value,
                upper_bound=max_value,
            )
        if self._timed_dataframe is not None and column in self._timed_dataframe.columns:
            self._timed_dataframe = apply_filter(
                self._timed_dataframe,
                column,
                lower_bound=min_value,
                upper_bound=max_value,
            )
        metadata = {
            "filter": {
                "column": column,
                "min_value": min_value,
                "max_value": max_value,
            },
        }
        self._attributes.add(metadata, "filter", duplicate_policy="merge")

    # Momentum calibration workflow
    # 1. Bin raw detector data for distortion correction
    def bin_and_load_momentum_calibration(
        self,
        df_partitions: Union[int, Sequence[int]] = 100,
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
            df_partitions (Union[int, Sequence[int]], optional): Number of dataframe partitions
                to use for the initial binning. Defaults to 100.
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
    def define_features(
        self,
        features: np.ndarray = None,
        rotation_symmetry: int = 6,
        auto_detect: bool = False,
        include_center: bool = True,
        apply: bool = False,
        **kwds,
    ):
        """2. Step of the distortion correction workflow: Define feature points in
        momentum space. They can be either manually selected using a GUI tool, be
        ptovided as list of feature points, or auto-generated using a
        feature-detection algorithm.

        Args:
            features (np.ndarray, optional): np.ndarray of features. Defaults to None.
            rotation_symmetry (int, optional): Number of rotational symmetry axes.
                Defaults to 6.
            auto_detect (bool, optional): Whether to auto-detect the features.
                Defaults to False.
            include_center (bool, optional): Option to include a point at the center
                in the feature list. Defaults to True.
            apply (bool, optional): Option to directly apply the values and select the
                slice. Defaults to False.
            **kwds: Keyword arguments for ``MomentumCorrector.feature_extract()`` and
                ``MomentumCorrector.feature_select()``.
        """
        if auto_detect:  # automatic feature selection
            sigma = kwds.pop("sigma", self._config["momentum"]["sigma"])
            fwhm = kwds.pop("fwhm", self._config["momentum"]["fwhm"])
            sigma_radius = kwds.pop(
                "sigma_radius",
                self._config["momentum"]["sigma_radius"],
            )
            self.mc.feature_extract(
                sigma=sigma,
                fwhm=fwhm,
                sigma_radius=sigma_radius,
                rotsym=rotation_symmetry,
                **kwds,
            )
            features = self.mc.peaks

        self.mc.feature_select(
            rotsym=rotation_symmetry,
            include_center=include_center,
            features=features,
            apply=apply,
            **kwds,
        )

    # 3. Generate the spline warp correction from momentum features.
    # If no features have been selected before, use class defaults.
    def generate_splinewarp(
        self,
        use_center: bool = None,
        verbose: bool = None,
        **kwds,
    ):
        """3. Step of the distortion correction workflow: Generate the correction
        function restoring the symmetry in the image using a splinewarp algortihm.

        Args:
            use_center (bool, optional): Option to use the position of the
                center point in the correction. Default is read from config, or set to True.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"].
            **kwds: Keyword arguments for MomentumCorrector.spline_warp_estimate().
        """
        if verbose is None:
            verbose = self.verbose

        self.mc.spline_warp_estimate(use_center=use_center, verbose=verbose, **kwds)

        if self.mc.slice is not None and verbose:
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

    # 3a. Save spline-warp parameters to config file.
    def save_splinewarp(
        self,
        filename: str = None,
        overwrite: bool = False,
    ):
        """Save the generated spline-warp parameters to the folder config file.

        Args:
            filename (str, optional): Filename of the config dictionary to save to.
                Defaults to "sed_config.yaml" in the current folder.
            overwrite (bool, optional): Option to overwrite the present dictionary.
                Defaults to False.
        """
        if filename is None:
            filename = "sed_config.yaml"
        if len(self.mc.correction) == 0:
            raise ValueError("No momentum correction parameters to save!")
        correction = {}
        for key, value in self.mc.correction.items():
            if key in ["reference_points", "target_points", "cdeform_field", "rdeform_field"]:
                continue
            if key in ["use_center", "rotation_symmetry"]:
                correction[key] = value
            elif key in ["center_point", "ascale"]:
                correction[key] = [float(i) for i in value]
            elif key in ["outer_points", "feature_points"]:
                correction[key] = []
                for point in value:
                    correction[key].append([float(i) for i in point])
            else:
                correction[key] = float(value)

        if "creation_date" not in correction:
            correction["creation_date"] = datetime.now().timestamp()

        config = {
            "momentum": {
                "correction": correction,
            },
        }
        save_config(config, filename, overwrite)
        print(f'Saved momentum correction parameters to "{filename}".')

    # 4. Pose corrections. Provide interactive interface for correcting
    # scaling, shift and rotation
    def pose_adjustment(
        self,
        transformations: Dict[str, Any] = None,
        apply: bool = False,
        use_correction: bool = True,
        reset: bool = True,
        verbose: bool = None,
        **kwds,
    ):
        """3. step of the distortion correction workflow: Generate an interactive panel
        to adjust affine transformations that are applied to the image. Applies first
        a scaling, next an x/y translation, and last a rotation around the center of
        the image.

        Args:
            transformations (dict, optional): Dictionary with transformations.
                Defaults to self.transformations or config["momentum"]["transformtions"].
            apply (bool, optional): Option to directly apply the provided
                transformations. Defaults to False.
            use_correction (bool, option): Whether to use the spline warp correction
                or not. Defaults to True.
            reset (bool, optional): Option to reset the correction before transformation.
                Defaults to True.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"].
            **kwds: Keyword parameters defining defaults for the transformations:

                - **scale** (float): Initial value of the scaling slider.
                - **xtrans** (float): Initial value of the xtrans slider.
                - **ytrans** (float): Initial value of the ytrans slider.
                - **angle** (float): Initial value of the angle slider.
        """
        if verbose is None:
            verbose = self.verbose

        # Generate homomorphy as default if no distortion correction has been applied
        if self.mc.slice_corrected is None:
            if self.mc.slice is None:
                self.mc.slice = np.zeros(self._config["momentum"]["bins"][0:2])
            self.mc.slice_corrected = self.mc.slice

        if not use_correction:
            self.mc.reset_deformation()

        if self.mc.cdeform_field is None or self.mc.rdeform_field is None:
            # Generate distortion correction from config values
            self.mc.spline_warp_estimate(verbose=verbose)

        self.mc.pose_adjustment(
            transformations=transformations,
            apply=apply,
            reset=reset,
            verbose=verbose,
            **kwds,
        )

    # 4a. Save pose adjustment parameters to config file.
    def save_transformations(
        self,
        filename: str = None,
        overwrite: bool = False,
    ):
        """Save the pose adjustment parameters to the folder config file.

        Args:
            filename (str, optional): Filename of the config dictionary to save to.
                Defaults to "sed_config.yaml" in the current folder.
            overwrite (bool, optional): Option to overwrite the present dictionary.
                Defaults to False.
        """
        if filename is None:
            filename = "sed_config.yaml"
        if len(self.mc.transformations) == 0:
            raise ValueError("No momentum transformation parameters to save!")
        transformations = {}
        for key, value in self.mc.transformations.items():
            transformations[key] = float(value)

        if "creation_date" not in transformations:
            transformations["creation_date"] = datetime.now().timestamp()

        config = {
            "momentum": {
                "transformations": transformations,
            },
        }
        save_config(config, filename, overwrite)
        print(f'Saved momentum transformation parameters to "{filename}".')

    # 5. Apply the momentum correction to the dataframe
    def apply_momentum_correction(
        self,
        preview: bool = False,
        verbose: bool = None,
        **kwds,
    ):
        """Applies the distortion correction and pose adjustment (optional)
        to the dataframe.

        Args:
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"].
            **kwds: Keyword parameters for ``MomentumCorrector.apply_correction``:

                - **rdeform_field** (np.ndarray, optional): Row deformation field.
                - **cdeform_field** (np.ndarray, optional): Column deformation field.
                - **inv_dfield** (np.ndarray, optional): Inverse deformation field.

        """
        if verbose is None:
            verbose = self.verbose

        x_column = self._config["dataframe"]["x_column"]
        y_column = self._config["dataframe"]["y_column"]

        if self._dataframe is not None:
            if verbose:
                print("Adding corrected X/Y columns to dataframe:")
            df, metadata = self.mc.apply_corrections(
                df=self._dataframe,
                verbose=verbose,
                **kwds,
            )
            if (
                self._timed_dataframe is not None
                and x_column in self._timed_dataframe.columns
                and y_column in self._timed_dataframe.columns
            ):
                tdf, _ = self.mc.apply_corrections(
                    self._timed_dataframe,
                    verbose=False,
                    **kwds,
                )

            # Add Metadata
            self._attributes.add(
                metadata,
                "momentum_correction",
                duplicate_policy="merge",
            )
            self._dataframe = df
            if (
                self._timed_dataframe is not None
                and x_column in self._timed_dataframe.columns
                and y_column in self._timed_dataframe.columns
            ):
                self._timed_dataframe = tdf
        else:
            raise ValueError("No dataframe loaded!")
        if preview:
            print(self._dataframe.head(10))
        else:
            if self.verbose:
                print(self._dataframe)

    # Momentum calibration work flow
    # 1. Calculate momentum calibration
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
        """1. step of the momentum calibration workflow. Calibrate momentum
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
            point_b = self._config["momentum"]["center_pixel"]

        self.mc.select_k_range(
            point_a=point_a,
            point_b=point_b,
            k_distance=k_distance,
            k_coord_a=k_coord_a,
            k_coord_b=k_coord_b,
            equiscale=equiscale,
            apply=apply,
        )

    # 1a. Save momentum calibration parameters to config file.
    def save_momentum_calibration(
        self,
        filename: str = None,
        overwrite: bool = False,
    ):
        """Save the generated momentum calibration parameters to the folder config file.

        Args:
            filename (str, optional): Filename of the config dictionary to save to.
                Defaults to "sed_config.yaml" in the current folder.
            overwrite (bool, optional): Option to overwrite the present dictionary.
                Defaults to False.
        """
        if filename is None:
            filename = "sed_config.yaml"
        if len(self.mc.calibration) == 0:
            raise ValueError("No momentum calibration parameters to save!")
        calibration = {}
        for key, value in self.mc.calibration.items():
            if key in ["kx_axis", "ky_axis", "grid", "extent"]:
                continue

            calibration[key] = float(value)

        if "creation_date" not in calibration:
            calibration["creation_date"] = datetime.now().timestamp()

        config = {"momentum": {"calibration": calibration}}
        save_config(config, filename, overwrite)
        print(f"Saved momentum calibration parameters to {filename}")

    # 2. Apply correction and calibration to the dataframe
    def apply_momentum_calibration(
        self,
        calibration: dict = None,
        preview: bool = False,
        verbose: bool = None,
        **kwds,
    ):
        """2. step of the momentum calibration work flow: Apply the momentum
        calibration stored in the class to the dataframe. If corrected X/Y axis exist,
        these are used.

        Args:
            calibration (dict, optional): Optional dictionary with calibration data to
                use. Defaults to None.
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"].
            **kwds: Keyword args passed to ``DelayCalibrator.append_delay_axis``.
        """
        if verbose is None:
            verbose = self.verbose

        x_column = self._config["dataframe"]["x_column"]
        y_column = self._config["dataframe"]["y_column"]

        if self._dataframe is not None:
            if verbose:
                print("Adding kx/ky columns to dataframe:")
            df, metadata = self.mc.append_k_axis(
                df=self._dataframe,
                calibration=calibration,
                **kwds,
            )
            if (
                self._timed_dataframe is not None
                and x_column in self._timed_dataframe.columns
                and y_column in self._timed_dataframe.columns
            ):
                tdf, _ = self.mc.append_k_axis(
                    df=self._timed_dataframe,
                    calibration=calibration,
                    **kwds,
                )

            # Add Metadata
            self._attributes.add(
                metadata,
                "momentum_calibration",
                duplicate_policy="merge",
            )
            self._dataframe = df
            if (
                self._timed_dataframe is not None
                and x_column in self._timed_dataframe.columns
                and y_column in self._timed_dataframe.columns
            ):
                self._timed_dataframe = tdf
        else:
            raise ValueError("No dataframe loaded!")
        if preview:
            print(self._dataframe.head(10))
        else:
            if self.verbose:
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
            **kwds: Keyword parameters passed to ``EnergyCalibrator.adjust_energy_correction()``.
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

    # 1a. Save energy correction parameters to config file.
    def save_energy_correction(
        self,
        filename: str = None,
        overwrite: bool = False,
    ):
        """Save the generated energy correction parameters to the folder config file.

        Args:
            filename (str, optional): Filename of the config dictionary to save to.
                Defaults to "sed_config.yaml" in the current folder.
            overwrite (bool, optional): Option to overwrite the present dictionary.
                Defaults to False.
        """
        if filename is None:
            filename = "sed_config.yaml"
        if len(self.ec.correction) == 0:
            raise ValueError("No energy correction parameters to save!")
        correction = {}
        for key, val in self.ec.correction.items():
            if key == "correction_type":
                correction[key] = val
            elif key == "center":
                correction[key] = [float(i) for i in val]
            else:
                correction[key] = float(val)

        if "creation_date" not in correction:
            correction["creation_date"] = datetime.now().timestamp()

        config = {"energy": {"correction": correction}}
        save_config(config, filename, overwrite)
        print(f"Saved energy correction parameters to {filename}")

    # 2. Apply energy correction to dataframe
    def apply_energy_correction(
        self,
        correction: dict = None,
        preview: bool = False,
        verbose: bool = None,
        **kwds,
    ):
        """2. step of the energy correction workflow: Apply the enery correction
        parameters stored in the class to the dataframe.

        Args:
            correction (dict, optional): Dictionary containing the correction
                parameters. Defaults to config["energy"]["calibration"].
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"].
            **kwds:
                Keyword args passed to ``EnergyCalibrator.apply_energy_correction()``.
        """
        if verbose is None:
            verbose = self.verbose

        tof_column = self._config["dataframe"]["tof_column"]

        if self._dataframe is not None:
            if verbose:
                print("Applying energy correction to dataframe...")
            df, metadata = self.ec.apply_energy_correction(
                df=self._dataframe,
                correction=correction,
                verbose=verbose,
                **kwds,
            )
            if self._timed_dataframe is not None and tof_column in self._timed_dataframe.columns:
                tdf, _ = self.ec.apply_energy_correction(
                    df=self._timed_dataframe,
                    correction=correction,
                    verbose=False,
                    **kwds,
                )

            # Add Metadata
            self._attributes.add(
                metadata,
                "energy_correction",
            )
            self._dataframe = df
            if self._timed_dataframe is not None and tof_column in self._timed_dataframe.columns:
                self._timed_dataframe = tdf
        else:
            raise ValueError("No dataframe loaded!")
        if preview:
            print(self._dataframe.head(10))
        else:
            if verbose:
                print(self._dataframe)

    # Energy calibrator workflow
    # 1. Load and normalize data
    def load_bias_series(
        self,
        binned_data: Union[xr.DataArray, Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        data_files: List[str] = None,
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
        single-event files, or load binned bias/TOF traces.

        Args:
            binned_data (Union[xr.DataArray, Tuple[np.ndarray, np.ndarray, np.ndarray]], optional):
                Binned data If provided as DataArray, Needs to contain dimensions
                config["dataframe"]["tof_column"] and config["dataframe"]["bias_column"]. If
                provided as tuple, needs to contain elements tof, biases, traces.
            data_files (List[str], optional): list of file paths to bin
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
        if binned_data is not None:
            if isinstance(binned_data, xr.DataArray):
                if (
                    self._config["dataframe"]["tof_column"] not in binned_data.dims
                    or self._config["dataframe"]["bias_column"] not in binned_data.dims
                ):
                    raise ValueError(
                        "If binned_data is provided as an xarray, it needs to contain dimensions "
                        f"'{self._config['dataframe']['tof_column']}' and "
                        f"'{self._config['dataframe']['bias_column']}'!.",
                    )
                tof = binned_data.coords[self._config["dataframe"]["tof_column"]].values
                biases = binned_data.coords[self._config["dataframe"]["bias_column"]].values
                traces = binned_data.values[:, :]
            else:
                try:
                    (tof, biases, traces) = binned_data
                except ValueError as exc:
                    raise ValueError(
                        "If binned_data is provided as tuple, it needs to contain "
                        "(tof, biases, traces)!",
                    ) from exc
            self.ec.load_data(biases=biases, traces=traces, tof=tof)

        elif data_files is not None:
            self.ec.bin_data(
                data_files=cast(List[str], self.cpy(data_files)),
                axes=axes,
                bins=bins,
                ranges=ranges,
                biases=biases,
                bias_key=bias_key,
            )

        else:
            raise ValueError("Either binned_data or data_files needs to be provided!")

        if (normalize is not None and normalize is True) or (
            normalize is None and self._config["energy"]["normalize"]
        ):
            if span is None:
                span = self._config["energy"]["normalize_span"]
            if order is None:
                order = self._config["energy"]["normalize_order"]
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
        apply: bool = False,
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
            ref_id (int, optional): The id of the trace the range refers to.
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
            apply (bool, optional): Option to directly apply the provided parameters.
                Defaults to False.
        """
        if radius is None:
            radius = self._config["energy"]["fastdtw_radius"]
        if peak_window is None:
            peak_window = self._config["energy"]["peak_window"]
        if not infer_others:
            self.ec.add_ranges(
                ranges=ranges,
                ref_id=ref_id,
                infer_others=infer_others,
                mode=mode,
                radius=radius,
            )
            print(self.ec.featranges)
            try:
                self.ec.feature_extract(peak_window=peak_window)
                self.ec.view(
                    traces=self.ec.traces_normed,
                    segs=self.ec.featranges,
                    xaxis=self.ec.tof,
                    peaks=self.ec.peaks,
                    backend="bokeh",
                )
            except IndexError:
                print("Could not determine all peaks!")
                raise
        else:
            # New adjustment tool
            assert isinstance(ranges, tuple)
            self.ec.adjust_ranges(
                ranges=ranges,
                ref_id=ref_id,
                traces=self.ec.traces_normed,
                infer_others=infer_others,
                radius=radius,
                peak_window=peak_window,
                apply=apply,
            )

    # 3. Fit the energy calibration relation
    def calibrate_energy_axis(
        self,
        ref_id: int,
        ref_energy: float,
        method: str = None,
        energy_scale: str = None,
        verbose: bool = None,
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
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"].
            **kwds**: Keyword parameters passed to ``EnergyCalibrator.calibrate()``.
        """
        if verbose is None:
            verbose = self.verbose

        if method is None:
            method = self._config["energy"]["calibration_method"]

        if energy_scale is None:
            energy_scale = self._config["energy"]["energy_scale"]

        self.ec.calibrate(
            ref_id=ref_id,
            ref_energy=ref_energy,
            method=method,
            energy_scale=energy_scale,
            verbose=verbose,
            **kwds,
        )
        if verbose:
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

    # 3a. Save energy calibration parameters to config file.
    def save_energy_calibration(
        self,
        filename: str = None,
        overwrite: bool = False,
    ):
        """Save the generated energy calibration parameters to the folder config file.

        Args:
            filename (str, optional): Filename of the config dictionary to save to.
                Defaults to "sed_config.yaml" in the current folder.
            overwrite (bool, optional): Option to overwrite the present dictionary.
                Defaults to False.
        """
        if filename is None:
            filename = "sed_config.yaml"
        if len(self.ec.calibration) == 0:
            raise ValueError("No energy calibration parameters to save!")
        calibration = {}
        for key, value in self.ec.calibration.items():
            if key in ["axis", "refid", "Tmat", "bvec"]:
                continue
            if key == "energy_scale":
                calibration[key] = value
            elif key == "coeffs":
                calibration[key] = [float(i) for i in value]
            else:
                calibration[key] = float(value)

        if "creation_date" not in calibration:
            calibration["creation_date"] = datetime.now().timestamp()

        config = {"energy": {"calibration": calibration}}
        save_config(config, filename, overwrite)
        print(f'Saved energy calibration parameters to "{filename}".')

    # 4. Apply energy calibration to the dataframe
    def append_energy_axis(
        self,
        calibration: dict = None,
        preview: bool = False,
        verbose: bool = None,
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
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"].
            **kwds:
                Keyword args passed to ``EnergyCalibrator.append_energy_axis()``.
        """
        if verbose is None:
            verbose = self.verbose

        tof_column = self._config["dataframe"]["tof_column"]

        if self._dataframe is not None:
            if verbose:
                print("Adding energy column to dataframe:")
            df, metadata = self.ec.append_energy_axis(
                df=self._dataframe,
                calibration=calibration,
                verbose=verbose,
                **kwds,
            )
            if self._timed_dataframe is not None and tof_column in self._timed_dataframe.columns:
                tdf, _ = self.ec.append_energy_axis(
                    df=self._timed_dataframe,
                    calibration=calibration,
                    verbose=False,
                    **kwds,
                )

            # Add Metadata
            self._attributes.add(
                metadata,
                "energy_calibration",
                duplicate_policy="merge",
            )
            self._dataframe = df
            if self._timed_dataframe is not None and tof_column in self._timed_dataframe.columns:
                self._timed_dataframe = tdf

        else:
            raise ValueError("No dataframe loaded!")
        if preview:
            print(self._dataframe.head(10))
        else:
            if verbose:
                print(self._dataframe)

    def add_energy_offset(
        self,
        constant: float = None,
        columns: Union[str, Sequence[str]] = None,
        weights: Union[float, Sequence[float]] = None,
        reductions: Union[str, Sequence[str]] = None,
        preserve_mean: Union[bool, Sequence[bool]] = None,
        preview: bool = False,
        verbose: bool = None,
    ) -> None:
        """Shift the energy axis of the dataframe by a given amount.

        Args:
            constant (float, optional): The constant to shift the energy axis by.
            columns (Union[str, Sequence[str]]): Name of the column(s) to apply the shift from.
            weights (Union[float, Sequence[float]]): weights to apply to the columns.
                Can also be used to flip the sign (e.g. -1). Defaults to 1.
            preserve_mean (bool): Whether to subtract the mean of the column before applying the
                shift. Defaults to False.
            reductions (str): The reduction to apply to the column. Should be an available method
                of dask.dataframe.Series. For example "mean". In this case the function is applied
                to the column to generate a single value for the whole dataset. If None, the shift
                is applied per-dataframe-row. Defaults to None. Currently only "mean" is supported.
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"].

        Raises:
            ValueError: If the energy column is not in the dataframe.
        """
        if verbose is None:
            verbose = self.verbose

        energy_column = self._config["dataframe"]["energy_column"]
        if energy_column not in self._dataframe.columns:
            raise ValueError(
                f"Energy column {energy_column} not found in dataframe! "
                "Run `append_energy_axis()` first.",
            )
        if self.dataframe is not None:
            if verbose:
                print("Adding energy offset to dataframe:")
            df, metadata = self.ec.add_offsets(
                df=self._dataframe,
                constant=constant,
                columns=columns,
                energy_column=energy_column,
                weights=weights,
                reductions=reductions,
                preserve_mean=preserve_mean,
                verbose=verbose,
            )
            if self._timed_dataframe is not None and energy_column in self._timed_dataframe.columns:
                tdf, _ = self.ec.add_offsets(
                    df=self._timed_dataframe,
                    constant=constant,
                    columns=columns,
                    energy_column=energy_column,
                    weights=weights,
                    reductions=reductions,
                    preserve_mean=preserve_mean,
                )

            self._attributes.add(
                metadata,
                "add_energy_offset",
                # TODO: allow only appending when no offset along this column(s) was applied
                # TODO: clear memory of modifications if the energy axis is recalculated
                duplicate_policy="append",
            )
            self._dataframe = df
            if self._timed_dataframe is not None and energy_column in self._timed_dataframe.columns:
                self._timed_dataframe = tdf
        else:
            raise ValueError("No dataframe loaded!")
        if preview:
            print(self._dataframe.head(10))
        elif verbose:
            print(self._dataframe)

    def save_energy_offset(
        self,
        filename: str = None,
        overwrite: bool = False,
    ):
        """Save the generated energy calibration parameters to the folder config file.

        Args:
            filename (str, optional): Filename of the config dictionary to save to.
                Defaults to "sed_config.yaml" in the current folder.
            overwrite (bool, optional): Option to overwrite the present dictionary.
                Defaults to False.
        """
        if filename is None:
            filename = "sed_config.yaml"
        if len(self.ec.offsets) == 0:
            raise ValueError("No energy offset parameters to save!")

        if "creation_date" not in self.ec.offsets.keys():
            self.ec.offsets["creation_date"] = datetime.now().timestamp()

        config = {"energy": {"offsets": self.ec.offsets}}
        save_config(config, filename, overwrite)
        print(f'Saved energy offset parameters to "{filename}".')

    def append_tof_ns_axis(
        self,
        preview: bool = False,
        verbose: bool = None,
        **kwds,
    ):
        """Convert time-of-flight channel steps to nanoseconds.

        Args:
            tof_ns_column (str, optional): Name of the generated column containing the
                time-of-flight in nanosecond.
                Defaults to config["dataframe"]["tof_ns_column"].
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"].
            **kwds: additional arguments are passed to ``EnergyCalibrator.tof_step_to_ns()``.

        """
        if verbose is None:
            verbose = self.verbose

        tof_column = self._config["dataframe"]["tof_column"]

        if self._dataframe is not None:
            if verbose:
                print("Adding time-of-flight column in nanoseconds to dataframe:")
            # TODO assert order of execution through metadata

            df, metadata = self.ec.append_tof_ns_axis(
                df=self._dataframe,
                **kwds,
            )
            if self._timed_dataframe is not None and tof_column in self._timed_dataframe.columns:
                tdf, _ = self.ec.append_tof_ns_axis(
                    df=self._timed_dataframe,
                    **kwds,
                )

            self._attributes.add(
                metadata,
                "tof_ns_conversion",
                duplicate_policy="overwrite",
            )
            self._dataframe = df
            if self._timed_dataframe is not None and tof_column in self._timed_dataframe.columns:
                self._timed_dataframe = tdf
        else:
            raise ValueError("No dataframe loaded!")
        if preview:
            print(self._dataframe.head(10))
        else:
            if verbose:
                print(self._dataframe)

    def align_dld_sectors(
        self,
        sector_delays: np.ndarray = None,
        preview: bool = False,
        verbose: bool = None,
        **kwds,
    ):
        """Align the 8s sectors of the HEXTOF endstation.

        Args:
            sector_delays (np.ndarray, optional): Array containing the sector delays. Defaults to
                config["dataframe"]["sector_delays"].
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"].
            **kwds: additional arguments are passed to ``EnergyCalibrator.align_dld_sectors()``.
        """
        if verbose is None:
            verbose = self.verbose

        tof_column = self._config["dataframe"]["tof_column"]

        if self._dataframe is not None:
            if verbose:
                print("Aligning 8s sectors of dataframe")
            # TODO assert order of execution through metadata

            df, metadata = self.ec.align_dld_sectors(
                df=self._dataframe,
                sector_delays=sector_delays,
                **kwds,
            )
            if self._timed_dataframe is not None and tof_column in self._timed_dataframe.columns:
                tdf, _ = self.ec.align_dld_sectors(
                    df=self._timed_dataframe,
                    sector_delays=sector_delays,
                    **kwds,
                )

            self._attributes.add(
                metadata,
                "dld_sector_alignment",
                duplicate_policy="raise",
            )
            self._dataframe = df
            if self._timed_dataframe is not None and tof_column in self._timed_dataframe.columns:
                self._timed_dataframe = tdf
        else:
            raise ValueError("No dataframe loaded!")
        if preview:
            print(self._dataframe.head(10))
        else:
            if verbose:
                print(self._dataframe)

    # Delay calibration function
    def calibrate_delay_axis(
        self,
        delay_range: Tuple[float, float] = None,
        datafile: str = None,
        preview: bool = False,
        verbose: bool = None,
        **kwds,
    ):
        """Append delay column to dataframe. Either provide delay ranges, or read
        them from a file.

        Args:
            delay_range (Tuple[float, float], optional): The scanned delay range in
                picoseconds. Defaults to None.
            datafile (str, optional): The file from which to read the delay ranges.
                Defaults to None.
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"].
            **kwds: Keyword args passed to ``DelayCalibrator.append_delay_axis``.
        """
        if verbose is None:
            verbose = self.verbose

        adc_column = self._config["dataframe"]["adc_column"]
        if adc_column not in self._dataframe.columns:
            raise ValueError(f"ADC column {adc_column} not found in dataframe, cannot calibrate!")

        if self._dataframe is not None:
            if verbose:
                print("Adding delay column to dataframe:")

            if delay_range is None and datafile is None:
                if len(self.dc.calibration) == 0:
                    try:
                        datafile = self._files[0]
                    except IndexError:
                        print(
                            "No datafile available, specify either",
                            " 'datafile' or 'delay_range'",
                        )
                        raise

            df, metadata = self.dc.append_delay_axis(
                self._dataframe,
                delay_range=delay_range,
                datafile=datafile,
                verbose=verbose,
                **kwds,
            )
            if self._timed_dataframe is not None and adc_column in self._timed_dataframe.columns:
                tdf, _ = self.dc.append_delay_axis(
                    self._timed_dataframe,
                    delay_range=delay_range,
                    datafile=datafile,
                    verbose=False,
                    **kwds,
                )

            # Add Metadata
            self._attributes.add(
                metadata,
                "delay_calibration",
                duplicate_policy="overwrite",
            )
            self._dataframe = df
            if self._timed_dataframe is not None and adc_column in self._timed_dataframe.columns:
                self._timed_dataframe = tdf
        else:
            raise ValueError("No dataframe loaded!")
        if preview:
            print(self._dataframe.head(10))
        else:
            if self.verbose:
                print(self._dataframe)

    def save_delay_calibration(
        self,
        filename: str = None,
        overwrite: bool = False,
    ) -> None:
        """Save the generated delay calibration parameters to the folder config file.

        Args:
            filename (str, optional): Filename of the config dictionary to save to.
                Defaults to "sed_config.yaml" in the current folder.
            overwrite (bool, optional): Option to overwrite the present dictionary.
                Defaults to False.
        """
        if filename is None:
            filename = "sed_config.yaml"

        if len(self.dc.calibration) == 0:
            raise ValueError("No delay calibration parameters to save!")
        calibration = {}
        for key, value in self.dc.calibration.items():
            if key == "datafile":
                calibration[key] = value
            elif key in ["adc_range", "delay_range", "delay_range_mm"]:
                calibration[key] = [float(i) for i in value]
            else:
                calibration[key] = float(value)

        if "creation_date" not in calibration:
            calibration["creation_date"] = datetime.now().timestamp()

        config = {
            "delay": {
                "calibration": calibration,
            },
        }
        save_config(config, filename, overwrite)

    def add_delay_offset(
        self,
        constant: float = None,
        flip_delay_axis: bool = None,
        columns: Union[str, Sequence[str]] = None,
        weights: Union[float, Sequence[float]] = 1.0,
        reductions: Union[str, Sequence[str]] = None,
        preserve_mean: Union[bool, Sequence[bool]] = False,
        preview: bool = False,
        verbose: bool = None,
    ) -> None:
        """Shift the delay axis of the dataframe by a constant or other columns.

        Args:
            constant (float, optional): The constant to shift the delay axis by.
            flip_delay_axis (bool, optional): Option to reverse the direction of the delay axis.
            columns (Union[str, Sequence[str]]): Name of the column(s) to apply the shift from.
            weights (Union[float, Sequence[float]]): weights to apply to the columns.
                Can also be used to flip the sign (e.g. -1). Defaults to 1.
            preserve_mean (bool): Whether to subtract the mean of the column before applying the
                shift. Defaults to False.
            reductions (str): The reduction to apply to the column. Should be an available method
                of dask.dataframe.Series. For example "mean". In this case the function is applied
                to the column to generate a single value for the whole dataset. If None, the shift
                is applied per-dataframe-row. Defaults to None. Currently only "mean" is supported.
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"].

        Raises:
            ValueError: If the delay column is not in the dataframe.
        """
        if verbose is None:
            verbose = self.verbose

        delay_column = self._config["dataframe"]["delay_column"]
        if delay_column not in self._dataframe.columns:
            raise ValueError(f"Delay column {delay_column} not found in dataframe! ")

        if self.dataframe is not None:
            if verbose:
                print("Adding delay offset to dataframe:")
            df, metadata = self.dc.add_offsets(
                df=self._dataframe,
                constant=constant,
                flip_delay_axis=flip_delay_axis,
                columns=columns,
                delay_column=delay_column,
                weights=weights,
                reductions=reductions,
                preserve_mean=preserve_mean,
                verbose=verbose,
            )
            if self._timed_dataframe is not None and delay_column in self._timed_dataframe.columns:
                tdf, _ = self.dc.add_offsets(
                    df=self._timed_dataframe,
                    constant=constant,
                    flip_delay_axis=flip_delay_axis,
                    columns=columns,
                    delay_column=delay_column,
                    weights=weights,
                    reductions=reductions,
                    preserve_mean=preserve_mean,
                    verbose=False,
                )

            self._attributes.add(
                metadata,
                "delay_offset",
                duplicate_policy="append",
            )
            self._dataframe = df
            if self._timed_dataframe is not None and delay_column in self._timed_dataframe.columns:
                self._timed_dataframe = tdf
        else:
            raise ValueError("No dataframe loaded!")
        if preview:
            print(self._dataframe.head(10))
        else:
            if verbose:
                print(self._dataframe)

    def save_delay_offsets(
        self,
        filename: str = None,
        overwrite: bool = False,
    ) -> None:
        """Save the generated delay calibration parameters to the folder config file.

        Args:
            filename (str, optional): Filename of the config dictionary to save to.
                Defaults to "sed_config.yaml" in the current folder.
            overwrite (bool, optional): Option to overwrite the present dictionary.
                Defaults to False.
        """
        if filename is None:
            filename = "sed_config.yaml"
        if len(self.dc.offsets) == 0:
            raise ValueError("No delay offset parameters to save!")

        if "creation_date" not in self.ec.offsets.keys():
            self.ec.offsets["creation_date"] = datetime.now().timestamp()

        config = {
            "delay": {
                "offsets": self.dc.offsets,
            },
        }
        save_config(config, filename, overwrite)
        print(f'Saved delay offset parameters to "{filename}".')

    def save_workflow_params(
        self,
        filename: str = None,
        overwrite: bool = False,
    ) -> None:
        """run all save calibration parameter methods

        Args:
            filename (str, optional): Filename of the config dictionary to save to.
                Defaults to "sed_config.yaml" in the current folder.
            overwrite (bool, optional): Option to overwrite the present dictionary.
                Defaults to False.
        """
        for method in [
            self.save_splinewarp,
            self.save_transformations,
            self.save_momentum_calibration,
            self.save_energy_correction,
            self.save_energy_calibration,
            self.save_energy_offset,
            self.save_delay_calibration,
            self.save_delay_offsets,
        ]:
            try:
                method(filename, overwrite)
            except (ValueError, AttributeError, KeyError):
                pass

    def add_jitter(
        self,
        cols: List[str] = None,
        amps: Union[float, Sequence[float]] = None,
        **kwds,
    ):
        """Add jitter to the selected dataframe columns.

        Args:
            cols (List[str], optional): The colums onto which to apply jitter.
                Defaults to config["dataframe"]["jitter_cols"].
            amps (Union[float, Sequence[float]], optional): Amplitude scalings for the
                jittering noise. If one number is given, the same is used for all axes.
                For uniform noise (default) it will cover the interval [-amp, +amp].
                Defaults to config["dataframe"]["jitter_amps"].
            **kwds: additional keyword arguments passed to ``apply_jitter``.
        """
        if cols is None:
            cols = self._config["dataframe"]["jitter_cols"]
        for loc, col in enumerate(cols):
            if col.startswith("@"):
                cols[loc] = self._config["dataframe"].get(col.strip("@"))

        if amps is None:
            amps = self._config["dataframe"]["jitter_amps"]

        self._dataframe = self._dataframe.map_partitions(
            apply_jitter,
            cols=cols,
            cols_jittered=cols,
            amps=amps,
            **kwds,
        )
        if self._timed_dataframe is not None:
            cols_timed = cols.copy()
            for col in cols:
                if col not in self._timed_dataframe.columns:
                    cols_timed.remove(col)

            if cols_timed:
                self._timed_dataframe = self._timed_dataframe.map_partitions(
                    apply_jitter,
                    cols=cols_timed,
                    cols_jittered=cols_timed,
                )
        metadata = []
        for col in cols:
            metadata.append(col)
        # TODO: allow only appending if columns are not jittered yet
        self._attributes.add(metadata, "jittering", duplicate_policy="append")

    def add_time_stamped_data(
        self,
        dest_column: str,
        time_stamps: np.ndarray = None,
        data: np.ndarray = None,
        archiver_channel: str = None,
        **kwds,
    ):
        """Add data in form of timestamp/value pairs to the dataframe using interpolation to the
        timestamps in the dataframe. The time-stamped data can either be provided, or fetched from
        an EPICS archiver instance.

        Args:
            dest_column (str): destination column name
            time_stamps (np.ndarray, optional): Time stamps of the values to add. If omitted,
                time stamps are retrieved from the epics archiver
            data (np.ndarray, optional): Values corresponding at the time stamps in time_stamps.
                If omitted, data are retrieved from the epics archiver.
            archiver_channel (str, optional): EPICS archiver channel from which to retrieve data.
                Either this or data and time_stamps have to be present.
            **kwds: additional keyword arguments passed to ``add_time_stamped_data``.
        """
        time_stamp_column = kwds.pop(
            "time_stamp_column",
            self._config["dataframe"].get("time_stamp_alias", ""),
        )

        if time_stamps is None and data is None:
            if archiver_channel is None:
                raise ValueError(
                    "Either archiver_channel or both time_stamps and data have to be present!",
                )
            if self.loader.__name__ != "mpes":
                raise NotImplementedError(
                    "This function is currently only implemented for the mpes loader!",
                )
            ts_from, ts_to = cast(MpesLoader, self.loader).get_start_and_end_time()
            # get channel data with +-5 seconds safety margin
            time_stamps, data = get_archiver_data(
                archiver_url=self._config["metadata"].get("archiver_url", ""),
                archiver_channel=archiver_channel,
                ts_from=ts_from - 5,
                ts_to=ts_to + 5,
            )

        self._dataframe = add_time_stamped_data(
            self._dataframe,
            time_stamps=time_stamps,
            data=data,
            dest_column=dest_column,
            time_stamp_column=time_stamp_column,
            **kwds,
        )
        if self._timed_dataframe is not None:
            if time_stamp_column in self._timed_dataframe:
                self._timed_dataframe = add_time_stamped_data(
                    self._timed_dataframe,
                    time_stamps=time_stamps,
                    data=data,
                    dest_column=dest_column,
                    time_stamp_column=time_stamp_column,
                    **kwds,
                )
        metadata: List[Any] = []
        metadata.append(dest_column)
        metadata.append(time_stamps)
        metadata.append(data)
        self._attributes.add(metadata, "time_stamped_data", duplicate_policy="append")

    def pre_binning(
        self,
        df_partitions: Union[int, Sequence[int]] = 100,
        axes: List[str] = None,
        bins: List[int] = None,
        ranges: Sequence[Tuple[float, float]] = None,
        **kwds,
    ) -> xr.DataArray:
        """Function to do an initial binning of the dataframe loaded to the class.

        Args:
            df_partitions (Union[int, Sequence[int]], optional): Number of dataframe partitions to
                use for the initial binning. Defaults to 100.
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
            axes = self._config["momentum"]["axes"]
        for loc, axis in enumerate(axes):
            if axis.startswith("@"):
                axes[loc] = self._config["dataframe"].get(axis.strip("@"))

        if bins is None:
            bins = self._config["momentum"]["bins"]
        if ranges is None:
            ranges_ = list(self._config["momentum"]["ranges"])
            ranges_[2] = np.asarray(ranges_[2]) / 2 ** (
                self._config["dataframe"]["tof_binning"] - 1
            )
            ranges = [cast(Tuple[float, float], tuple(v)) for v in ranges_]

        assert self._dataframe is not None, "dataframe needs to be loaded first!"

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
        normalize_to_acquisition_time: Union[bool, str] = False,
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
            normalize_to_acquisition_time (Union[bool, str]): Option to normalize the
                result to the acquistion time. If a "slow" axis was scanned, providing
                the name of the scanned axis will compute and apply the corresponding
                normalization histogram. Defaults to False.
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
                - **df_partitions**: A sequence of dataframe partitions, or the
                  number of the dataframe partitions to use. Defaults to all partitions.
                - **filter**: A Sequence of Dictionaries with entries "col", "lower_bound",
                  "upper_bound" to apply as filter to the dataframe before binning. The
                  dataframe in the class remains unmodified by this.

                Additional kwds are passed to ``bin_dataframe``.

        Raises:
            AssertError: Rises when no dataframe has been loaded.

        Returns:
            xr.DataArray: The result of the n-dimensional binning represented in an
            xarray object, combining the data with the axes.
        """
        assert self._dataframe is not None, "dataframe needs to be loaded first!"

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
        df_partitions: Union[int, Sequence[int]] = kwds.pop("df_partitions", None)
        if isinstance(df_partitions, int):
            df_partitions = list(range(0, min(df_partitions, self._dataframe.npartitions)))
        if df_partitions is not None:
            dataframe = self._dataframe.partitions[df_partitions]
        else:
            dataframe = self._dataframe

        filter_params = kwds.pop("filter", None)
        if filter_params is not None:
            try:
                for param in filter_params:
                    if "col" not in param:
                        raise ValueError(
                            "'col' needs to be defined for each filter entry! ",
                            f"Not present in {param}.",
                        )
                    assert set(param.keys()).issubset({"col", "lower_bound", "upper_bound"})
                    dataframe = apply_filter(dataframe, **param)
            except AssertionError as exc:
                invalid_keys = set(param.keys()) - {"lower_bound", "upper_bound"}
                raise ValueError(
                    "Only 'col', 'lower_bound' and 'upper_bound' allowed as filter entries. ",
                    f"Parameters {invalid_keys} not valid in {param}.",
                ) from exc

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
                self._binned[dim].attrs["unit"] = self._config["dataframe"]["units"][dim]
            except KeyError:
                pass

        self._binned.attrs["units"] = "counts"
        self._binned.attrs["long_name"] = "photoelectron counts"
        self._binned.attrs["metadata"] = self._attributes.metadata

        if normalize_to_acquisition_time:
            if isinstance(normalize_to_acquisition_time, str):
                axis = normalize_to_acquisition_time
                print(
                    f"Calculate normalization histogram for axis '{axis}'...",
                )
                self._normalization_histogram = self.get_normalization_histogram(
                    axis=axis,
                    df_partitions=df_partitions,
                )
                # if the axes are named correctly, xarray figures out the normalization correctly
                self._normalized = self._binned / self._normalization_histogram
                self._attributes.add(
                    self._normalization_histogram.values,
                    name="normalization_histogram",
                    duplicate_policy="overwrite",
                )
            else:
                acquisition_time = self.loader.get_elapsed_time(
                    fids=df_partitions,
                )
                if acquisition_time > 0:
                    self._normalized = self._binned / acquisition_time
                self._attributes.add(
                    acquisition_time,
                    name="normalization_histogram",
                    duplicate_policy="overwrite",
                )

            self._normalized.attrs["units"] = "counts/second"
            self._normalized.attrs["long_name"] = "photoelectron counts per second"
            self._normalized.attrs["metadata"] = self._attributes.metadata

            return self._normalized

        return self._binned

    def get_normalization_histogram(
        self,
        axis: str = "delay",
        use_time_stamps: bool = False,
        **kwds,
    ) -> xr.DataArray:
        """Generates a normalization histogram from the timed dataframe. Optionally,
        use the TimeStamps column instead.

        Args:
            axis (str, optional): The axis for which to compute histogram.
                Defaults to "delay".
            use_time_stamps (bool, optional): Use the TimeStamps column of the
                dataframe, rather than the timed dataframe. Defaults to False.
            **kwds: Keyword arguments:

                - **df_partitions**: A sequence of dataframe partitions, or the
                  number of the dataframe partitions to use. Defaults to all partitions.

        Raises:
            ValueError: Raised if no data are binned.
            ValueError: Raised if 'axis' not in binned coordinates.
            ValueError: Raised if config["dataframe"]["time_stamp_alias"] not found
                in Dataframe.

        Returns:
            xr.DataArray: The computed normalization histogram (in TimeStamp units
            per bin).
        """

        if self._binned is None:
            raise ValueError("Need to bin data first!")
        if axis not in self._binned.coords:
            raise ValueError(f"Axis '{axis}' not found in binned data!")

        df_partitions: Union[int, Sequence[int]] = kwds.pop("df_partitions", None)
        if isinstance(df_partitions, int):
            df_partitions = list(range(0, min(df_partitions, self._dataframe.npartitions)))
        if use_time_stamps or self._timed_dataframe is None:
            if df_partitions is not None:
                self._normalization_histogram = normalization_histogram_from_timestamps(
                    self._dataframe.partitions[df_partitions],
                    axis,
                    self._binned.coords[axis].values,
                    self._config["dataframe"]["time_stamp_alias"],
                )
            else:
                self._normalization_histogram = normalization_histogram_from_timestamps(
                    self._dataframe,
                    axis,
                    self._binned.coords[axis].values,
                    self._config["dataframe"]["time_stamp_alias"],
                )
        else:
            if df_partitions is not None:
                self._normalization_histogram = normalization_histogram_from_timed_dataframe(
                    self._timed_dataframe.partitions[df_partitions],
                    axis,
                    self._binned.coords[axis].values,
                    self._config["dataframe"]["timed_dataframe_unit_time"],
                )
            else:
                self._normalization_histogram = normalization_histogram_from_timed_dataframe(
                    self._timed_dataframe,
                    axis,
                    self._binned.coords[axis].values,
                    self._config["dataframe"]["timed_dataframe_unit_time"],
                )

        return self._normalization_histogram

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
        axes = list(axes)
        for loc, axis in enumerate(axes):
            if axis.startswith("@"):
                axes[loc] = self._config["dataframe"].get(axis.strip("@"))
        if ranges is None:
            ranges = list(self._config["histogram"]["ranges"])
            for loc, axis in enumerate(axes):
                if axis == self._config["dataframe"]["tof_column"]:
                    ranges[loc] = np.asarray(ranges[loc]) / 2 ** (
                        self._config["dataframe"]["tof_binning"] - 1
                    )
                elif axis == self._config["dataframe"]["adc_column"]:
                    ranges[loc] = np.asarray(ranges[loc]) / 2 ** (
                        self._config["dataframe"]["adc_binning"] - 1
                    )

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
                - **eln_data**: An electronic-lab-notebook file in '.yaml' format
                  to add to the list of files to pass to the reader.
        """
        if self._binned is None:
            raise NameError("Need to bin data first!")

        if self._normalized is not None:
            data = self._normalized
        else:
            data = self._binned

        extension = pathlib.Path(faddr).suffix

        if extension in (".tif", ".tiff"):
            to_tiff(
                data=data,
                faddr=faddr,
                **kwds,
            )
        elif extension in (".h5", ".hdf5"):
            to_h5(
                data=data,
                faddr=faddr,
                **kwds,
            )
        elif extension in (".nxs", ".nexus"):
            try:
                reader = kwds.pop("reader", self._config["nexus"]["reader"])
                definition = kwds.pop(
                    "definition",
                    self._config["nexus"]["definition"],
                )
                input_files = kwds.pop(
                    "input_files",
                    self._config["nexus"]["input_files"],
                )
            except KeyError as exc:
                raise ValueError(
                    "The nexus reader, definition and input files need to be provide!",
                ) from exc

            if isinstance(input_files, str):
                input_files = [input_files]

            if "eln_data" in kwds:
                input_files.append(kwds.pop("eln_data"))

            to_nexus(
                data=data,
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
