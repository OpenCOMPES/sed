"""This module contains the core class for the sed package

"""
from __future__ import annotations

import pathlib
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime
from typing import Any
from typing import cast

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
from sed.core.logging import call_logger
from sed.core.logging import set_verbosity
from sed.core.logging import setup_logging
from sed.core.metadata import MetaHandler
from sed.diagnostics import grid_histogram
from sed.io import to_h5
from sed.io import to_nexus
from sed.io import to_tiff
from sed.loader import CopyTool
from sed.loader import get_loader
from sed.loader.mpes.loader import MpesLoader
from sed.loader.mpes.metadata import get_archiver_data

N_CPU = psutil.cpu_count()

# Configure logging
logger = setup_logging("processor")


class SedProcessor:
    """Processor class of sed. Contains wrapper functions defining a work flow for data
    correction, calibration and binning.

    Args:
        metadata (dict, optional): Dict of external Metadata. Defaults to None.
        config (dict | str, optional): Config dictionary or config file name.
            Defaults to None.
        dataframe (pd.DataFrame | ddf.DataFrame, optional): dataframe to load
            into the class. Defaults to None.
        files (list[str], optional): List of files to pass to the loader defined in
            the config. Defaults to None.
        folder (str, optional): Folder containing files to pass to the loader
            defined in the config. Defaults to None.
        runs (Sequence[str], optional): List of run identifiers to pass to the loader
            defined in the config. Defaults to None.
        collect_metadata (bool): Option to collect metadata from files.
            Defaults to False.
        verbose (bool, optional): Option to print out diagnostic information.
            Defaults to config["core"]["verbose"] or True.
        **kwds: Keyword arguments passed to the reader.
    """

    @call_logger(logger)
    def __init__(
        self,
        metadata: dict = None,
        config: dict | str = None,
        dataframe: pd.DataFrame | ddf.DataFrame = None,
        files: list[str] = None,
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
            config (dict | str, optional): Config dictionary or config file name.
                Defaults to None.
            dataframe (pd.DataFrame | ddf.DataFrame, optional): dataframe to load
                into the class. Defaults to None.
            files (list[str], optional): List of files to pass to the loader defined in
                the config. Defaults to None.
            folder (str, optional): Folder containing files to pass to the loader
                defined in the config. Defaults to None.
            runs (Sequence[str], optional): List of run identifiers to pass to the loader
                defined in the config. Defaults to None.
            collect_metadata (bool, optional): Option to collect metadata from files.
                Defaults to False.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to config["core"]["verbose"] or True.
            **kwds: Keyword arguments passed to parse_config and to the reader.
        """
        # split off config keywords
        config_kwds = {
            key: value for key, value in kwds.items() if key in parse_config.__code__.co_varnames
        }
        for key in config_kwds.keys():
            del kwds[key]
        self._config = parse_config(config, **config_kwds)
        num_cores = self._config["core"].get("num_cores", N_CPU - 1)
        if num_cores >= N_CPU:
            num_cores = N_CPU - 1
        self._config["core"]["num_cores"] = num_cores
        logger.debug(f"Use {num_cores} cores for processing.")

        if verbose is None:
            self._verbose = self._config["core"].get("verbose", True)
        else:
            self._verbose = verbose
        set_verbosity(logger, self._verbose)

        self._dataframe: pd.DataFrame | ddf.DataFrame = None
        self._timed_dataframe: pd.DataFrame | ddf.DataFrame = None
        self._files: list[str] = []

        self._binned: xr.DataArray = None
        self._pre_binned: xr.DataArray = None
        self._normalization_histogram: xr.DataArray = None
        self._normalized: xr.DataArray = None

        self._attributes = MetaHandler(meta=metadata)

        loader_name = self._config["core"]["loader"]
        self.loader = get_loader(
            loader_name=loader_name,
            config=self._config,
            verbose=verbose,
        )
        logger.debug(f"Use loader: {loader_name}")

        self.ec = EnergyCalibrator(
            loader=get_loader(
                loader_name=loader_name,
                config=self._config,
                verbose=verbose,
            ),
            config=self._config,
            verbose=self._verbose,
        )

        self.mc = MomentumCorrector(
            config=self._config,
            verbose=self._verbose,
        )

        self.dc = DelayCalibrator(
            config=self._config,
            verbose=self._verbose,
        )

        self.use_copy_tool = "copy_tool" in self._config["core"] and self._config["core"][
            "copy_tool"
        ].pop("use", True)
        if self.use_copy_tool:
            try:
                self.ct = CopyTool(
                    num_cores=self._config["core"]["num_cores"],
                    **self._config["core"]["copy_tool"],
                )
                logger.debug(
                    f"Initialized copy tool: Copy files from "
                    f"'{self._config['core']['copy_tool']['source']}' "
                    f"to '{self._config['core']['copy_tool']['dest']}'.",
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
            df_str = "Dataframe: No Data loaded"
        else:
            df_str = self._dataframe.__repr__()
        pretty_str = df_str + "\n" + "Metadata: " + "\n" + self._attributes.__repr__()
        return pretty_str

    def _repr_html_(self):
        html = "<div>"

        if self._dataframe is None:
            df_html = "Dataframe: No Data loaded"
        else:
            df_html = self._dataframe._repr_html_()

        html += f"<details><summary>Dataframe</summary>{df_html}</details>"

        # Add expandable section for attributes
        html += "<details><summary>Metadata</summary>"
        html += "<div style='padding-left: 10px;'>"
        html += self._attributes._repr_html_()
        html += "</div></details>"

        html += "</div>"

        return html

    ## Suggestion:
    # @property
    # def overview_panel(self):
    #     """Provides an overview panel with plots of different data attributes."""
    #     self.view_event_histogram(dfpid=2, backend="matplotlib")

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
        self.mc.verbose = verbose
        self.ec.verbose = verbose
        self.dc.verbose = verbose
        self.loader.verbose = verbose

    @property
    def dataframe(self) -> pd.DataFrame | ddf.DataFrame:
        """Accessor to the underlying dataframe.

        Returns:
            pd.DataFrame | ddf.DataFrame: Dataframe object.
        """
        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame | ddf.DataFrame):
        """Setter for the underlying dataframe.

        Args:
            dataframe (pd.DataFrame | ddf.DataFrame): The dataframe object to set.
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
    def timed_dataframe(self) -> pd.DataFrame | ddf.DataFrame:
        """Accessor to the underlying timed_dataframe.

        Returns:
            pd.DataFrame | ddf.DataFrame: Timed Dataframe object.
        """
        return self._timed_dataframe

    @timed_dataframe.setter
    def timed_dataframe(self, timed_dataframe: pd.DataFrame | ddf.DataFrame):
        """Setter for the underlying timed dataframe.

        Args:
            timed_dataframe (pd.DataFrame | ddf.DataFrame): The timed dataframe object to set
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
    def attributes(self) -> MetaHandler:
        """Accessor to the metadata dict.

        Returns:
            MetaHandler: The metadata object
        """
        return self._attributes

    def add_attribute(self, attributes: dict, name: str, **kwds):
        """Function to add element to the attributes dict.

        Args:
            attributes (dict): The attributes dictionary object to add.
            name (str): Key under which to add the dictionary to the attributes.
            **kwds: Additional keywords are passed to the ``MetaHandler.add()`` function.
        """
        self._attributes.add(
            entry=attributes,
            name=name,
            **kwds,
        )

    @property
    def config(self) -> dict[Any, Any]:
        """Getter attribute for the config dictionary

        Returns:
            dict: The config dictionary.
        """
        return self._config

    @property
    def files(self) -> list[str]:
        """Getter attribute for the list of files

        Returns:
            list[str]: The list of loaded files
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
            xr.DataArray: The normalization histogram
        """
        if self._normalization_histogram is None:
            raise ValueError("No normalization histogram available, generate histogram first!")
        return self._normalization_histogram

    def cpy(self, path: str | list[str]) -> str | list[str]:
        """Function to mirror a list of files or a folder from a network drive to a
        local storage. Returns either the original or the copied path to the given
        path. The option to use this functionality is set by
        config["core"]["use_copy_tool"].

        Args:
            path (str | list[str]): Source path or path list.

        Returns:
            str | list[str]: Source or destination path or path list.
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

    @call_logger(logger)
    def load(
        self,
        dataframe: pd.DataFrame | ddf.DataFrame = None,
        metadata: dict = None,
        files: list[str] = None,
        folder: str = None,
        runs: Sequence[str] = None,
        collect_metadata: bool = False,
        **kwds,
    ):
        """Load tabular data of single events into the dataframe object in the class.

        Args:
            dataframe (pd.DataFrame | ddf.DataFrame, optional): data in tabular
                format. Accepts anything which can be interpreted by pd.DataFrame as
                an input. Defaults to None.
            metadata (dict, optional): Dict of external Metadata. Defaults to None.
            files (list[str], optional): List of file paths to pass to the loader.
                Defaults to None.
            runs (Sequence[str], optional): List of run identifiers to pass to the
                loader. Defaults to None.
            folder (str, optional): Folder path to pass to the loader.
                Defaults to None.
            collect_metadata (bool, optional): Option for collecting metadata in the reader.
            **kwds:
                - *timed_dataframe*: timed dataframe if dataframe is provided.

                Additional keyword parameters are passed to ``loader.read_dataframe()``.

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
                files=cast(list[str], self.cpy(files)),
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

    @call_logger(logger)
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
    @call_logger(logger)
    def bin_and_load_momentum_calibration(
        self,
        df_partitions: int | Sequence[int] = 100,
        axes: list[str] = None,
        bins: list[int] = None,
        ranges: Sequence[tuple[float, float]] = None,
        plane: int = 0,
        width: int = 5,
        apply: bool = False,
        **kwds,
    ):
        """1st step of momentum correction work flow. Function to do an initial binning
        of the dataframe loaded to the class, slice a plane from it using an
        interactive view, and load it into the momentum corrector class.

        Args:
            df_partitions (int | Sequence[int], optional): Number of dataframe partitions
                to use for the initial binning. Defaults to 100.
            axes (list[str], optional): Axes to bin.
                Defaults to config["momentum"]["axes"].
            bins (list[int], optional): Bin numbers to use for binning.
                Defaults to config["momentum"]["bins"].
            ranges (Sequence[tuple[float, float]], optional): Ranges to use for binning.
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
    @call_logger(logger)
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
        provided as list of feature points, or auto-generated using a
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
    @call_logger(logger)
    def generate_splinewarp(
        self,
        use_center: bool = None,
        **kwds,
    ):
        """3. Step of the distortion correction workflow: Generate the correction
        function restoring the symmetry in the image using a splinewarp algorithm.

        Args:
            use_center (bool, optional): Option to use the position of the
                center point in the correction. Default is read from config, or set to True.
            **kwds: Keyword arguments for MomentumCorrector.spline_warp_estimate().
        """

        self.mc.spline_warp_estimate(use_center=use_center, **kwds)

        if self.mc.slice is not None and self._verbose:
            self.mc.view(
                annotated=True,
                backend="matplotlib",
                crosshair=True,
                title="Original slice with reference features",
            )

            self.mc.view(
                image=self.mc.slice_corrected,
                annotated=True,
                points={"feats": self.mc.ptargs},
                backend="matplotlib",
                crosshair=True,
                title="Corrected slice with target features",
            )

            self.mc.view(
                image=self.mc.slice,
                points={"feats": self.mc.ptargs},
                annotated=True,
                backend="matplotlib",
                title="Original slice with target features",
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
            elif key == "creation_date":
                correction[key] = value.isoformat()
            else:
                correction[key] = float(value)

        if "creation_date" not in correction:
            correction["creation_date"] = datetime.now().isoformat()

        config = {
            "momentum": {
                "correction": correction,
            },
        }
        save_config(config, filename, overwrite)
        logger.info(f'Saved momentum correction parameters to "{filename}".')

    # 4. Pose corrections. Provide interactive interface for correcting
    # scaling, shift and rotation
    @call_logger(logger)
    def pose_adjustment(
        self,
        transformations: dict[str, Any] = None,
        apply: bool = False,
        use_correction: bool = True,
        reset: bool = True,
        **kwds,
    ):
        """3. step of the distortion correction workflow: Generate an interactive panel
        to adjust affine transformations that are applied to the image. Applies first
        a scaling, next an x/y translation, and last a rotation around the center of
        the image.

        Args:
            transformations (dict[str, Any], optional): Dictionary with transformations.
                Defaults to self.transformations or config["momentum"]["transformations"].
            apply (bool, optional): Option to directly apply the provided
                transformations. Defaults to False.
            use_correction (bool, option): Whether to use the spline warp correction
                or not. Defaults to True.
            reset (bool, optional): Option to reset the correction before transformation.
                Defaults to True.
            **kwds: Keyword parameters defining defaults for the transformations:

                - **scale** (float): Initial value of the scaling slider.
                - **xtrans** (float): Initial value of the xtrans slider.
                - **ytrans** (float): Initial value of the ytrans slider.
                - **angle** (float): Initial value of the angle slider.
        """
        # Generate homography as default if no distortion correction has been applied
        if self.mc.slice_corrected is None:
            if self.mc.slice is None:
                self.mc.slice = np.zeros(self._config["momentum"]["bins"][0:2])
                self.mc.bin_ranges = self._config["momentum"]["ranges"]
            self.mc.slice_corrected = self.mc.slice

        if not use_correction:
            self.mc.reset_deformation()
            reset = False

        if self.mc.cdeform_field is None or self.mc.rdeform_field is None:
            # Generate distortion correction from config values
            self.mc.spline_warp_estimate()

        self.mc.pose_adjustment(
            transformations=transformations,
            apply=apply,
            reset=reset,
            **kwds,
        )

    # 4a. Save pose adjustment parameters to config file.
    @call_logger(logger)
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
            if key == "creation_date":
                transformations[key] = value.isoformat()
            else:
                transformations[key] = float(value)

        if "creation_date" not in transformations:
            transformations["creation_date"] = datetime.now().isoformat()

        config = {
            "momentum": {
                "transformations": transformations,
            },
        }
        save_config(config, filename, overwrite)
        logger.info(f'Saved momentum transformation parameters to "{filename}".')

    # 5. Apply the momentum correction to the dataframe
    @call_logger(logger)
    def apply_momentum_correction(
        self,
        preview: bool = False,
        **kwds,
    ):
        """Applies the distortion correction and pose adjustment (optional)
        to the dataframe.

        Args:
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            **kwds: Keyword parameters for ``MomentumCorrector.apply_correction``:

                - **rdeform_field** (np.ndarray, optional): Row deformation field.
                - **cdeform_field** (np.ndarray, optional): Column deformation field.
                - **inv_dfield** (np.ndarray, optional): Inverse deformation field.

        """
        x_column = self._config["dataframe"]["columns"]["x"]
        y_column = self._config["dataframe"]["columns"]["y"]

        if self._dataframe is not None:
            logger.info("Adding corrected X/Y columns to dataframe:")
            df, metadata = self.mc.apply_corrections(
                df=self._dataframe,
                **kwds,
            )
            if (
                self._timed_dataframe is not None
                and x_column in self._timed_dataframe.columns
                and y_column in self._timed_dataframe.columns
            ):
                tdf, _ = self.mc.apply_corrections(
                    self._timed_dataframe,
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
            logger.info(self._dataframe.head(10))
        else:
            logger.info(self._dataframe)

    # Momentum calibration work flow
    # 1. Calculate momentum calibration
    @call_logger(logger)
    def calibrate_momentum_axes(
        self,
        point_a: np.ndarray | list[int] = None,
        point_b: np.ndarray | list[int] = None,
        k_distance: float = None,
        k_coord_a: np.ndarray | list[float] = None,
        k_coord_b: np.ndarray | list[float] = np.array([0.0, 0.0]),
        equiscale: bool = True,
        apply=False,
    ):
        """1. step of the momentum calibration workflow. Calibrate momentum
        axes using either provided pixel coordinates of a high-symmetry point and its
        distance to the BZ center, or the k-coordinates of two points in the BZ
        (depending on the equiscale option). Opens an interactive panel for selecting
        the points.

        Args:
            point_a (np.ndarray | list[int], optional): Pixel coordinates of the first
                point used for momentum calibration.
            point_b (np.ndarray | list[int], optional): Pixel coordinates of the
                second point used for momentum calibration.
                Defaults to config["momentum"]["center_pixel"].
            k_distance (float, optional): Momentum distance between point a and b.
                Needs to be provided if no specific k-coordinates for the two points
                are given. Defaults to None.
            k_coord_a (np.ndarray | list[float], optional): Momentum coordinate
                of the first point used for calibration. Used if equiscale is False.
                Defaults to None.
            k_coord_b (np.ndarray | list[float], optional): Momentum coordinate
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
            elif key == "creation_date":
                calibration[key] = value.isoformat()
            else:
                calibration[key] = float(value)

        if "creation_date" not in calibration:
            calibration["creation_date"] = datetime.now().isoformat()

        config = {"momentum": {"calibration": calibration}}
        save_config(config, filename, overwrite)
        logger.info(f"Saved momentum calibration parameters to {filename}")

    # 2. Apply correction and calibration to the dataframe
    @call_logger(logger)
    def apply_momentum_calibration(
        self,
        calibration: dict = None,
        preview: bool = False,
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
            **kwds: Keyword args passed to ``MomentumCorrector.append_k_axis``.
        """
        x_column = self._config["dataframe"]["columns"]["x"]
        y_column = self._config["dataframe"]["columns"]["y"]

        if self._dataframe is not None:
            logger.info("Adding kx/ky columns to dataframe:")
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
                    suppress_output=True,
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
            logger.info(self._dataframe.head(10))
        else:
            logger.info(self._dataframe)

    # Energy correction workflow
    # 1. Adjust the energy correction parameters
    @call_logger(logger)
    def adjust_energy_correction(
        self,
        correction_type: str = None,
        amplitude: float = None,
        center: tuple[float, float] = None,
        apply=False,
        **kwds,
    ):
        """1. step of the energy correction workflow: Opens an interactive plot to
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
            center (tuple[float, float], optional): Center X/Y coordinates for the
                correction. Defaults to config["energy"]["correction"]["center"].
            apply (bool, optional): Option to directly apply the provided or default
                correction parameters. Defaults to False.
            **kwds: Keyword parameters passed to ``EnergyCalibrator.adjust_energy_correction()``.
        """
        if self._pre_binned is None:
            logger.warn("Pre-binned data not present, binning using defaults from config...")
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
        for key, value in self.ec.correction.items():
            if key == "correction_type":
                correction[key] = value
            elif key == "center":
                correction[key] = [float(i) for i in value]
            elif key == "creation_date":
                correction[key] = value.isoformat()
            else:
                correction[key] = float(value)

        if "creation_date" not in correction:
            correction["creation_date"] = datetime.now().isoformat()

        config = {"energy": {"correction": correction}}
        save_config(config, filename, overwrite)
        logger.info(f"Saved energy correction parameters to {filename}")

    # 2. Apply energy correction to dataframe
    @call_logger(logger)
    def apply_energy_correction(
        self,
        correction: dict = None,
        preview: bool = False,
        **kwds,
    ):
        """2. step of the energy correction workflow: Apply the energy correction
        parameters stored in the class to the dataframe.

        Args:
            correction (dict, optional): Dictionary containing the correction
                parameters. Defaults to config["energy"]["calibration"].
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            **kwds:
                Keyword args passed to ``EnergyCalibrator.apply_energy_correction()``.
        """
        tof_column = self._config["dataframe"]["columns"]["tof"]

        if self._dataframe is not None:
            logger.info("Applying energy correction to dataframe...")
            df, metadata = self.ec.apply_energy_correction(
                df=self._dataframe,
                correction=correction,
                **kwds,
            )
            if self._timed_dataframe is not None and tof_column in self._timed_dataframe.columns:
                tdf, _ = self.ec.apply_energy_correction(
                    df=self._timed_dataframe,
                    correction=correction,
                    suppress_output=True,
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
            logger.info(self._dataframe.head(10))
        else:
            logger.info(self._dataframe)

    # Energy calibrator workflow
    # 1. Load and normalize data
    @call_logger(logger)
    def load_bias_series(
        self,
        binned_data: xr.DataArray | tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        data_files: list[str] = None,
        axes: list[str] = None,
        bins: list = None,
        ranges: Sequence[tuple[float, float]] = None,
        biases: np.ndarray = None,
        bias_key: str = None,
        normalize: bool = None,
        span: int = None,
        order: int = None,
    ):
        """1. step of the energy calibration workflow: Load and bin data from
        single-event files, or load binned bias/TOF traces.

        Args:
            binned_data (xr.DataArray | tuple[np.ndarray, np.ndarray, np.ndarray], optional):
                Binned data If provided as DataArray, Needs to contain dimensions
                config["dataframe"]["columns"]["tof"] and config["dataframe"]["columns"]["bias"].
                If provided as tuple, needs to contain elements tof, biases, traces.
            data_files (list[str], optional): list of file paths to bin
            axes (list[str], optional): bin axes.
                Defaults to config["dataframe"]["columns"]["tof"].
            bins (list, optional): number of bins.
                Defaults to config["energy"]["bins"].
            ranges (Sequence[tuple[float, float]], optional): bin ranges.
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
                    self._config["dataframe"]["columns"]["tof"] not in binned_data.dims
                    or self._config["dataframe"]["columns"]["bias"] not in binned_data.dims
                ):
                    raise ValueError(
                        "If binned_data is provided as an xarray, it needs to contain dimensions "
                        f"'{self._config['dataframe']['columns']['tof']}' and "
                        f"'{self._config['dataframe']['columns']['bias']}'!.",
                    )
                tof = binned_data.coords[self._config["dataframe"]["columns"]["tof"]].values
                biases = binned_data.coords[self._config["dataframe"]["columns"]["bias"]].values
                traces = binned_data.values[:, :]
            else:
                try:
                    (tof, biases, traces) = binned_data
                except ValueError as exc:
                    raise ValueError(
                        "If binned_data is provided as tuple, it needs to contain "
                        "(tof, biases, traces)!",
                    ) from exc
            logger.debug(f"Energy calibration data loaded from binned data. Bias values={biases}.")
            self.ec.load_data(biases=biases, traces=traces, tof=tof)

        elif data_files is not None:
            self.ec.bin_data(
                data_files=cast(list[str], self.cpy(data_files)),
                axes=axes,
                bins=bins,
                ranges=ranges,
                biases=biases,
                bias_key=bias_key,
            )
            logger.debug(
                f"Energy calibration data binned from files {data_files} data. "
                f"Bias values={biases}.",
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
            backend="matplotlib",
        )
        plt.xlabel("Time-of-flight")
        plt.ylabel("Intensity")
        plt.tight_layout()

    # 2. extract ranges and get peak positions
    @call_logger(logger)
    def find_bias_peaks(
        self,
        ranges: list[tuple] | tuple,
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
            ranges (list[tuple] | tuple): Tuple of TOF values indicating a range.
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
                algorithm. amount of points that have to have to behave monotonously
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
            logger.info(f"Use feature ranges: {self.ec.featranges}.")
            try:
                self.ec.feature_extract(peak_window=peak_window)
                logger.info(f"Extracted energy features: {self.ec.peaks}.")
                self.ec.view(
                    traces=self.ec.traces_normed,
                    segs=self.ec.featranges,
                    xaxis=self.ec.tof,
                    peaks=self.ec.peaks,
                    backend="matplotlib",
                )
            except IndexError:
                logger.error("Could not determine all peaks!")
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
    @call_logger(logger)
    def calibrate_energy_axis(
        self,
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
            ref_energy (float): Binding/kinetic energy of the detected feature.
            method (str, optional): Method for determining the energy calibration.

                - **'lmfit'**: Energy calibration using lmfit and 1/t^2 form.
                - **'lstsq'**, **'lsqr'**: Energy calibration using polynomial form.

                Defaults to config["energy"]["calibration_method"]
            energy_scale (str, optional): Direction of increasing energy scale.

                - **'kinetic'**: increasing energy with decreasing TOF.
                - **'binding'**: increasing energy with increasing TOF.

                Defaults to config["energy"]["energy_scale"]
            **kwds**: Keyword parameters passed to ``EnergyCalibrator.calibrate()``.
        """
        if method is None:
            method = self._config["energy"]["calibration_method"]

        if energy_scale is None:
            energy_scale = self._config["energy"]["energy_scale"]

        self.ec.calibrate(
            ref_energy=ref_energy,
            method=method,
            energy_scale=energy_scale,
            **kwds,
        )
        if self._verbose:
            if self.ec.traces_normed is not None:
                self.ec.view(
                    traces=self.ec.traces_normed,
                    xaxis=self.ec.calibration["axis"],
                    align=True,
                    energy_scale=energy_scale,
                    backend="matplotlib",
                    title="Quality of Calibration",
                )
                plt.xlabel("Energy (eV)")
                plt.ylabel("Intensity")
                plt.tight_layout()
                plt.show()
            if energy_scale == "kinetic":
                self.ec.view(
                    traces=self.ec.calibration["axis"][None, :] + self.ec.biases[0],
                    xaxis=self.ec.tof,
                    backend="matplotlib",
                    show_legend=False,
                    title="E/TOF relationship",
                )
                plt.scatter(
                    self.ec.peaks[:, 0],
                    -(self.ec.biases - self.ec.biases[0]) + ref_energy,
                    s=50,
                    c="k",
                )
                plt.tight_layout()
            elif energy_scale == "binding":
                self.ec.view(
                    traces=self.ec.calibration["axis"][None, :] - self.ec.biases[0],
                    xaxis=self.ec.tof,
                    backend="matplotlib",
                    show_legend=False,
                    title="E/TOF relationship",
                )
                plt.scatter(
                    self.ec.peaks[:, 0],
                    self.ec.biases - self.ec.biases[0] + ref_energy,
                    s=50,
                    c="k",
                )
            else:
                raise ValueError(
                    'energy_scale needs to be either "binding" or "kinetic"',
                    f", got {energy_scale}.",
                )
            plt.xlabel("Time-of-flight")
            plt.ylabel("Energy (eV)")
            plt.tight_layout()
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
            elif key == "creation_date":
                calibration[key] = value.isoformat()
            else:
                calibration[key] = float(value)

        if "creation_date" not in calibration:
            calibration["creation_date"] = datetime.now().isoformat()

        config = {"energy": {"calibration": calibration}}
        save_config(config, filename, overwrite)
        logger.info(f'Saved energy calibration parameters to "{filename}".')

    # 4. Apply energy calibration to the dataframe
    @call_logger(logger)
    def append_energy_axis(
        self,
        calibration: dict = None,
        bias_voltage: float = None,
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
            bias_voltage (float, optional): Sample bias voltage of the scan data. If omitted,
                the bias voltage is being read from the dataframe. If it is not found there,
                a warning is printed and the calibrated data might have an offset.
            preview (bool): Option to preview the first elements of the data frame.
            **kwds:
                Keyword args passed to ``EnergyCalibrator.append_energy_axis()``.
        """
        tof_column = self._config["dataframe"]["columns"]["tof"]

        if self._dataframe is not None:
            logger.info("Adding energy column to dataframe:")
            df, metadata = self.ec.append_energy_axis(
                df=self._dataframe,
                calibration=calibration,
                bias_voltage=bias_voltage,
                **kwds,
            )
            if self._timed_dataframe is not None and tof_column in self._timed_dataframe.columns:
                tdf, _ = self.ec.append_energy_axis(
                    df=self._timed_dataframe,
                    calibration=calibration,
                    bias_voltage=bias_voltage,
                    suppress_output=True,
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
            logger.info(self._dataframe.head(10))
        else:
            logger.info(self._dataframe)

    @call_logger(logger)
    def add_energy_offset(
        self,
        constant: float = None,
        columns: str | Sequence[str] = None,
        weights: float | Sequence[float] = None,
        reductions: str | Sequence[str] = None,
        preserve_mean: bool | Sequence[bool] = None,
        preview: bool = False,
    ) -> None:
        """Shift the energy axis of the dataframe by a given amount.

        Args:
            constant (float, optional): The constant to shift the energy axis by.
            columns (str | Sequence[str], optional): Name of the column(s) to apply the shift from.
            weights (float | Sequence[float], optional): weights to apply to the columns.
                Can also be used to flip the sign (e.g. -1). Defaults to 1.
            reductions (str | Sequence[str], optional): The reduction to apply to the column.
                Should be an available method of dask.dataframe.Series. For example "mean". In this
                case the function is applied to the column to generate a single value for the whole
                dataset. If None, the shift is applied per-dataframe-row. Defaults to None.
                Currently only "mean" is supported.
            preserve_mean (bool | Sequence[bool], optional): Whether to subtract the mean of the
                column before applying the shift. Defaults to False.
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.

        Raises:
            ValueError: If the energy column is not in the dataframe.
        """
        energy_column = self._config["dataframe"]["columns"]["energy"]
        if energy_column not in self._dataframe.columns:
            raise ValueError(
                f"Energy column {energy_column} not found in dataframe! "
                "Run `append_energy_axis()` first.",
            )
        if self.dataframe is not None:
            logger.info("Adding energy offset to dataframe:")
            df, metadata = self.ec.add_offsets(
                df=self._dataframe,
                constant=constant,
                columns=columns,
                energy_column=energy_column,
                weights=weights,
                reductions=reductions,
                preserve_mean=preserve_mean,
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
                    suppress_output=True,
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
            logger.info(self._dataframe.head(10))
        else:
            logger.info(self._dataframe)

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

        offsets = deepcopy(self.ec.offsets)

        if "creation_date" not in offsets.keys():
            offsets["creation_date"] = datetime.now()

        offsets["creation_date"] = offsets["creation_date"].isoformat()

        config = {"energy": {"offsets": offsets}}
        save_config(config, filename, overwrite)
        logger.info(f'Saved energy offset parameters to "{filename}".')

    @call_logger(logger)
    def append_tof_ns_axis(
        self,
        preview: bool = False,
        **kwds,
    ):
        """Convert time-of-flight channel steps to nanoseconds.

        Args:
            tof_ns_column (str, optional): Name of the generated column containing the
                time-of-flight in nanosecond.
                Defaults to config["dataframe"]["columns"]["tof_ns"].
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            **kwds: additional arguments are passed to ``EnergyCalibrator.append_tof_ns_axis()``.

        """
        tof_column = self._config["dataframe"]["columns"]["tof"]

        if self._dataframe is not None:
            logger.info("Adding time-of-flight column in nanoseconds to dataframe.")
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
            logger.info(self._dataframe.head(10))
        else:
            logger.info(self._dataframe)

    @call_logger(logger)
    def align_dld_sectors(
        self,
        sector_delays: np.ndarray = None,
        preview: bool = False,
        **kwds,
    ):
        """Align the 8s sectors of the HEXTOF endstation.

        Args:
            sector_delays (np.ndarray, optional): Array containing the sector delays. Defaults to
                config["dataframe"]["sector_delays"].
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            **kwds: additional arguments are passed to ``EnergyCalibrator.align_dld_sectors()``.
        """
        tof_column = self._config["dataframe"]["columns"]["tof"]

        if self._dataframe is not None:
            logger.info("Aligning 8s sectors of dataframe")
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
            logger.info(self._dataframe.head(10))
        else:
            logger.info(self._dataframe)

    # Delay calibration function
    @call_logger(logger)
    def calibrate_delay_axis(
        self,
        delay_range: tuple[float, float] = None,
        read_delay_ranges: bool = True,
        datafile: str = None,
        preview: bool = False,
        **kwds,
    ):
        """Append delay column to dataframe. Either provide delay ranges, or read
        them from a file.

        Args:
            delay_range (tuple[float, float], optional): The scanned delay range in
                picoseconds. Defaults to None.
            read_delay_ranges (bool, optional): Option whether to read the delay ranges from the
                data. Defaults to True. If false, parameters in the config will be used.
            datafile (str, optional): The file from which to read the delay ranges.
                Defaults to the first file of the dataset.
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.
            **kwds: Keyword args passed to ``DelayCalibrator.append_delay_axis``.
        """
        adc_column = self._config["dataframe"]["columns"]["adc"]
        if adc_column not in self._dataframe.columns:
            raise ValueError(f"ADC column {adc_column} not found in dataframe, cannot calibrate!")

        if self._dataframe is not None:
            logger.info("Adding delay column to dataframe:")

            if read_delay_ranges and delay_range is None and datafile is None:
                try:
                    datafile = self._files[0]
                except IndexError as exc:
                    raise ValueError(
                        "No datafile available, specify either 'datafile' or 'delay_range'.",
                    ) from exc

            df, metadata = self.dc.append_delay_axis(
                self._dataframe,
                delay_range=delay_range,
                datafile=datafile,
                **kwds,
            )
            if self._timed_dataframe is not None and adc_column in self._timed_dataframe.columns:
                tdf, _ = self.dc.append_delay_axis(
                    self._timed_dataframe,
                    delay_range=delay_range,
                    datafile=datafile,
                    suppress_output=True,
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
            logger.info(self._dataframe.head(10))
        else:
            logger.debug(self._dataframe)

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
            elif key == "creation_date":
                calibration[key] = value.isoformat()
            else:
                calibration[key] = float(value)

        if "creation_date" not in calibration:
            calibration["creation_date"] = datetime.now().isoformat()

        config = {
            "delay": {
                "calibration": calibration,
            },
        }
        save_config(config, filename, overwrite)

    @call_logger(logger)
    def add_delay_offset(
        self,
        constant: float = None,
        flip_delay_axis: bool = None,
        columns: str | Sequence[str] = None,
        weights: float | Sequence[float] = 1.0,
        reductions: str | Sequence[str] = None,
        preserve_mean: bool | Sequence[bool] = False,
        preview: bool = False,
    ) -> None:
        """Shift the delay axis of the dataframe by a constant or other columns.

        Args:
            constant (float, optional): The constant to shift the delay axis by.
            flip_delay_axis (bool, optional): Option to reverse the direction of the delay axis.
            columns (str | Sequence[str], optional): Name of the column(s) to apply the shift from.
            weights (float | Sequence[float], optional): weights to apply to the columns.
                Can also be used to flip the sign (e.g. -1). Defaults to 1.
            reductions (str | Sequence[str], optional): The reduction to apply to the column.
                Should be an available method of dask.dataframe.Series. For example "mean". In this
                case the function is applied to the column to generate a single value for the whole
                dataset. If None, the shift is applied per-dataframe-row. Defaults to None.
                Currently only "mean" is supported.
            preserve_mean (bool | Sequence[bool], optional): Whether to subtract the mean of the
                column before applying the shift. Defaults to False.
            preview (bool, optional): Option to preview the first elements of the data frame.
                Defaults to False.

        Raises:
            ValueError: If the delay column is not in the dataframe.
        """
        delay_column = self._config["dataframe"]["columns"]["delay"]
        if delay_column not in self._dataframe.columns:
            raise ValueError(f"Delay column {delay_column} not found in dataframe! ")

        if self.dataframe is not None:
            logger.info("Adding delay offset to dataframe:")
            df, metadata = self.dc.add_offsets(
                df=self._dataframe,
                constant=constant,
                flip_delay_axis=flip_delay_axis,
                columns=columns,
                delay_column=delay_column,
                weights=weights,
                reductions=reductions,
                preserve_mean=preserve_mean,
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
                    suppress_output=True,
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
            logger.info(self._dataframe.head(10))
        else:
            logger.info(self._dataframe)

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

        offsets = deepcopy(self.dc.offsets)

        if "creation_date" not in offsets.keys():
            offsets["creation_date"] = datetime.now()

        offsets["creation_date"] = offsets["creation_date"].isoformat()

        config = {"delay": {"offsets": offsets}}
        save_config(config, filename, overwrite)
        logger.info(f'Saved delay offset parameters to "{filename}".')

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

    @call_logger(logger)
    def add_jitter(
        self,
        cols: list[str] = None,
        amps: float | Sequence[float] = None,
        **kwds,
    ):
        """Add jitter to the selected dataframe columns.

        Args:
            cols (list[str], optional): The columns onto which to apply jitter.
                Defaults to config["dataframe"]["jitter_cols"].
            amps (float | Sequence[float], optional): Amplitude scalings for the
                jittering noise. If one number is given, the same is used for all axes.
                For uniform noise (default) it will cover the interval [-amp, +amp].
                Defaults to config["dataframe"]["jitter_amps"].
            **kwds: additional keyword arguments passed to ``apply_jitter``.
        """
        if cols is None:
            cols = self._config["dataframe"]["jitter_cols"]
        for loc, col in enumerate(cols):
            if col.startswith("@"):
                cols[loc] = self._config["dataframe"]["columns"].get(col.strip("@"))

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
        logger.info(f"add_jitter: Added jitter to columns {cols}.")

    @call_logger(logger)
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
            **kwds:

                - **time_stamp_column**: Dataframe column containing time-stamp data

                Additional keyword arguments passed to ``add_time_stamped_data``.
        """
        time_stamp_column = kwds.pop(
            "time_stamp_column",
            self._config["dataframe"]["columns"].get("timestamp", ""),
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
        metadata: list[Any] = []
        metadata.append(dest_column)
        metadata.append(time_stamps)
        metadata.append(data)
        self._attributes.add(metadata, "time_stamped_data", duplicate_policy="append")
        logger.info(f"add_time_stamped_data: Added time-stamped data as column {dest_column}.")

    @call_logger(logger)
    def pre_binning(
        self,
        df_partitions: int | Sequence[int] = 100,
        axes: list[str] = None,
        bins: list[int] = None,
        ranges: Sequence[tuple[float, float]] = None,
        **kwds,
    ) -> xr.DataArray:
        """Function to do an initial binning of the dataframe loaded to the class.

        Args:
            df_partitions (int | Sequence[int], optional): Number of dataframe partitions to
                use for the initial binning. Defaults to 100.
            axes (list[str], optional): Axes to bin.
                Defaults to config["momentum"]["axes"].
            bins (list[int], optional): Bin numbers to use for binning.
                Defaults to config["momentum"]["bins"].
            ranges (Sequence[tuple[float, float]], optional): Ranges to use for binning.
                Defaults to config["momentum"]["ranges"].
            **kwds: Keyword argument passed to ``compute``.

        Returns:
            xr.DataArray: pre-binned data-array.
        """
        if axes is None:
            axes = self._config["momentum"]["axes"]
        for loc, axis in enumerate(axes):
            if axis.startswith("@"):
                axes[loc] = self._config["dataframe"]["columns"].get(axis.strip("@"))

        if bins is None:
            bins = self._config["momentum"]["bins"]
        if ranges is None:
            ranges_ = list(self._config["momentum"]["ranges"])
            ranges_[2] = np.asarray(ranges_[2]) / self._config["dataframe"]["tof_binning"]
            ranges = [cast(tuple[float, float], tuple(v)) for v in ranges_]

        assert self._dataframe is not None, "dataframe needs to be loaded first!"

        return self.compute(
            bins=bins,
            axes=axes,
            ranges=ranges,
            df_partitions=df_partitions,
            **kwds,
        )

    @call_logger(logger)
    def compute(
        self,
        bins: int | dict | tuple | list[int] | list[np.ndarray] | list[tuple] = 100,
        axes: str | Sequence[str] = None,
        ranges: Sequence[tuple[float, float]] = None,
        normalize_to_acquisition_time: bool | str = False,
        **kwds,
    ) -> xr.DataArray:
        """Compute the histogram along the given dimensions.

        Args:
            bins (int | dict | tuple | list[int] | list[np.ndarray] | list[tuple], optional):
                Definition of the bins. Can be any of the following cases:

                - an integer describing the number of bins in on all dimensions
                - a tuple of 3 numbers describing start, end and step of the binning
                  range
                - a np.arrays defining the binning edges
                - a list (NOT a tuple) of any of the above (int, tuple or np.ndarray)
                - a dictionary made of the axes as keys and any of the above as values.

                This takes priority over the axes and range arguments. Defaults to 100.
            axes (str | Sequence[str], optional): The names of the axes (columns)
                on which to calculate the histogram. The order will be the order of the
                dimensions in the resulting array. Defaults to None.
            ranges (Sequence[tuple[float, float]], optional): list of tuples containing
                the start and end point of the binning range. Defaults to None.
            normalize_to_acquisition_time (bool | str): Option to normalize the
                result to the acquisition time. If a "slow" axis was scanned, providing
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
                  Defaults to config["core"]["num_cores"] or N_CPU-1.
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
        num_cores = kwds.pop("num_cores", self._config["core"]["num_cores"])
        threads_per_worker = kwds.pop(
            "threads_per_worker",
            self._config["binning"]["threads_per_worker"],
        )
        threadpool_api = kwds.pop(
            "threadpool_API",
            self._config["binning"]["threadpool_API"],
        )
        df_partitions: int | Sequence[int] = kwds.pop("df_partitions", None)
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
                logger.info(f"Calculate normalization histogram for axis '{axis}'...")
                self._normalization_histogram = self.get_normalization_histogram(
                    axis=axis,
                    df_partitions=df_partitions,
                )
                # if the axes are named correctly, xarray figures out the normalization correctly
                self._normalized = self._binned / self._normalization_histogram
                # Set datatype of binned data
                self._normalized.data = self._normalized.data.astype(self._binned.data.dtype)
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

    @call_logger(logger)
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

        df_partitions: int | Sequence[int] = kwds.pop("df_partitions", None)

        if len(kwds) > 0:
            raise TypeError(
                f"get_normalization_histogram() got unexpected keyword arguments {kwds.keys()}.",
            )

        if isinstance(df_partitions, int):
            df_partitions = list(range(0, min(df_partitions, self._dataframe.npartitions)))

        if use_time_stamps or self._timed_dataframe is None:
            if df_partitions is not None:
                dataframe = self._dataframe.partitions[df_partitions]
            else:
                dataframe = self._dataframe
            self._normalization_histogram = normalization_histogram_from_timestamps(
                df=dataframe,
                axis=axis,
                bin_centers=self._binned.coords[axis].values,
                time_stamp_column=self._config["dataframe"]["columns"]["timestamp"],
            )
        else:
            if df_partitions is not None:
                timed_dataframe = self._timed_dataframe.partitions[df_partitions]
            else:
                timed_dataframe = self._timed_dataframe
            self._normalization_histogram = normalization_histogram_from_timed_dataframe(
                df=timed_dataframe,
                axis=axis,
                bin_centers=self._binned.coords[axis].values,
                time_unit=self._config["dataframe"]["timed_dataframe_unit_time"],
                hist_mode=self.config["binning"]["hist_mode"],
                mode=self.config["binning"]["mode"],
                pbar=self.config["binning"]["pbar"],
                n_cores=self.config["core"]["num_cores"],
                threads_per_worker=self.config["binning"]["threads_per_worker"],
                threadpool_api=self.config["binning"]["threadpool_API"],
            )

        return self._normalization_histogram

    def view_event_histogram(
        self,
        dfpid: int,
        ncol: int = 2,
        bins: Sequence[int] = None,
        axes: Sequence[str] = None,
        ranges: Sequence[tuple[float, float]] = None,
        backend: str = "matplotlib",
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
            bins (Sequence[int], optional): Number of bins to use for the specified
                axes. Defaults to config["histogram"]["bins"].
            axes (Sequence[str], optional): Names of the axes to display.
                Defaults to config["histogram"]["axes"].
            ranges (Sequence[tuple[float, float]], optional): Value ranges of all
                specified axes. Defaults to config["histogram"]["ranges"].
            backend (str, optional): Backend of the plotting library
                ("matplotlib" or "bokeh"). Defaults to "matplotlib".
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
                axes[loc] = self._config["dataframe"]["columns"].get(axis.strip("@"))
        if ranges is None:
            ranges = list(self._config["histogram"]["ranges"])
            for loc, axis in enumerate(axes):
                if axis == self._config["dataframe"]["columns"]["tof"]:
                    ranges[loc] = np.asarray(ranges[loc]) / self._config["dataframe"]["tof_binning"]
                elif axis == self._config["dataframe"]["columns"]["adc"]:
                    ranges[loc] = np.asarray(ranges[loc]) / self._config["dataframe"]["adc_binning"]

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

    @call_logger(logger)
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

            **kwds: Keyword arguments, which are passed to the writer functions:
                For TIFF writing:

                - **alias_dict**: Dictionary of dimension aliases to use.

                For HDF5 writing:

                - **mode**: hdf5 read/write mode. Defaults to "w".

                For NeXus:

                - **reader**: Name of the pynxtools reader to use.
                  Defaults to config["nexus"]["reader"]
                - **definition**: NeXus application definition to use for saving.
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
                    [str(path) for path in self._config["nexus"]["input_files"]],
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
