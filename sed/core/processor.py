from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import dask.dataframe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import xarray as xr

from sed.binning import bin_dataframe
from sed.calibrator.energy import EnergyCalibrator
from sed.calibrator.momentum import MomentumCorrector
from sed.config.settings import parse_config
from sed.core.dfops import apply_jitter
from sed.core.metadata import MetaHandler
from sed.diagnostics import grid_histogram
from sed.loader.mirrorutil import CopyTool

N_CPU = psutil.cpu_count()


class SedProcessor:  # pylint: disable=R0902
    """[summary]"""

    def __init__(  # pylint: disable=W0102
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame] = None,
        metadata: dict = {},
        config: Union[dict, str] = {},
    ):

        self._config = parse_config(config)
        num_cores = self._config.get("binning", {}).get("num_cores", N_CPU - 1)
        if num_cores >= N_CPU:
            num_cores = N_CPU - 1
        self._config["binning"]["num_cores"] = num_cores

        self._dataframe = df

        self._binned: xr.DataArray = None
        self._pre_binned: xr.DataArray = None

        self._dimensions = []
        self._coordinates = {}
        self._attributes = MetaHandler(meta=metadata)
        self.ec = EnergyCalibrator(  # pylint: disable=invalid-name
            config=self._config,
        )
        self.mc = MomentumCorrector(config=self._config)

        self.use_copy_tool = self._config.get("core", {}).get(
            "use_copy_tool",
            False,
        )
        if self.use_copy_tool:
            try:
                self.ct = CopyTool(  # pylint: disable=invalid-name
                    source=self._config["core"]["copy_tool_source"],
                    dest=self._config["core"]["copy_tool_dest"],
                )
            except KeyError:
                self.use_copy_tool = False

    def __repr__(self):
        if self._dataframe is None:
            df_str = "Data Frame: No Data loaded"
        else:
            df_str = self._dataframe.__repr__()
        coordinates_str = f"Coordinates: {self._coordinates}"
        dimensions_str = f"Dimensions: {self._dimensions}"
        s = df_str + "\n" + coordinates_str + "\n" + dimensions_str
        return s

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
    def config(self):
        return self._config

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims):
        assert isinstance(dims, list)
        self._dimensions = dims

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords):
        assert isinstance(coords, dict)
        self._coordinates = {}
        for k, v in coords.items():
            self._coordinates[k] = xr.DataArray(v)

    @config.setter
    def config(self, config: Union[dict, str]):
        self._config = parse_config(config)
        num_cores = self._config.get("binning", {}).get("num_cores", N_CPU - 1)
        if num_cores >= N_CPU:
            num_cores = N_CPU - 1
        self._config["binning"]["num_cores"] = num_cores

    def cpy(self, path: Union[str, List[str]]) -> List[str]:
        """Returns either the original or the copied path to the given path.

        Args:
            path (Union[str, List[str]]): Source path or path list

        Returns:
            Union[str, List[str]]: Source or destination path or path list.
        """
        if self.use_copy_tool:
            path_out = []
            if isinstance(path, list):
                for file in path:
                    path_out.append(self.ct.copy(file))
            else:
                path_out.append(self.ct.copy(path))

            return path_out

        if isinstance(path, list):
            return path
        else:
            return [path]

    def load(
        self,
        data: Union[pd.DataFrame, dask.dataframe.DataFrame],
    ) -> None:
        """Load tabular data of Single Events

        Args:
            data: data in tabular format. Accepts anything which
                can be interpreted by pd.DataFrame as an input

        Returns:
            None
        """
        self._dataframe = data

    # Momentum calibration workflow
    # 1. Bin raw detector data for distortion correction
    def bin_and_load_momentum_calibration(
        self,
        df_partitions: int = 100,
        rotation_symmetry: int = 6,
        axes: List[str] = None,
        bins: List[int] = None,
        ranges: List[Tuple] = None,
        **kwds,
    ):
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
            ranges = [tuple(v) for v in ranges_]

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
        threadpool_API = kwds.pop(
            "threadpool_API",
            self._config["binning"]["threadpool_API"],
        )

        df_partitions = min(df_partitions, self._dataframe.npartitions)
        self._pre_binned = bin_dataframe(
            df=self._dataframe.partitions[0:df_partitions],
            bins=bins,
            axes=axes,
            ranges=ranges,
            histMode=hist_mode,
            mode=mode,
            pbar=pbar,
            nCores=num_cores,
            nThreadsPerWorker=threads_per_worker,
            threadpoolAPI=threadpool_API,
            **kwds,
        )

        self.mc.load_data(data=self._pre_binned, rotsym=rotation_symmetry)
        self.mc.select_slicer()

    # 2. Generate the splin warp correction from momentum features.
    # Either autoselect features, or input features from view above.
    def generate_splinewarp(
        self,
        features: np.ndarray = None,
        auto_detect: bool = False,
        use_center: bool = False,
        **kwds,
    ):
        center_det = kwds.pop(
            "center_det",
            self._config.get("momentum", {}).get("center_det", "centroidnn"),
        )
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
            if use_center:
                self.mc.feature_extract(
                    sigma=sigma,
                    fwhm=fwhm,
                    sigma_radius=sigma_radius,
                    center_det=center_det,
                    **kwds,
                )
            else:
                self.mc.feature_extract(
                    sigma=sigma,
                    fwhm=fwhm,
                    sigma_radius=sigma_radius,
                    center_det=None,
                    **kwds,
                )
        else:  # Manual feature selection
            assert features is not None
            if use_center:
                self.mc.add_features(features, center_det=center_det, **kwds)
            else:
                self.mc.add_features(features, center_det=None, **kwds)

        print("Original slice with reference features")
        self.mc.view(annotated=True, backend="bokeh", crosshair=True)

        self.mc.spline_warp_estimate(include_center=use_center, **kwds)

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
    def pose_adjustment(self):
        self.mc.pose_adjustment()

    # Energy calibrator workflow
    # 1. Load and normalize data
    def load_bias_series(  # pylint: disable=R0913
        self,
        data_files: List[str],
        axes: List[str] = None,
        bins: List = None,
        ranges: List[Tuple] = None,
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
            data_files=self.cpy(data_files),
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
    def find_bias_peaks(  # pylint: disable=too-many-arguments
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
    def calibrate_energy_axis(  # pylint: disable=R0913
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
            print("Adding energy columng to dataframe:")
            self._dataframe = self.ec.append_energy_axis(self._dataframe)
            print(self._dataframe.head(10))

    def add_jitter(self, cols: Sequence[str] = None) -> None:
        """Add jitter to the selected dataframe columns.


        Args:
            cols: the colums onto which to apply jitter. If omitted,
            the comlums are taken from the config.

        Returns:
            None
        """
        if cols is None:
            try:
                cols = self._config["jitter_cols"]
            except KeyError:
                cols = self._dataframe.columns  # jitter all columns

        self._dataframe = self._dataframe.map_partitions(
            apply_jitter,
            cols=cols,
            cols_jittered=cols,
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
        threadpool_API = kwds.pop(
            "threadpool_API",
            self._config["binning"]["threadpool_API"],
        )

        self._binned = bin_dataframe(
            df=self._dataframe,
            bins=bins,
            axes=axes,
            ranges=ranges,
            histMode=hist_mode,
            mode=mode,
            pbar=pbar,
            nCores=num_cores,
            nThreadsPerWorker=threads_per_worker,
            threadpoolAPI=threadpool_API,
            **kwds,
        )
        return self._binned

    def viewEventHistogram(
        self,
        dfpid: int,
        ncol: int = 2,
        bins: Sequence[int] = None,
        axes: Sequence[str] = None,
        ranges: Sequence[Tuple[float, float]] = None,
        backend: str = "bokeh",
        legend: bool = True,
        histkwds: dict = {},
        legkwds: dict = {},
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

    def add_dimension(self, name, range):
        if name in self._coordinates:
            raise ValueError(f"Axis {name} already exists")
        else:
            self.axis[name] = self.make_axis(range)
