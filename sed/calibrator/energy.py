"""sed.calibrator.energy module. Code for energy calibration and
correction. Mostly ported from https://github.com/mpes-kit/mpes.
"""
import itertools as it
import pickle
import warnings as wn
from copy import deepcopy
from functools import partial
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import bokeh.plotting as pbk
import dask.dataframe
import deepdish.io as dio
import h5py
import ipywidgets as ipw
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr
from bokeh.io import output_notebook
from bokeh.palettes import Category10 as ColorCycle
from fastdtw import fastdtw
from IPython.display import display
from lmfit import Minimizer
from lmfit import Parameters
from lmfit.printfuncs import report_fit
from numpy.linalg import lstsq
from scipy.signal import savgol_filter
from scipy.sparse.linalg import lsqr
from silx.io import dictdump

from sed.binning import bin_dataframe
from sed.loader.mpes import MpesLoader


class EnergyCalibrator:
    """
    Electron binding energy calibration workflow.
    """

    def __init__(
        self,
        biases: np.ndarray = None,
        traces: np.ndarray = None,
        tof: np.ndarray = None,
        config: dict = None,
    ):
        """Initialization of the EnergyCalibrator class can follow different ways,

        1. Initialize with all the file paths in a list
        1a. Use an hdf5 file containing all binned traces and tof
        1b. Use a mat file containing all binned traces and tof
        1c. Use the raw data hdf5 files
        2. Initialize with the folder path containing any of the above files
        3. Initialize with the binned traces and the time-of-flight
        """

        if biases is not None:
            self.biases = biases
        else:
            self.biases = np.asarray([])
        if tof is not None:
            self.tof = tof
        else:
            self.tof = np.asarray([])
        if traces is not None:
            self.traces = self.traces_normed = traces
        else:
            self.traces = self.traces_normed = np.asarray([])

        if config is None:
            config = {}

        self._config = config

        self.featranges: List[Tuple] = []  # Value ranges for feature detection
        self.peaks: np.ndarray = np.asarray([])
        self.calibration: Dict[Any, Any] = {}

        self.tof_column = self._config.get("dataframe", {}).get(
            "tof_column",
            "t",
        )
        self.energy_column = self._config.get("dataframe", {}).get(
            "energy_column",
            "E",
        )
        self.x_column = self._config.get("dataframe", {}).get("x_column", "X")
        self.y_column = self._config.get("dataframe", {}).get("y_column", "Y")
        self.binwidth: float = self._config.get("dataframe", {}).get(
            "tof_binwidth",
            4.125e-12,
        )
        self.binning: int = self._config.get("dataframe", {}).get(
            "tof_binning",
            1,
        )

        self.tof_fermi = self._config.get("energy", {}).get(
            "tof_fermi",
            132250,
        )
        self.x_width = self._config.get("energy", {}).get("x_width", (-20, 20))
        self.y_width = self._config.get("energy", {}).get("y_width", (-20, 20))
        self.tof_width = self._config.get("energy", {}).get(
            "tof_width",
            (-300, 500),
        )
        self.color_clip = self._config.get("energy", {}).get("color_clip", 300)

        self.correction: Dict[Any, Any] = self._config.get("energy", {}).get(
            "correction",
            {
                "correction_type": "Lorentzian",
                "amplitude": 8,
                "center": (750, 730),
                "kwds": {"sigma": 920},
            },
        )

    @property
    def ntraces(self) -> int:
        """The number of loaded/calculated traces."""

        return len(self.traces)

    @property
    def nranges(self) -> int:
        """The number of specified feature ranges."""

        return len(self.featranges)

    @property
    def dup(self) -> int:
        """The duplication number."""

        return int(np.round(self.nranges / self.ntraces))

    def load_data(
        self,
        biases: np.ndarray = None,
        traces: np.ndarray = None,
        tof: np.ndarray = None,
    ):
        """Load data to the class

        Parameters:
            biases: Bias voltages used
            traces: TOF-Data traces corresponding to the bias values
            tof: TOF-values for the data traces
        """
        self.biases = biases
        self.tof = tof
        self.traces = self.traces_normed = traces

    def bin_data(
        self,
        data_files: List[str],
        axes: List[str] = None,
        bins: List[int] = None,
        ranges: Sequence[Tuple[float, float]] = None,
        biases: np.ndarray = None,
        bias_key: str = None,
        **kwds,
    ):
        """Load and bin data from single-event files

        Parameters:
            data_files: list of file names to bin
            axes: bin axes | _config["dataframe"]["tof_column"] / "t"
            bins: number of bins | _config["energy"]["bins"] / 1000
            ranges: bin ranges | _config["energy"]["ranges"] / [128000, 138000]
            biases: Bias voltages used
            bias_key: hdf5 path where bias values are stored.
                    | _config["energy"]["bias_key"]
            **kwds: Keyword parameters for bin_dataframe

        """
        if axes is None:
            axes = [self.tof_column]
        if bins is None:
            bins = [self._config.get("energy", {}).get("bins", 1000)]
        if ranges is None:
            ranges_ = [
                self._config.get("energy", {}).get("ranges", [128000, 138000]),
            ]
            ranges = [cast(Tuple[float, float], tuple(v)) for v in ranges_]
        # pylint: disable=duplicate-code
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

        read_biases = False
        if biases is None:
            read_biases = True
            if bias_key is None:
                bias_key = self._config.get("energy", {}).get("bias_key", "")

        loader = MpesLoader(config=self._config)
        dataframe = loader.read_dataframe(files=data_files)
        traces = bin_dataframe(
            dataframe,
            bins=bins,
            axes=axes,
            ranges=ranges,
            histMode=hist_mode,
            mode=mode,
            pbar=pbar,
            nCores=num_cores,
            nThreadsPerWorker=threads_per_worker,
            threadpoolAPI=threadpool_api,
            return_partitions=True,
            **kwds,
        )
        if read_biases:
            biases = extract_bias(data_files, bias_key)
        tof = traces.coords[(axes[0])]
        self.traces = self.traces_normed = np.asarray(traces.T)
        self.tof = np.asarray(tof)
        self.biases = np.asarray(biases)

    def normalize(self, smooth: bool = False, span: int = 7, order: int = 1):
        """Normalize the spectra along an axis.

        **Parameters**\n
        smooth: bool | False
            Option to smooth the signals before normalization.
        span, order: int, int | 7, 1
            Smoothing parameters of the LOESS method
            (see ``scipy.signal.savgol_filter()``).
        """

        self.traces_normed = normspec(
            self.traces,
            smooth=smooth,
            span=span,
            order=order,
        )

    def add_features(
        self,
        ranges: Union[List[Tuple], Tuple],
        ref_id: int = 0,
        traces: np.ndarray = None,
        infer_others: bool = True,
        mode: str = "replace",
        **kwds,
    ):
        """Select or extract the equivalent landmarks (e.g. peaks) among all traces.

        **Parameters**\n
        ranges: list/tuple
            Collection of feature detection ranges, within which an algorithm
            (i.e. 1D peak detector) with look for the feature.
        ref_id: int | 0
            Index of the reference trace (EDC).
        traces: 2D array | None
            Collection of energy dispersion curves (EDCs).
        infer_others: bool | True
            Option to infer the feature detection range in other traces (EDCs) from a
            given one.
        mode: str | 'append'
            Specification on how to change the feature ranges ('append' or 'replace').
        **kwds: keyword arguments
            Dictionarized keyword arguments for trace alignment
            (See ``self.findCorrespondence()``)
        """

        if traces is None:
            traces = self.traces_normed

        # Infer the corresponding feature detection range of other traces by alignment
        if infer_others:
            assert isinstance(ranges, tuple)
            newranges: List[Tuple] = []

            for i in range(self.ntraces):

                pathcorr = find_correspondence(
                    traces[ref_id, :],
                    traces[i, :],
                    **kwds,
                )
                newranges.append(range_convert(self.tof, ranges, pathcorr))

        else:
            if isinstance(ranges, list):
                newranges = ranges
            else:
                newranges = [ranges]

        if mode == "append":
            self.featranges += newranges
        elif mode == "replace":
            self.featranges = newranges

    def feature_extract(
        self,
        ranges: List[Tuple] = None,
        traces: np.ndarray = None,
        peak_window: int = 7,
    ):
        """Select or extract the equivalent landmarks (e.g. peaks) among all traces.

        **Parameters**\n
        ranges: list/tuple | None
            Range in each trace to look for the peak feature, [start, end].
        traces: 2D array | None
            Collection of 1D spectra to use for calibration.
        peak_window:
            area around a peak to check for other peaks.
        """

        if ranges is None:
            ranges = self.featranges

        if traces is None:
            traces = self.traces_normed

        # Augment the content of the calibration data
        traces_aug = np.tile(traces, (self.dup, 1))
        # Run peak detection for each trace within the specified ranges
        self.peaks = peaksearch(
            traces_aug,
            self.tof,
            ranges=ranges,
            pkwindow=peak_window,
        )

    def calibrate(
        self,
        ref_id: int = 0,
        method: str = "lmfit",
        energy_scale: str = "kinetic",
        landmarks: np.ndarray = None,
        biases: np.ndarray = None,
        t: np.ndarray = None,
        **kwds,
    ) -> dict:
        """Calculate the functional mapping between time-of-flight and the energy
        scale using optimization methods.

        **Parameters**\n
        refid: int | 0
            The reference trace index (an integer).
        ret: list | ['coeffs']
            Options for return values (see ``mpes.analysis.calibrateE()``).
        method: str | lmfit
            Method for determining the energy calibration. "lmfit" or "lstsq", "lsqr"
        energy_scale: str | kinetic
            Direction of increasing energy scale. "kinetic" (decreasing TOF) or
            "binding" (increasing TOF).
        **kwds: keyword arguments
            See available keywords for ``poly_energy_calibration()`` and
            ``fit_energy_calibation()``
        """

        if landmarks is None:
            landmarks = self.peaks[:, 0]
        if biases is None:
            biases = self.biases
        if t is None:
            t = self.tof
        if energy_scale == "kinetic":
            sign = -1
        elif energy_scale == "binding":
            sign = 1
        else:
            raise ValueError(
                'energy_scale needs to be either "binding" or "kinetic"',
                f", got {energy_scale}.",
            )

        binwidth = kwds.pop("binwidth", self.binwidth)
        binning = kwds.pop("binning", self.binning)

        if method == "lmfit":
            self.calibration = fit_energy_calibation(
                landmarks,
                sign * biases,
                binwidth,
                binning,
                ref_id=ref_id,
                t=t,
                energy_scale=energy_scale,
                **kwds,
            )
        elif method in ("lstsq", "lsqr"):
            self.calibration = poly_energy_calibration(
                landmarks,
                sign * biases,
                ref_id=ref_id,
                aug=self.dup,
                method=method,
                t=t,
                **kwds,
            )
        else:
            raise NotImplementedError()

        return self.calibration

    def view(  # pylint: disable=dangerous-default-value
        self,
        traces: np.ndarray,
        segs: List[Tuple] = None,
        peaks: np.ndarray = None,
        show_legend: bool = True,
        backend: str = "matplotlib",
        linekwds: dict = {},
        linesegkwds: dict = {},
        scatterkwds: dict = {},
        legkwds: dict = {},
        **kwds,
    ):
        """Display a plot showing line traces with annotation.

        **Parameters**\n
        traces: 2d array
            Matrix of traces to visualize.
        segs: list/tuple
            Segments to be highlighted in the visualization.
        peaks: 2d array
            Peak positions for labelling the traces.
        ret: bool
            Return specification.
        backend: str | 'matplotlib'
            Backend specification, choose between 'matplotlib' (static) or 'bokeh'
            (interactive).
        linekwds: dict | {}
            Keyword arguments for line plotting (see ``matplotlib.pyplot.plot()``).
        scatterkwds: dict | {}
            Keyword arguments for scatter plot (see ``matplotlib.pyplot.scatter()``).
        legkwds: dict | {}
            Keyword arguments for legend (see ``matplotlib.pyplot.legend()``).
        **kwds: keyword arguments
            ===============  ==========  ================================
            keyword          data type   meaning
            ===============  ==========  ================================
            labels           list        Labels for each curve
            xaxis            1d array    x (horizontal) axis values
            title            str         Title of the plot
            legend_location  str         Location of the plot legend
            align            bool        Option to shift traces by bias voltage
            ===============  ==========  ================================
        """

        lbs = kwds.pop("labels", [str(b) + " V" for b in self.biases])
        xaxis = kwds.pop("xaxis", self.tof)
        ttl = kwds.pop("title", "")
        align = kwds.pop("align", False)
        energy_scale = kwds.pop("energy_scale", "kinetic")

        sign = 1 if energy_scale == "kinetic" else -1

        if backend == "matplotlib":

            figsize = kwds.pop("figsize", (12, 4))
            fig, ax = plt.subplots(figsize=figsize)
            for itr, trace in enumerate(traces):
                if align:
                    ax.plot(
                        xaxis
                        + sign
                        * (
                            self.biases[itr]
                            - self.biases[self.calibration["refid"]]
                        ),
                        trace,
                        ls="--",
                        linewidth=1,
                        label=lbs[itr],
                        **linekwds,
                    )
                else:
                    ax.plot(
                        xaxis,
                        trace,
                        ls="--",
                        linewidth=1,
                        label=lbs[itr],
                        **linekwds,
                    )

                # Emphasize selected EDC segments
                if segs is not None:
                    seg = segs[itr]
                    cond = (self.tof >= seg[0]) & (self.tof <= seg[1])
                    tofseg, traceseg = self.tof[cond], trace[cond]
                    ax.plot(
                        tofseg,
                        traceseg,
                        color="k",
                        linewidth=2,
                        **linesegkwds,
                    )
                # Emphasize extracted local maxima
                if peaks is not None:
                    ax.scatter(
                        peaks[itr, 0],
                        peaks[itr, 1],
                        s=30,
                        **scatterkwds,
                    )

            if show_legend:
                try:
                    ax.legend(fontsize=12, **legkwds)
                except TypeError:
                    pass

            ax.set_title(ttl)

        elif backend == "bokeh":

            output_notebook(hide_banner=True)
            colors = it.cycle(ColorCycle[10])
            ttp = [("(x, y)", "($x, $y)")]

            figsize = kwds.pop("figsize", (800, 300))
            fig = pbk.figure(
                title=ttl,
                plot_width=figsize[0],
                plot_height=figsize[1],
                tooltips=ttp,
            )
            # Plotting the main traces
            for itr, color in zip(range(len(traces)), colors):
                trace = traces[itr, :]
                if align:
                    fig.line(
                        xaxis
                        + sign
                        * (
                            self.biases[itr]
                            - self.biases[self.calibration["refid"]]
                        ),
                        trace,
                        color=color,
                        line_dash="solid",
                        line_width=1,
                        line_alpha=1,
                        legend_label=lbs[itr],
                        **kwds,
                    )
                else:
                    fig.line(
                        xaxis,
                        trace,
                        color=color,
                        line_dash="solid",
                        line_width=1,
                        line_alpha=1,
                        legend_label=lbs[itr],
                        **kwds,
                    )

                # Emphasize selected EDC segments
                if segs is not None:
                    seg = segs[itr]
                    cond = (self.tof >= seg[0]) & (self.tof <= seg[1])
                    tofseg, traceseg = self.tof[cond], trace[cond]
                    fig.line(
                        tofseg,
                        traceseg,
                        color=color,
                        line_width=3,
                        **linekwds,
                    )

                # Plot detected peaks
                if peaks is not None:
                    fig.scatter(
                        peaks[itr, 0],
                        peaks[itr, 1],
                        fill_color=color,
                        fill_alpha=0.8,
                        line_color=None,
                        size=5,
                        **scatterkwds,
                    )

            if show_legend:
                fig.legend.location = kwds.pop("legend_location", "top_right")
                fig.legend.spacing = 0
                fig.legend.padding = 2

            pbk.show(fig)

    def append_energy_axis(
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        tof_column: str = None,
        energy_column: str = None,
        **kwds,
    ) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
        """Calculate and append the E axis to the events dataframe.
        This method can be reused.

        **Parameter**\n
        ...
        """

        if tof_column is None:
            tof_column = self.tof_column

        if energy_column is None:
            energy_column = self.energy_column

        binwidth = kwds.pop("binwidth", self.binwidth)
        binning = kwds.pop("binning", self.binning)

        calib_type = ""

        if "t0" in kwds and "d" in kwds and "E0" in kwds:
            time_offset = kwds.pop("t0")
            drift_distance = kwds.pop("d")
            energy_offset = kwds.pop("E0")
            energy_scale = kwds.pop("energy_scale", "kinetic")
            calib_type = "fit"

        elif "a" in kwds and "E0" in kwds:
            poly_a = kwds.pop("a")
            energy_offset = kwds.pop("E0")
            calib_type = "poly"

        elif (
            "t0" in self.calibration
            and "d" in self.calibration
            and "E0" in self.calibration
            and "energy_scale" in self.calibration
        ):
            time_offset = self.calibration["t0"]
            drift_distance = self.calibration["d"]
            energy_offset = self.calibration["E0"]
            energy_scale = self.calibration["energy_scale"]
            calib_type = "fit"
        elif "coeffs" in self.calibration and "E0" in self.calibration:
            poly_a = self.calibration["coeffs"]
            energy_offset = self.calibration["E0"]
            calib_type = "poly"

        if calib_type == "fit":
            df[energy_column] = tof2ev(
                drift_distance,
                time_offset,
                binwidth,
                binning,
                energy_scale,
                energy_offset,
                df[tof_column].astype("float64"),
            )
        elif calib_type == "poly":
            df[energy_column] = tof2evpoly(
                poly_a,
                energy_offset,
                df[tof_column].astype("float64"),
            )
        else:
            raise NotImplementedError

        return df

    def adjust_energy_correction(
        self,
        image: xr.DataArray,
        correction_type: str = None,
        amplitude: float = None,
        center: Tuple[float, float] = None,
        apply=False,
        **kwds,
    ):
        """Visualize the energy correction function ontop of the TOF/X/Y graphs.

        :Parameters:
            image: xarray
                Image data cube (x, y, tof) of binned data to plot
            correction_type: str
                Type of correction to apply to the TOF axis. Defaults to config value.
            :amplitude: numeric | config
                Amplitude of the time-of-flight correction term
            apply: bool | False
                whether to store the provided parameters within the class
            **kwds: keyword arguments
                Additional parameters to use for the correction.
                :x_column: str | config
                    String name of the x axis.
                :y_column: str | config
                    String name of the y axis.
                :tof_column: str | config
                    String name of the tof axis to correct.
                :center: list/tuple | config
                    Image center pixel positions in (x, y) format.
                :x_width: (int, int):
                    x range to integrate around the center
                :y_width: (int, int):
                    y range to integrate around the center
                :tof_fermi: int:
                    TOF value of the Fermi level
                :tof_width: (int, int):
                    TOF range to plot around tof_fermi
                :color_clip: int:
                    highest value to plot in the color range
                *** Additional parameters for correction functions: ***
                :d: numeric | 0.9
                    Field-free drift distance.
                :gamma: numeric
                    Linewidth value for correction using a 2D Lorentz profile.
                :sigma: numeric
                    Standard deviation for correction using a 2D Gaussian profile.
                :gamma2: numeric
                    Linewidth value for correction using an asymmetric 2D Lorentz
                    profile, X-direction.
                :amplitude2: numeric
                    Amplitude value for correction using an asymmetric 2D Lorentz
                    profile, X-direction.

        """
        matplotlib.use("module://ipympl.backend_nbagg")
        if correction_type is None:
            correction_type = self.correction["correction_type"]

        if amplitude is None:
            amplitude = self.correction["amplitude"]
        if center is None:
            center = self.correction["center"]

        kwds = {**(self.correction["kwds"]), **kwds}

        x_column = kwds.pop("x_column", self.x_column)
        y_column = kwds.pop("y_column", self.y_column)
        tof_column = kwds.pop("tof_column", self.tof_column)
        x_width = kwds.pop("x_width", self.x_width)
        y_width = kwds.pop("y_width", self.y_width)
        tof_fermi = kwds.pop("tof_fermi", self.tof_fermi)
        tof_width = kwds.pop("tof_width", self.tof_width)
        color_clip = kwds.pop("color_clip", self.color_clip)

        x = image.coords[x_column].values
        y = image.coords[y_column].values

        x_center = center[0]
        y_center = center[1]

        correction_x = tof_fermi - correction_function(
            x=x,
            y=y_center,
            correction_type=correction_type,
            center=center,
            amplitude=amplitude,
            **kwds,
        )
        correction_y = tof_fermi - correction_function(
            x=x_center,
            y=y,
            correction_type=correction_type,
            center=center,
            amplitude=amplitude,
            **kwds,
        )
        fig, ax = plt.subplots(2, 1)
        image.loc[
            {
                y_column: slice(y_center + y_width[0], y_center + y_width[1]),
                tof_column: slice(
                    tof_fermi + tof_width[0],
                    tof_fermi + tof_width[1],
                ),
            }
        ].sum(dim=y_column).T.plot(
            ax=ax[0],
            cmap="terrain_r",
            vmax=color_clip,
            yincrease=False,
        )
        image.loc[
            {
                x_column: slice(x_center + x_width[0], x_center + x_width[1]),
                tof_column: slice(
                    tof_fermi + tof_width[0],
                    tof_fermi + tof_width[1],
                ),
            }
        ].sum(dim=x_column).T.plot(
            ax=ax[1],
            cmap="terrain_r",
            vmax=color_clip,
            yincrease=False,
        )
        (trace1,) = ax[0].plot(x, correction_x)
        line1 = ax[0].axvline(x=center[0])
        (trace2,) = ax[1].plot(y, correction_y)
        line2 = ax[1].axvline(x=center[1])

        amplitude_slider = ipw.FloatSlider(
            value=amplitude,
            min=0,
            max=10,
            step=0.1,
        )
        x_center_slider = ipw.FloatSlider(
            value=x_center,
            min=0,
            max=self._config.get("momentum", {}).get(
                "detector_ranges",
                [[0, 2048], [0, 2048]],
            )[0][1],
            step=1,
        )
        y_center_slider = ipw.FloatSlider(
            value=x_center,
            min=0,
            max=self._config.get("momentum", {}).get(
                "detector_ranges",
                [[0, 2048], [0, 2048]],
            )[1][1],
            step=1,
        )

        def update(amplitude, x_center, y_center, **kwds):
            correction_x = tof_fermi - correction_function(
                x=x,
                y=y_center,
                correction_type=correction_type,
                center=(x_center, y_center),
                amplitude=amplitude,
                **kwds,
            )
            correction_y = tof_fermi - correction_function(
                x=x_center,
                y=y,
                correction_type=correction_type,
                center=(x_center, y_center),
                amplitude=amplitude,
                **kwds,
            )

            trace1.set_ydata(correction_x)
            line1.set_xdata(x=x_center)
            trace2.set_ydata(correction_y)
            line2.set_xdata(x=y_center)

            fig.canvas.draw_idle()

        if correction_type == "spherical":
            diameter = kwds.pop("diameter", 50)

            update(amplitude, x_center, y_center, d=diameter)

            diameter_slider = ipw.FloatSlider(
                value=diameter,
                min=0,
                max=100,
                step=1,
            )

            ipw.interact(
                update,
                amplitude=amplitude_slider,
                x_center=x_center_slider,
                y_center=y_center_slider,
                diameter=diameter_slider,
            )

            def apply_func(apply: bool):  # pylint: disable=unused-argument
                self.correction["amplitude"] = amplitude_slider.value
                self.correction["center"] = (
                    x_center_slider.value,
                    y_center_slider.value,
                )
                self.correction["correction_type"] = correction_type
                kwds["diameter"] = diameter_slider.value
                self.correction["kwds"] = kwds
                amplitude_slider.close()
                x_center_slider.close()
                y_center_slider.close()
                diameter_slider.close()
                apply_button.close()

        elif correction_type == "Lorentzian":
            gamma = kwds.pop("gamma", 700)

            update(amplitude, x_center, y_center, gamma=gamma)

            gamma_slider = ipw.FloatSlider(
                value=gamma,
                min=0,
                max=2000,
                step=1,
            )

            ipw.interact(
                update,
                amplitude=amplitude_slider,
                x_center=x_center_slider,
                y_center=y_center_slider,
                gamma=gamma_slider,
            )

            def apply_func(apply: bool):  # pylint: disable=unused-argument
                self.correction["amplitude"] = amplitude_slider.value
                self.correction["center"] = (
                    x_center_slider.value,
                    y_center_slider.value,
                )
                self.correction["correction_type"] = correction_type
                kwds["gamma"] = gamma_slider.value
                self.correction["kwds"] = kwds
                amplitude_slider.close()
                x_center_slider.close()
                y_center_slider.close()
                gamma_slider.close()
                apply_button.close()

        elif correction_type == "Gaussian":
            sigma = kwds.pop("sigma", 400)

            update(amplitude, x_center, y_center, sigma=sigma)

            sigma_slider = ipw.FloatSlider(
                value=sigma,
                min=0,
                max=1000,
                step=1,
            )

            ipw.interact(
                update,
                amplitude=amplitude_slider,
                x_center=x_center_slider,
                y_center=y_center_slider,
                sigma=sigma_slider,
            )

            def apply_func(apply: bool):  # pylint: disable=unused-argument
                self.correction["amplitude"] = amplitude_slider.value
                self.correction["center"] = (
                    x_center_slider.value,
                    y_center_slider.value,
                )
                self.correction["correction_type"] = correction_type
                kwds["sigma"] = sigma_slider.value
                self.correction["kwds"] = kwds
                amplitude_slider.close()
                x_center_slider.close()
                y_center_slider.close()
                sigma_slider.close()
                apply_button.close()

        elif correction_type == "Lorentzian_asymmetric":
            gamma = kwds.pop("gamma", 700)
            amplitude2 = kwds.pop("amplitude2", amplitude)
            gamma2 = kwds.pop("gamma2", gamma)

            update(
                amplitude,
                x_center,
                y_center,
                gamma=gamma,
                amplitude2=amplitude2,
                gamma2=gamma2,
            )

            gamma_slider = ipw.FloatSlider(
                value=gamma,
                min=0,
                max=2000,
                step=1,
            )

            amplitude2_slider = ipw.FloatSlider(
                value=amplitude,
                min=0,
                max=10,
                step=0.1,
            )

            gamma2_slider = ipw.FloatSlider(
                value=gamma2,
                min=0,
                max=2000,
                step=1,
            )

            ipw.interact(
                update,
                amplitude=amplitude_slider,
                x_center=x_center_slider,
                y_center=y_center_slider,
                gamma=gamma_slider,
                amplitude2=amplitude2_slider,
                gamma2=gamma2_slider,
            )

            def apply_func(apply: bool):  # pylint: disable=unused-argument
                self.correction["amplitude"] = amplitude_slider.value
                self.correction["center"] = (
                    x_center_slider.value,
                    y_center_slider.value,
                )
                self.correction["correction_type"] = correction_type
                kwds["gamma"] = gamma_slider.value
                kwds["amplitude2"] = amplitude2_slider.value
                kwds["gamma2"] = gamma2_slider.value
                self.correction["kwds"] = kwds
                amplitude_slider.close()
                x_center_slider.close()
                y_center_slider.close()
                gamma_slider.close()
                amplitude2_slider.close()
                gamma2_slider.close()
                apply_button.close()

        else:
            raise NotImplementedError
        # pylint: disable=duplicate-code
        apply_button = ipw.Button(description="apply")
        display(apply_button)
        apply_button.on_click(apply_func)
        plt.show()

        if apply:
            apply_func(True)

    def apply_energy_correction(
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        correction_type: str = None,
        amplitude: float = None,
        **kwds,
    ) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
        """Apply correction to the time-of-flight (TOF) axis of single-event data.

        :Parameters:
            df: The dataframe where to apply the energy correction to
            correction_type: str
                Type of correction to apply to the TOF axis. Defaults to config value.
            :amplitude: numeric | config
                Amplitude of the time-of-flight correction term
            **kwds: keyword arguments
                Additional parameters to use for the correction.
                :x_column: str | config
                    String name of the x axis.
                :y_column: str | config
                    String name of the y axis.
                :tof_column: str | config
                    String name of the tof axis to correct.
                :center: list/tuple | config
                    Image center pixel positions in (x, y) format.
                :diameter: numeric | 0.9
                    Field-free drift distance.
                :gamma: numeric
                    Linewidth value for correction using a 2D Lorentz profile.
                :sigma: numeric
                    Standard deviation for correction using a 2D Gaussian profile.
                :gamma2: numeric
                    Linewidth value for correction using an asymmetric 2D Lorentz
                    profile, X-direction.
                :amplitude2: numeric
                    Amplitude value for correction using an asymmetric 2D Lorentz
                    profile, X-direction.

        """

        if correction_type is None:
            correction_type = self.correction["correction_type"]

        if amplitude is None:
            amplitude = self.correction["amplitude"]

        kwds = {**(self.correction["kwds"]), **kwds}

        center = kwds.pop("center", self.correction["center"])

        x_column = kwds.pop("x_column", self.x_column)
        y_column = kwds.pop("y_column", self.y_column)
        tof_column = kwds.pop("tof_column", self.tof_column)

        df[tof_column] += correction_function(
            x=df[x_column],
            y=df[y_column],
            correction_type=correction_type,
            center=center,
            amplitude=amplitude,
            **kwds,
        )

        return df

    def save_class_parameters(
        self,
        form: str = "dmp",
        save_addr: str = "./energy",
    ):
        """
        Save all the attributes of the workflow instance for later use
        (e.g. energy scale conversion).

        Parameters:
            form: str | 'dmp'
                The file format to save the attributes in
                ('h5'/'hdf5', 'mat' or 'dmp'/'dump').
            save_addr: str | './energy'
                The filename to save the files with.
        """
        save_addr = append_extension(save_addr, form)

        save_class_attributes(self, form, save_addr)


def append_extension(filepath: str, extension: str) -> str:
    """
    Append an extension to the end of a file path.

    **Parameters**\n
    filepath: str
        File path of interest.
    extension: str
        File extension
    """

    format_string = "." + extension
    if filepath:
        if not filepath.endswith(format_string):
            filepath += format_string

    return filepath


def save_class_attributes(clss, form, save_addr):
    """Save class attributes.

    **Parameters**\n
    clss: instance
        Handle of the instance to be saved.
    form: str
        Format to save in ('h5'/'hdf5', 'mat', or 'dmp'/'dump').
    save_addr: str
        The address to save the attributes in.
    """
    # Modify the data type for HDF5-convertibility (temporary fix)
    if form == "mat":
        sio.savemat(save_addr, clss.__dict__)
    elif form in ("dmp", "dump"):
        with open(save_addr, "wb") as file_handle:
            pickle.dump(clss, file_handle)
    elif form in ("h5", "hdf5"):
        dictcopy = deepcopy(clss.__dict__)
        dictcopy["featranges"] = np.asarray(dictcopy["featranges"])
        try:
            dictdump.dicttoh5(dictcopy, save_addr)
        except KeyError:
            dio.save(save_addr, dictcopy, compression=None)

    else:
        raise NotImplementedError


def extract_bias(files: List[str], bias_key: str) -> np.ndarray:
    """
    Read bias value from hdf5 file

    Parameters:
        file: filename
        bias_key: hdf5 path to the bias value

    Returns:
        bias value
    """
    bias_list: List[float] = []
    for file in files:
        with h5py.File(file, "r") as file_handle:
            if bias_key[0] == "@":
                bias_list.append(round(file_handle.attrs[bias_key[1:]], 2))
            else:
                bias_list.append(round(file_handle[bias_key], 2))

    return np.asarray(bias_list)


def correction_function(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    correction_type: str,
    center: Tuple[float, float],
    amplitude: float,
    **kwds,
) -> Union[float, np.ndarray]:
    """Calculate the TOF correction based on the given X/Y coordinates and a model

    Args:
        x (float): x coordinate
        y (float): y coordinate
        correction_type (str): type of correction. One of
            "spherical", "Lorentzian", "Gaussian", or "Lorentzian_asymmetric"
        center (Tuple[int, int]): center position of the distribution (x,y)
        amplitude (float): Amplitude of the correction
        **kwds: Keyword arguments:
            :diameter: numeric | 0.9
                Field-free drift distance.
            :gamma: numeric
                Linewidth value for correction using a 2D Lorentz profile.
            :sigma: numeric
                Standard deviation for correction using a 2D Gaussian profile.
            :gam2: numeric
                Linewidth value for correction using an asymmetric 2D Lorentz
                profile, X-direction.
            :amplitude2: numeric
                Amplitude value for correction using an asymmetric 2D Lorentz
                profile, X-direction.

    Returns:
        float: calculated correction value
    """
    if correction_type == "spherical":
        diameter = kwds.pop("diameter", 50)
        correction = -(
            (
                np.sqrt(
                    1
                    + ((x - center[0]) ** 2 + (y - center[1]) ** 2)
                    / diameter**2,
                )
                - 1
            )
            * amplitude
        )

    elif correction_type == "Lorentzian":
        gamma = kwds.pop("gamma", 700)
        correction = (
            100000
            * amplitude
            / (gamma * np.pi)
            * (
                gamma**2
                / ((x - center[0]) ** 2 + (y - center[1]) ** 2 + gamma**2)
                - 1
            )
        )

    elif correction_type == "Gaussian":
        sigma = kwds.pop("sigma", 400)
        correction = (
            20000
            * amplitude
            / np.sqrt(2 * np.pi * sigma**2)
            * (
                np.exp(
                    -((x - center[0]) ** 2 + (y - center[1]) ** 2)
                    / (2 * sigma**2),
                )
                - 1
            )
        )

    elif correction_type == "Lorentzian_asymmetric":
        gamma = kwds.pop("gamma", 700)
        gamma2 = kwds.pop("gamma2", gamma)
        amplitude2 = kwds.pop("amplitude2", amplitude)
        correction = (
            100000
            * amplitude
            / (gamma * np.pi)
            * (gamma**2 / ((y - center[1]) ** 2 + gamma**2) - 1)
        )
        correction += (
            100000
            * amplitude2
            / (gamma2 * np.pi)
            * (gamma2**2 / ((x - center[0]) ** 2 + gamma2**2) - 1)
        )

    else:
        raise NotImplementedError

    return correction


def normspec(
    specs: np.ndarray,
    smooth: bool = False,
    span: int = 7,
    order: int = 1,
) -> np.ndarray:
    """
    Normalize a series of 1D signals.

    Parameters:
    *specs: list/2D array
        Collection of 1D signals.
    smooth: bool | False
        Option to smooth the signals before normalization.
    span, order: int, int | 7, 1
        Smoothing parameters of the LOESS method (see ``scipy.signal.savgol_filter()``).

    **Return**\n
    normalized_specs: 2D array
        The matrix assembled from a list of maximum-normalized signals.
    """

    nspec = len(specs)
    specnorm = []

    for i in range(nspec):

        spec = specs[i]

        if smooth:
            spec = savgol_filter(spec, span, order)

        if type(spec) in (list, tuple):
            nsp = spec / max(spec)
        else:
            nsp = spec / spec.max()
        specnorm.append(nsp)

        # Align 1D spectrum
        normalized_specs = np.asarray(specnorm)

    return normalized_specs


def find_correspondence(
    sig_still: np.ndarray,
    sig_mov: np.ndarray,
    **kwds,
) -> np.ndarray:
    """Determine the correspondence between two 1D traces by alignment.

    **Parameters**\n
    sig_still, sig_mov: 1D array, 1D array
        Input 1D signals.
    **kwds: keyword arguments
        See available keywords for the following functions,
        (1) ``fastdtw.fastdtw()`` (when ``method=='dtw'``)
        (2) ``ptw.ptw.timeWarp()`` (when ``method=='ptw'``)

    **Return**\n
    pathcorr: list
        Pixel-wise path correspondences between two input 1D arrays
        (sig_still, sig_mov).
    """

    dist = kwds.pop("dist_metric", None)
    rad = kwds.pop("radius", 1)
    _, pathcorr = fastdtw(sig_still, sig_mov, dist=dist, radius=rad)
    return np.asarray(pathcorr)


def range_convert(
    x: np.ndarray,
    xrng: Tuple,
    pathcorr: np.ndarray,
) -> Tuple:
    """Convert value range using a pairwise path correspondence (e.g. obtained
    from time warping techniques).

    **Parameters**\n
    x: 1D array
        Values of the x axis (e.g. time-of-flight values).
    xrng: list/tuple
        Boundary value range on the x axis.
    pathcorr: list/tuple
        Path correspondence between two 1D arrays in the following form,
        [(id_1_trace_1, id_1_trace_2), (id_2_trace_1, id_2_trace_2), ...]

    **Return**\n
    xrange_trans: tuple
        Transformed range according to the path correspondence.
    """

    pathcorr = np.asarray(pathcorr)
    xrange_trans = []

    for xval in xrng:  # Transform each value in the range
        xind = find_nearest(xval, x)
        xind_alt = find_nearest(xind, pathcorr[:, 0])
        xind_trans = pathcorr[xind_alt, 1]
        xrange_trans.append(x[xind_trans])

    return tuple(xrange_trans)


def find_nearest(val: float, narray: np.ndarray) -> int:
    """
    Find the value closest to a given one in a 1D array.

    **Parameters**\n
    val: float
        Value of interest.
    narray: 1D numeric array
        The array to look for the nearest value.

    **Return**\n
    ind: int
        Array index of the value nearest to the given one.
    """

    return int(np.argmin(np.abs(narray - val)))


def peaksearch(
    traces: np.ndarray,
    tof: np.ndarray,
    ranges: List[Tuple] = None,
    pkwindow: int = 3,
    plot: bool = False,
):
    """
    Detect a list of peaks in the corresponding regions of multiple EDCs

    **Parameters**\n
    traces: 2D array
        Collection of EDCs.
    tof: 1D array
        Time-of-flight values.
    ranges: list of tuples/lists | None
        List of ranges for peak detection in the format
        [(LowerBound1, UpperBound1), (LowerBound2, UpperBound2), ....].
    pkwindow: int | 3
        Window width of a peak (amounts to lookahead in ``peakdetect1d``).
    plot: bool | False
        Specify whether to display a custom plot of the peak search results.

    **Returns**\n
    pkmaxs: 1D array
        Collection of peak positions.
    """

    pkmaxs = []
    if plot:
        plt.figure(figsize=(10, 4))

    for rng, trace in zip(ranges, traces.tolist()):

        cond = (tof >= rng[0]) & (tof <= rng[1])
        trace = np.array(trace).ravel()
        tofseg, trseg = tof[cond], trace[cond]
        maxs, _ = peakdetect1d(trseg, tofseg, lookahead=pkwindow)
        try:
            pkmaxs.append(maxs[0, :])
        except IndexError:  # No peak found for this range
            print(f"No peak detected in range {rng}.")
            raise

        if plot:
            plt.plot(tof, trace, "--k", linewidth=1)
            plt.plot(tofseg, trseg, linewidth=2)
            plt.scatter(maxs[0, 0], maxs[0, 1], s=30)

    return np.asarray(pkmaxs)


# 1D peak detection algorithm adapted from Sixten Bergman
# https://gist.github.com/sixtenbe/1178136#file-peakdetect-py
def _datacheck_peakdetect(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input format checking
    """

    if x_axis is None:
        x_axis = np.arange(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise ValueError(
            "Input vectors y_axis and x_axis must have same length",
        )

    # Needs to be a numpy array
    y_axis = np.asarray(y_axis)
    x_axis = np.asarray(x_axis)

    return x_axis, y_axis


def peakdetect1d(
    y_axis: np.ndarray,
    x_axis: np.ndarray = None,
    lookahead: int = 200,
    delta: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively

    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    **Parameters**\n
    y_axis: list
        A list containing the signal over which to find peaks
    x_axis: list | None
        A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
    lookahead: int | 200
        distance to look ahead from a peak candidate to determine if
        it is the actual peak
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value
    delta: numeric | 0
        this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.

    **Returns**\n
    max_peaks: list
        positions of the positive peaks
    min_peaks: list
        positions of the negative peaks
    """

    max_peaks = []
    min_peaks = []
    dump = []  # Used to pop the first hit which almost always is false

    # Check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # Store data length for later use
    length = len(y_axis)

    # Perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")

    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    # maxima and minima candidates are temporarily stored in
    # mx and mn respectively
    _min, _max = np.Inf, -np.Inf

    # Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(
        zip(x_axis[:-lookahead], y_axis[:-lookahead]),
    ):

        if y > _max:
            _max = y
            _max_pos = x

        if y < _min:
            _min = y
            _min_pos = x

        # Find local maxima
        if y < _max - delta and _max != np.Inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index : index + lookahead].max() < _max:

                max_peaks.append([_max_pos, _max])
                dump.append(True)
                # Set algorithm to only find minima now
                _max = np.Inf
                _min = np.Inf

                if index + lookahead >= length:
                    # The end is within lookahead no more peaks can be found
                    break
                continue
            # else:
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        # Find local minima
        if y > _min + delta and _min != -np.Inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index : index + lookahead].min() > _min:

                min_peaks.append([_min_pos, _min])
                dump.append(False)
                # Set algorithm to only find maxima now
                _min = -np.Inf
                _max = -np.Inf

                if index + lookahead >= length:
                    # The end is within lookahead no more peaks can be found
                    break
            # else:
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]

    # Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump

    except IndexError:  # When no peaks have been found
        pass

    return (np.asarray(max_peaks), np.asarray(min_peaks))


def fit_energy_calibation(
    pos: Union[List[float], np.ndarray],
    vals: Union[List[float], np.ndarray],
    binwidth: float,
    binning: int,
    ref_id: int = 0,
    ref_energy: float = None,
    t: Union[List[float], np.ndarray] = None,
    energy_scale: str = "kinetic",
    **kwds,
) -> dict:
    """
    Energy calibration by nonlinear least squares fitting of spectral landmarks on
    a set of (energy dispersion curves (EDCs). This is done here by fitting to the
    function d/(t-t0)**2.

    **Parameters**\n
    pos: list/array
        Positions of the spectral landmarks (e.g. peaks) in the EDCs.
    vals: list/array
        Bias voltage value associated with each EDC.
    refid: int | 0
        Reference dataset index, varies from 0 to vals.size - 1.
    Eref: float | None
        Energy of the reference value.
    t: numeric array | None
        Drift time.
    binwidth: float | 4.125e-12
        time width in ns of a TOF bin
    binning: int | 1
        exponent of binning steps in TOF (i.e. 1 bin=binwidth*2^binning)

    **Returns**\n
    ecalibdict: dict
        A dictionary of fitting parameters including the following,
        :coeffs: Fitted function coefficents.
        :axis: Fitted energy axis.
    """

    vals = np.asarray(vals)
    nvals = vals.size

    if ref_id >= nvals:
        wn.warn(
            "Reference index (refid) cannot be larger than the number of traces!\
                Reset to the largest allowed number.",
        )
        ref_id = nvals - 1

    def residual(pars, time, data, binwidth, binning, energy_scale):
        model = tof2ev(
            pars["d"],
            pars["t0"],
            binwidth,
            binning,
            energy_scale,
            pars["E0"],
            time,
        )
        if data is None:
            return model
        return model - data

    pars = Parameters()
    pars.add(name="d", value=kwds.pop("d_init", 1))
    pars.add(
        name="t0",
        value=kwds.pop("t0_init", 1e-6),
        max=(min(pos) - 1) * binwidth * 2**binning,
    )
    pars.add(name="E0", value=kwds.pop("E0_init", min(vals)))
    fit = Minimizer(
        residual,
        pars,
        fcn_args=(pos, vals, binwidth, binning, energy_scale),
    )
    result = fit.leastsq()
    report_fit(result)

    # Construct the calibrating function
    pfunc = partial(
        tof2ev,
        result.params["d"].value,
        result.params["t0"].value,
        binwidth,
        binning,
        energy_scale,
    )

    # Return results according to specification
    ecalibdict = {}
    ecalibdict["d"] = result.params["d"].value
    ecalibdict["t0"] = result.params["t0"].value
    ecalibdict["E0"] = result.params["E0"].value
    ecalibdict["energy_scale"] = energy_scale

    if (ref_energy is not None) and (t is not None):
        energy_offset = pfunc(-1 * ref_energy, pos[ref_id])
        ecalibdict["axis"] = pfunc(-energy_offset, t)
        ecalibdict["E0"] = -energy_offset
        ecalibdict["refid"] = ref_id

    return ecalibdict


def poly_energy_calibration(
    pos: Union[List[float], np.ndarray],
    vals: Union[List[float], np.ndarray],
    order: int = 3,
    ref_id: int = 0,
    ref_energy: float = None,
    t: Union[List[float], np.ndarray] = None,
    aug: int = 1,
    method: str = "lstsq",
    **kwds,
) -> dict:
    """
    Energy calibration by nonlinear least squares fitting of spectral landmarks on
    a set of (energy dispersion curves (EDCs). This amounts to solving for the
    coefficient vector, a, in the system of equations T.a = b. Here T is the
    differential drift time matrix and b the differential bias vector, and
    assuming that the energy-drift-time relationship can be written in the form,
    E = sum_n (a_n * t**n) + E0

    **Parameters**\n
    pos: list/array
        Positions of the spectral landmarks (e.g. peaks) in the EDCs.
    vals: list/array
        Bias voltage value associated with each EDC.
    order: int | 3
        Polynomial order of the fitting function.
    refid: int | 0
        Reference dataset index, varies from 0 to vals.size - 1.
    ret: str | 'func'
        Return type, including 'func', 'coeffs', 'full', and 'axis' (see below).
    E0: float | None
        Constant energy offset.
    t: numeric array | None
        Drift time.
    aug: int | 1
        Fitting dimension augmentation (1=no change, 2=double, etc).

    **Returns**\n
    pfunc: partial function
        Calibrating function with determined polynomial coefficients
        (except the constant offset).
    ecalibdict: dict
        A dictionary of fitting parameters including the following,
        :coeffs: Fitted polynomial coefficients (the a's).
        :offset: Minimum time-of-flight corresponding to a peak.
        :Tmat: the T matrix (differential time-of-flight) in the equation Ta=b.
        :bvec: the b vector (differential bias) in the fitting Ta=b.
        :axis: Fitted energy axis.
    """

    vals = np.asarray(vals)
    nvals = vals.size

    if ref_id >= nvals:
        wn.warn(
            "Reference index (refid) cannot be larger than the number of traces!\
                Reset to the largest allowed number.",
        )
        ref_id = nvals - 1

    # Top-to-bottom ordering of terms in the T matrix
    termorder = np.delete(range(0, nvals, 1), ref_id)
    termorder = np.tile(termorder, aug)
    # Left-to-right ordering of polynomials in the T matrix
    polyorder = np.linspace(order, 1, order, dtype="int")

    # Construct the T (differential drift time) matrix, Tmat = Tmain - Tsec
    t_main = np.array([pos[ref_id] ** p for p in polyorder])
    # Duplicate to the same order as the polynomials
    t_main = np.tile(t_main, (aug * (nvals - 1), 1))

    t_sec = []

    for term in termorder:
        t_sec.append([pos[term] ** p for p in polyorder])

    t_mat = t_main - np.asarray(t_sec)

    # Construct the b vector (differential bias)
    bvec = vals[ref_id] - np.delete(vals, ref_id)
    bvec = np.tile(bvec, aug)

    # Solve for the a vector (polynomial coefficients) using least squares
    if method == "lstsq":
        sol = lstsq(t_mat, bvec, rcond=None)
    elif method == "lsqr":
        sol = lsqr(t_mat, bvec, **kwds)
    poly_a = sol[0]

    # Construct the calibrating function
    pfunc = partial(tof2evpoly, poly_a)

    # Return results according to specification
    ecalibdict = {}
    ecalibdict["offset"] = np.asarray(pos).min()
    ecalibdict["coeffs"] = poly_a
    ecalibdict["Tmat"] = t_mat
    ecalibdict["bvec"] = bvec

    if ref_energy is not None and t is not None:
        energy_offset = pfunc(-1 * ref_energy, pos[ref_id])
        ecalibdict["axis"] = pfunc(-energy_offset, t)
        ecalibdict["E0"] = -energy_offset
        ecalibdict["refid"] = ref_id

    return ecalibdict


def tof2ev(
    tof_distance: float,
    time_offset: float,
    binwidth: float,
    binning: int,
    energy_scale: str,
    energy_offset: float,
    t: float,
) -> float:
    """
    (d/(t-t0))**2 expression of the time-of-flight to electron volt
    conversion formula.

    Parameters:
    tof_distance: float
        Drift distance
    time_offset: float
        time offset
    energy_offset: float
        Energy offset.
    t: numeric array
        Drift time of electron.

    **Return**\n
    E: numeric array
        Converted energy
    """

    sign = 1 if energy_scale == "kinetic" else -1

    #         m_e/2 [eV]                      bin width [s]
    energy = (
        2.84281e-12
        * sign
        * (tof_distance / (t * binwidth * 2**binning - time_offset)) ** 2
        + energy_offset
    )

    return energy


def tof2evpoly(
    poly_a: Union[List[float], np.ndarray],
    energy_offset: float,
    t: float,
) -> float:
    """
    Polynomial approximation of the time-of-flight to electron volt
    conversion formula.

    **Parameters**\n
    a: 1D array
        Polynomial coefficients.
    E0: float
        Energy offset.
    t: numeric array
        Drift time of electron.

    **Return**\n
    E: numeric array
        Converted energy
    """

    odr = len(poly_a)  # Polynomial order
    poly_a = poly_a[::-1]
    energy = 0.0

    for i, order in enumerate(range(1, odr + 1)):
        energy += poly_a[i] * t**order
    energy += energy_offset

    return energy
