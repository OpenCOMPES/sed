"""sed.calibrator.energy module. Code for energy calibration and
correction. Mostly ported from https://github.com/mpes-kit/mpes.
"""
import itertools as it
import warnings as wn
from copy import deepcopy
from datetime import datetime
from functools import partial
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Literal
from typing import Sequence
from typing import Tuple
from typing import Union

import bokeh.plotting as pbk
import dask.dataframe
import h5py
import ipywidgets as ipw
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
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

from sed.binning import bin_dataframe
from sed.core import dfops
from sed.loader.base.loader import BaseLoader


class EnergyCalibrator:
    """Electron binding energy calibration workflow.

    For the initialization of the EnergyCalibrator class an instance of a
    loader is required. The data can be loaded using the optional arguments,
    or using the load_data method or bin_data method.

    Args:
        loader (BaseLoader): Instance of a loader, subclassed from BaseLoader.
        biases (np.ndarray, optional): Bias voltages used. Defaults to None.
        traces (np.ndarray, optional): TOF-Data traces corresponding to the bias
            values. Defaults to None.
        tof (np.ndarray, optional): TOF-values for the data traces.
            Defaults to None.
        config (dict, optional): Config dictionary. Defaults to None.
    """

    def __init__(
        self,
        loader: BaseLoader,
        biases: np.ndarray = None,
        traces: np.ndarray = None,
        tof: np.ndarray = None,
        config: dict = None,
    ):
        """For the initialization of the EnergyCalibrator class an instance of a
        loader is required. The data can be loaded using the optional arguments,
        or using the load_data method or bin_data method.

        Args:
            loader (BaseLoader): Instance of a loader, subclassed from BaseLoader.
            biases (np.ndarray, optional): Bias voltages used. Defaults to None.
            traces (np.ndarray, optional): TOF-Data traces corresponding to the bias
                values. Defaults to None.
            tof (np.ndarray, optional): TOF-values for the data traces.
                Defaults to None.
            config (dict, optional): Config dictionary. Defaults to None.
        """
        self.loader = loader
        self.biases: np.ndarray = None
        self.traces: np.ndarray = None
        self.traces_normed: np.ndarray = None
        self.tof: np.ndarray = None

        if traces is not None and tof is not None and biases is not None:
            self.load_data(biases=biases, traces=traces, tof=tof)

        if config is None:
            config = {}

        self._config = config

        self.featranges: List[Tuple] = []  # Value ranges for feature detection
        self.peaks: np.ndarray = np.asarray([])
        self.calibration: Dict[str, Any] = self._config["energy"].get("calibration", {})

        self.tof_column = self._config["dataframe"]["tof_column"]
        self.tof_ns_column = self._config["dataframe"].get("tof_ns_column", None)
        self.corrected_tof_column = self._config["dataframe"]["corrected_tof_column"]
        self.energy_column = self._config["dataframe"]["energy_column"]
        self.x_column = self._config["dataframe"]["x_column"]
        self.y_column = self._config["dataframe"]["y_column"]
        self.binwidth: float = self._config["dataframe"]["tof_binwidth"]
        self.binning: int = self._config["dataframe"]["tof_binning"]
        self.x_width = self._config["energy"]["x_width"]
        self.y_width = self._config["energy"]["y_width"]
        self.tof_width = np.asarray(
            self._config["energy"]["tof_width"],
        ) / 2 ** (self.binning - 1)
        self.tof_fermi = self._config["energy"]["tof_fermi"] / 2 ** (self.binning - 1)
        self.color_clip = self._config["energy"]["color_clip"]
        self.sector_delays = self._config["dataframe"].get("sector_delays", None)
        self.sector_id_column = self._config["dataframe"].get("sector_id_column", None)
        self.offsets: Dict[str, Any] = self._config["energy"].get("offsets", {})
        self.correction: Dict[str, Any] = self._config["energy"].get("correction", {})

    @property
    def ntraces(self) -> int:
        """Property returning the number of traces.

        Returns:
            int: The number of loaded/calculated traces.
        """
        return len(self.traces)

    @property
    def nranges(self) -> int:
        """Property returning the number of specified feature ranges which Can be a
        multiple of ntraces.

        Returns:
            int: The number of specified feature ranges.
        """
        return len(self.featranges)

    @property
    def dup(self) -> int:
        """Property returning the duplication number, i.e. the number of feature
        ranges per trace.

        Returns:
            int: The duplication number.
        """
        return int(np.round(self.nranges / self.ntraces))

    def load_data(
        self,
        biases: np.ndarray = None,
        traces: np.ndarray = None,
        tof: np.ndarray = None,
    ):
        """Load data into the class. Not provided parameters will be overwritten by
        empty arrays.

        Args:
            biases (np.ndarray, optional): Bias voltages used. Defaults to None.
            traces (np.ndarray, optional): TOF-Data traces corresponding to the bias
                values. Defaults to None.
            tof (np.ndarray, optional): TOF-values for the data traces.
                Defaults to None.
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
        """Bin data from single-event files, and load into class.

        Args:
            data_files (List[str]): list of file names to bin
            axes (List[str], optional): bin axes. Defaults to
                config["dataframe"]["tof_column"].
            bins (List[int], optional): number of bins.
                Defaults to config["energy"]["bins"].
            ranges (Sequence[Tuple[float, float]], optional): bin ranges.
                Defaults to config["energy"]["ranges"].
            biases (np.ndarray, optional): Bias voltages used.
                If not provided, biases are extracted from the file meta data.
            bias_key (str, optional): hdf5 path where bias values are stored.
                Defaults to config["energy"]["bias_key"].
            **kwds: Keyword parameters for bin_dataframe
        """
        if axes is None:
            axes = [self.tof_column]
        if bins is None:
            bins = [self._config["energy"]["bins"]]
        if ranges is None:
            ranges_ = [
                np.array(self._config["energy"]["ranges"]) / 2 ** (self.binning - 1),
            ]
            ranges = [cast(Tuple[float, float], tuple(v)) for v in ranges_]
        # pylint: disable=duplicate-code
        hist_mode = kwds.pop("hist_mode", self._config["binning"]["hist_mode"])
        mode = kwds.pop("mode", self._config["binning"]["mode"])
        pbar = kwds.pop("pbar", self._config["binning"]["pbar"])
        try:
            num_cores = kwds.pop("num_cores", self._config["binning"]["num_cores"])
        except KeyError:
            num_cores = psutil.cpu_count() - 1
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
                try:
                    bias_key = self._config["energy"]["bias_key"]
                except KeyError as exc:
                    raise ValueError(
                        "Either Bias Values or a valid bias_key has to be present!",
                    ) from exc

        dataframe, _, _ = self.loader.read_dataframe(
            files=data_files,
            collect_metadata=False,
        )
        traces = bin_dataframe(
            dataframe,
            bins=bins,
            axes=axes,
            ranges=ranges,
            hist_mode=hist_mode,
            mode=mode,
            pbar=pbar,
            n_cores=num_cores,
            threads_per_worker=threads_per_worker,
            threadpool_api=threadpool_api,
            return_partitions=True,
            **kwds,
        )
        if read_biases:
            if bias_key:
                try:
                    biases = extract_bias(data_files, bias_key)
                except KeyError as exc:
                    raise ValueError(
                        "Either Bias Values or a valid bias_key has to be present!",
                    ) from exc
        tof = traces.coords[(axes[0])]
        self.traces = self.traces_normed = np.asarray(traces.T)
        self.tof = np.asarray(tof)
        self.biases = np.asarray(biases)

    def normalize(self, smooth: bool = False, span: int = 7, order: int = 1):
        """Normalize the spectra along an axis.

        Args:
            smooth (bool, optional): Option to smooth the signals before normalization.
                Defaults to False.
            span (int, optional): span smoothing parameters of the LOESS method
                (see ``scipy.signal.savgol_filter()``). Defaults to 7.
            order (int, optional): order smoothing parameters of the LOESS method
                (see ``scipy.signal.savgol_filter()``). Defaults to 1.
        """
        self.traces_normed = normspec(
            self.traces,
            smooth=smooth,
            span=span,
            order=order,
        )

    def adjust_ranges(
        self,
        ranges: Tuple,
        ref_id: int = 0,
        traces: np.ndarray = None,
        peak_window: int = 7,
        apply: bool = False,
        **kwds,
    ):
        """Display a tool to select or extract the equivalent feature ranges
        (containing the peaks) among all traces.

        Args:
            ranges (Tuple):
                Collection of feature detection ranges, within which an algorithm
                (i.e. 1D peak detector) with look for the feature.
            ref_id (int, optional): Index of the reference trace. Defaults to 0.
            traces (np.ndarray, optional): Collection of energy dispersion curves.
                Defaults to self.traces_normed.
            peak_window (int, optional): area around a peak to check for other peaks.
                Defaults to 7.
            apply (bool, optional): Option to directly apply the provided parameters.
                Defaults to False.
            **kwds:
                keyword arguments for trace alignment (see ``find_correspondence()``).
        """
        if traces is None:
            traces = self.traces_normed

        self.add_ranges(
            ranges=ranges,
            ref_id=ref_id,
            traces=traces,
            infer_others=True,
            mode="replace",
        )
        self.feature_extract(peak_window=peak_window)

        # make plot
        labels = kwds.pop("labels", [str(b) + " V" for b in self.biases])
        figsize = kwds.pop("figsize", (8, 4))
        plot_segs = []
        plot_peaks = []
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.get_cmap("rainbow")(np.linspace(0, 1, len(traces)))
        for itr, color in zip(range(len(traces)), colors):
            trace = traces[itr, :]
            # main traces
            ax.plot(
                self.tof,
                trace,
                ls="-",
                color=color,
                linewidth=1,
                label=labels[itr],
            )
            # segments:
            seg = self.featranges[itr]
            cond = (self.tof >= seg[0]) & (self.tof <= seg[1])
            tofseg, traceseg = self.tof[cond], trace[cond]
            (line,) = ax.plot(
                tofseg,
                traceseg,
                ls="-",
                color=color,
                linewidth=3,
            )
            plot_segs.append(line)
            # markers
            (scatt,) = ax.plot(
                self.peaks[itr, 0],
                self.peaks[itr, 1],
                ls="",
                marker=".",
                color="k",
                markersize=10,
            )
            plot_peaks.append(scatt)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title("")

        def update(refid, ranges):
            self.add_ranges(ranges, refid, traces=traces)
            self.feature_extract(peak_window=7)
            for itr, _ in enumerate(self.traces_normed):
                seg = self.featranges[itr]
                cond = (self.tof >= seg[0]) & (self.tof <= seg[1])
                tofseg, traceseg = (
                    self.tof[cond],
                    self.traces_normed[itr][cond],
                )
                plot_segs[itr].set_ydata(traceseg)
                plot_segs[itr].set_xdata(tofseg)

                plot_peaks[itr].set_xdata(self.peaks[itr, 0])
                plot_peaks[itr].set_ydata(self.peaks[itr, 1])

            fig.canvas.draw_idle()

        refid_slider = ipw.IntSlider(
            value=ref_id,
            min=0,
            max=10,
            step=1,
        )

        ranges_slider = ipw.IntRangeSlider(
            value=list(ranges),
            min=min(self.tof),
            max=max(self.tof),
            step=1,
        )

        update(ranges=ranges, refid=ref_id)

        ipw.interact(
            update,
            refid=refid_slider,
            ranges=ranges_slider,
        )

        def apply_func(apply: bool):  # noqa: ARG001
            self.add_ranges(
                ranges_slider.value,
                refid_slider.value,
                traces=self.traces_normed,
            )
            self.feature_extract(peak_window=7)
            ranges_slider.close()
            refid_slider.close()
            apply_button.close()

        apply_button = ipw.Button(description="apply")
        display(apply_button)  # pylint: disable=duplicate-code
        apply_button.on_click(apply_func)
        plt.show()

        if apply:
            apply_func(True)

    def add_ranges(
        self,
        ranges: Union[List[Tuple], Tuple],
        ref_id: int = 0,
        traces: np.ndarray = None,
        infer_others: bool = True,
        mode: str = "replace",
        **kwds,
    ):
        """Select or extract the equivalent feature ranges (containing the peaks) among all traces.

        Args:
            ranges (Union[List[Tuple], Tuple]):
                Collection of feature detection ranges, within which an algorithm
                (i.e. 1D peak detector) with look for the feature.
            ref_id (int, optional): Index of the reference trace. Defaults to 0.
            traces (np.ndarray, optional): Collection of energy dispersion curves.
                Defaults to self.traces_normed.
            infer_others (bool, optional): Option to infer the feature detection range
                in other traces from a given one using a time warp algorthm.
                Defaults to True.
            mode (str, optional): Specification on how to change the feature ranges
                ('append' or 'replace'). Defaults to "replace".
            **kwds:
                keyword arguments for trace alignment (see ``find_correspondence()``).
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

        Args:
            ranges (List[Tuple], optional):  List of ranges in each trace to look for
                the peak feature, [start, end]. Defaults to self.featranges.
            traces (np.ndarray, optional): Collection of 1D spectra to use for
                calibration. Defaults to self.traces_normed.
            peak_window (int, optional): area around a peak to check for other peaks.
                Defaults to 7.
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
        verbose: bool = True,
        **kwds,
    ) -> dict:
        """Calculate the functional mapping between time-of-flight and the energy
        scale using optimization methods.

        Args:
            ref_id (int, optional): The reference trace index (an integer).
                Defaults to 0.
            method (str, optional):  Method for determining the energy calibration.

                - **'lmfit'**: Energy calibration using lmfit and 1/t^2 form.
                - **'lstsq'**, **'lsqr'**: Energy calibration using polynomial form.

                Defaults to 'lmfit'.
            energy_scale (str, optional): Direction of increasing energy scale.

                - **'kinetic'**: increasing energy with decreasing TOF.
                - **'binding'**: increasing energy with increasing TOF.

                Defaults to "kinetic".
            landmarks (np.ndarray, optional): Extracted peak positions (TOF) used for
                calibration. Defaults to self.peaks.
            biases (np.ndarray, optional): Bias values. Defaults to self.biases.
            t (np.ndarray, optional): TOF values. Defaults to self.tof.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to True.
            **kwds: keyword arguments.
                See available keywords for ``poly_energy_calibration()`` and
                ``fit_energy_calibration()``

        Raises:
            ValueError: Raised if invalid 'energy_scale' is passed.
            NotImplementedError: Raised if invalid 'method' is passed.

        Returns:
            dict: Calibration dictionary with coefficients.
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
            self.calibration = fit_energy_calibration(
                landmarks,
                sign * biases,
                binwidth,
                binning,
                ref_id=ref_id,
                t=t,
                energy_scale=energy_scale,
                verbose=verbose,
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
                energy_scale=energy_scale,
                **kwds,
            )
        else:
            raise NotImplementedError()

        self.calibration["creation_date"] = datetime.now().timestamp()
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

        Args:
            traces (np.ndarray): Matrix of traces to visualize.
            segs (List[Tuple], optional): Segments to be highlighted in the
                visualization. Defaults to None.
            peaks (np.ndarray, optional): Peak positions for labelling the traces.
                Defaults to None.
            show_legend (bool, optional): Option to display bias voltages as legends.
                Defaults to True.
            backend (str, optional): Backend specification, choose between 'matplotlib'
                (static) or 'bokeh' (interactive). Defaults to "matplotlib".
            linekwds (dict, optional): Keyword arguments for line plotting
                (see ``matplotlib.pyplot.plot()``). Defaults to {}.
            linesegkwds (dict, optional): Keyword arguments for line segments plotting
                (see ``matplotlib.pyplot.plot()``). Defaults to {}.
            scatterkwds (dict, optional): Keyword arguments for scatter plot
                (see ``matplotlib.pyplot.scatter()``). Defaults to {}.
            legkwds (dict, optional): Keyword arguments for legend
                (see ``matplotlib.pyplot.legend()``). Defaults to {}.
            **kwds: keyword arguments:

                - **labels** (list): Labels for each curve
                - **xaxis** (np.ndarray): x (horizontal) axis values
                - **title** (str): Title of the plot
                - **legend_location** (str): Location of the plot legend
                - **align** (bool): Option to shift traces by bias voltage
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
                        xaxis + sign * (self.biases[itr] - self.biases[self.calibration["refid"]]),
                        trace,
                        ls="-",
                        linewidth=1,
                        label=lbs[itr],
                        **linekwds,
                    )
                else:
                    ax.plot(
                        xaxis,
                        trace,
                        ls="-",
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
                        ls="-",
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
                width=figsize[0],
                height=figsize[1],
                tooltips=ttp,
            )
            # Plotting the main traces
            for itr, color in zip(range(len(traces)), colors):
                trace = traces[itr, :]
                if align:
                    fig.line(
                        xaxis + sign * (self.biases[itr] - self.biases[self.calibration["refid"]]),
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
        calibration: dict = None,
        verbose: bool = True,
        **kwds,
    ) -> Tuple[Union[pd.DataFrame, dask.dataframe.DataFrame], dict]:
        """Calculate and append the energy axis to the events dataframe.

        Args:
            df (Union[pd.DataFrame, dask.dataframe.DataFrame]):
                Dataframe to apply the energy axis calibration to.
            tof_column (str, optional): Label of the source column.
                Defaults to config["dataframe"]["tof_column"].
            energy_column (str, optional): Label of the destination column.
                Defaults to config["dataframe"]["energy_column"].
            calibration (dict, optional): Calibration dictionary. If provided,
                overrides calibration from class or config.
                Defaults to self.calibration or config["energy"]["calibration"].
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to True.
            **kwds: additional keyword arguments for the energy conversion. They are
                added to the calibration dictionary.

        Raises:
            ValueError: Raised if expected calibration parameters are missing.
            NotImplementedError: Raised if an invalid calib_type is found.

        Returns:
            Union[pd.DataFrame, dask.dataframe.DataFrame]: dataframe with added column
            and energy calibration metadata dictionary.
        """
        if tof_column is None:
            if self.corrected_tof_column in df.columns:
                tof_column = self.corrected_tof_column
            else:
                tof_column = self.tof_column

        if energy_column is None:
            energy_column = self.energy_column

        binwidth = kwds.pop("binwidth", self.binwidth)
        binning = kwds.pop("binning", self.binning)

        # pylint: disable=duplicate-code
        if calibration is None:
            calibration = deepcopy(self.calibration)

        if len(kwds) > 0:
            for key, value in kwds.items():
                calibration[key] = value
            calibration["creation_date"] = datetime.now().timestamp()

        elif "creation_date" in calibration and verbose:
            datestring = datetime.fromtimestamp(calibration["creation_date"]).strftime(
                "%m/%d/%Y, %H:%M:%S",
            )
            print(f"Using energy calibration parameters generated on {datestring}")

        # try to determine calibration type if not provided
        if "calib_type" not in calibration:
            if "t0" in calibration and "d" in calibration and "E0" in calibration:
                calibration["calib_type"] = "fit"
                if "energy_scale" not in calibration:
                    calibration["energy_scale"] = "kinetic"

            elif "coeffs" in calibration and "E0" in calibration:
                calibration["calib_type"] = "poly"
            else:
                raise ValueError("No valid calibration parameters provided!")

        if calibration["calib_type"] == "fit":
            # Fitting metadata for nexus
            calibration["fit_function"] = "(a0/(x0-a1))**2 + a2"
            calibration["coefficients"] = np.array(
                [
                    calibration["d"],
                    calibration["t0"],
                    calibration["E0"],
                ],
            )
            df[energy_column] = tof2ev(
                calibration["d"],
                calibration["t0"],
                binwidth,
                binning,
                calibration["energy_scale"],
                calibration["E0"],
                df[tof_column].astype("float64"),
            )
        elif calibration["calib_type"] == "poly":
            # Fitting metadata for nexus
            fit_function = "a0"
            for term in range(1, len(calibration["coeffs"]) + 1):
                fit_function += f" + a{term}*x0**{term}"
            calibration["fit_function"] = fit_function
            calibration["coefficients"] = np.concatenate(
                (calibration["coeffs"], [calibration["E0"]]),
            )[::-1]
            df[energy_column] = tof2evpoly(
                calibration["coeffs"],
                calibration["E0"],
                df[tof_column].astype("float64"),
            )
        else:
            raise NotImplementedError

        metadata = self.gather_calibration_metadata(calibration)

        return df, metadata

    def append_tof_ns_axis(
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        tof_column: str = None,
        tof_ns_column: str = None,
        **kwds,
    ) -> Tuple[Union[pd.DataFrame, dask.dataframe.DataFrame], dict]:
        """Converts the time-of-flight time from steps to time in ns.

        Args:
            df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to convert.
            tof_column (str, optional): Name of the column containing the
                time-of-flight steps. Defaults to config["dataframe"]["tof_column"].
            tof_ns_column (str, optional): Name of the column to store the
                time-of-flight in nanoseconds. Defaults to config["dataframe"]["tof_ns_column"].
            binwidth (float, optional): Time-of-flight binwidth in ns.
                Defaults to config["energy"]["tof_binwidth"].
            binning (int, optional): Time-of-flight binning factor.
                Defaults to config["energy"]["tof_binning"].

        Returns:
            dask.dataframe.DataFrame: Dataframe with the new columns.
            dict: Metadata dictionary.
        """
        binwidth = kwds.pop("binwidth", self.binwidth)
        binning = kwds.pop("binning", self.binning)
        if tof_column is None:
            if self.corrected_tof_column in df.columns:
                tof_column = self.corrected_tof_column
            else:
                tof_column = self.tof_column

        if tof_ns_column is None:
            tof_ns_column = self.tof_ns_column

        df[tof_ns_column] = tof2ns(
            binwidth,
            binning,
            df[tof_column].astype("float64"),
        )
        metadata: Dict[str, Any] = {
            "applied": True,
            "binwidth": binwidth,
            "binning": binning,
        }
        return df, metadata

    def gather_calibration_metadata(self, calibration: dict = None) -> dict:
        """Collects metadata from the energy calibration

        Args:
            calibration (dict, optional): Dictionary with energy calibration
                parameters. Defaults to None.

        Returns:
            dict: Generated metadata dictionary.
        """
        if calibration is None:
            calibration = self.calibration
        metadata: Dict[Any, Any] = {}
        metadata["applied"] = True
        metadata["calibration"] = deepcopy(calibration)
        metadata["tof"] = deepcopy(self.tof)
        # create empty calibrated axis entry, if it is not present.
        if "axis" not in metadata["calibration"]:
            metadata["calibration"]["axis"] = 0

        return metadata

    def adjust_energy_correction(
        self,
        image: xr.DataArray,
        correction_type: str = None,
        amplitude: float = None,
        center: Tuple[float, float] = None,
        correction: dict = None,
        apply: bool = False,
        **kwds,
    ):
        """Visualize the energy correction function on top of the TOF/X/Y graphs.

        Args:
            image (xr.DataArray): Image data cube (x, y, tof) of binned data to plot.
            correction_type (str, optional): Type of correction to apply to the TOF
                axis. Valid values are:

                - 'spherical'
                - 'Lorentzian'
                - 'Gaussian'
                - 'Lorentzian_asymmetric'

                Defaults to config["energy"]["correction_type"].
            amplitude (float, optional): Amplitude of the time-of-flight correction
                term. Defaults to config["energy"]["correction"]["correction_type"].
            center (Tuple[float, float], optional): Center (x/y) coordinates for the
                correction. Defaults to config["energy"]["correction"]["center"].
            correction (dict, optional): Correction dict. Defaults to the config values
                and is updated from provided and adjusted parameters.
            apply (bool, optional): whether to store the provided parameters within
                the class. Defaults to False.
            **kwds: Additional parameters to use for the adjustment plots:

                - **x_column** (str): Name of the x column.
                - **y_column** (str): Name of the y column.
                - **tof_column** (str): Name of the tog column to convert.
                - **x_width** (int, int): x range to integrate around the center
                - **y_width** (int, int): y range to integrate around the center
                - **tof_fermi** (int): TOF value of the Fermi level
                - **tof_width** (int, int): TOF range to plot around tof_fermi
                - **color_clip** (int): highest value to plot in the color range

                Additional parameters for the correction functions:

                - **d** (float): Field-free drift distance.
                - **gamma** (float): Linewidth value for correction using a 2D
                  Lorentz profile.
                - **sigma** (float): Standard deviation for correction using a 2D
                  Gaussian profile.
                - **gamma2** (float): Linewidth value for correction using an
                  asymmetric 2D Lorentz profile, X-direction.
                - **amplitude2** (float): Amplitude value for correction using an
                  asymmetric 2D Lorentz profile, X-direction.

        Raises:
            NotImplementedError: Raised for invalid correction_type.
        """
        matplotlib.use("module://ipympl.backend_nbagg")

        if correction is None:
            correction = deepcopy(self.correction)

        if correction_type is not None:
            correction["correction_type"] = correction_type

        if amplitude is not None:
            correction["amplitude"] = amplitude

        if center is not None:
            correction["center"] = center

        x_column = kwds.pop("x_column", self.x_column)
        y_column = kwds.pop("y_column", self.y_column)
        tof_column = kwds.pop("tof_column", self.tof_column)
        x_width = kwds.pop("x_width", self.x_width)
        y_width = kwds.pop("y_width", self.y_width)
        tof_fermi = kwds.pop("tof_fermi", self.tof_fermi)
        tof_width = kwds.pop("tof_width", self.tof_width)
        color_clip = kwds.pop("color_clip", self.color_clip)

        correction = {**correction, **kwds}

        if not {"correction_type", "amplitude", "center"}.issubset(set(correction.keys())):
            raise ValueError(
                "No valid energy correction found in config and required parameters missing!",
            )

        if isinstance(correction["center"], list):
            correction["center"] = tuple(correction["center"])

        x = image.coords[x_column].values
        y = image.coords[y_column].values

        x_center = correction["center"][0]
        y_center = correction["center"][1]

        correction_x = tof_fermi - correction_function(
            x=x,
            y=y_center,
            **correction,
        )
        correction_y = tof_fermi - correction_function(
            x=x_center,
            y=y,
            **correction,
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
        line1 = ax[0].axvline(x=x_center)
        (trace2,) = ax[1].plot(y, correction_y)
        line2 = ax[1].axvline(x=y_center)

        amplitude_slider = ipw.FloatSlider(
            value=correction["amplitude"],
            min=0,
            max=10,
            step=0.1,
        )
        x_center_slider = ipw.FloatSlider(
            value=x_center,
            min=0,
            max=self._config["momentum"]["detector_ranges"][0][1],
            step=1,
        )
        y_center_slider = ipw.FloatSlider(
            value=y_center,
            min=0,
            max=self._config["momentum"]["detector_ranges"][1][1],
            step=1,
        )

        def update(amplitude, x_center, y_center, **kwds):
            nonlocal correction
            correction["amplitude"] = amplitude
            correction["center"] = (x_center, y_center)
            correction = {**correction, **kwds}
            correction_x = tof_fermi - correction_function(
                x=x,
                y=y_center,
                **correction,
            )
            correction_y = tof_fermi - correction_function(
                x=x_center,
                y=y,
                **correction,
            )

            trace1.set_ydata(correction_x)
            line1.set_xdata(x=x_center)
            trace2.set_ydata(correction_y)
            line2.set_xdata(x=y_center)

            fig.canvas.draw_idle()

        def common_apply_func(apply: bool):  # noqa: ARG001
            self.correction = {}
            self.correction["amplitude"] = correction["amplitude"]
            self.correction["center"] = correction["center"]
            self.correction["correction_type"] = correction["correction_type"]
            self.correction["creation_date"] = datetime.now().timestamp()
            amplitude_slider.close()
            x_center_slider.close()
            y_center_slider.close()
            apply_button.close()

        if correction["correction_type"] == "spherical":
            try:
                update(correction["amplitude"], x_center, y_center, diameter=correction["diameter"])
            except KeyError as exc:
                raise ValueError(
                    "Parameter 'diameter' required for correction type 'sperical', ",
                    "but not present!",
                ) from exc

            diameter_slider = ipw.FloatSlider(
                value=correction["diameter"],
                min=0,
                max=10000,
                step=100,
            )

            ipw.interact(
                update,
                amplitude=amplitude_slider,
                x_center=x_center_slider,
                y_center=y_center_slider,
                diameter=diameter_slider,
            )

            def apply_func(apply: bool):
                common_apply_func(apply)
                self.correction["diameter"] = correction["diameter"]
                diameter_slider.close()

        elif correction["correction_type"] == "Lorentzian":
            try:
                update(correction["amplitude"], x_center, y_center, gamma=correction["gamma"])
            except KeyError as exc:
                raise ValueError(
                    "Parameter 'gamma' required for correction type 'Lorentzian', but not present!",
                ) from exc

            gamma_slider = ipw.FloatSlider(
                value=correction["gamma"],
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

            def apply_func(apply: bool):
                common_apply_func(apply)
                self.correction["gamma"] = correction["gamma"]
                gamma_slider.close()

        elif correction["correction_type"] == "Gaussian":
            try:
                update(correction["amplitude"], x_center, y_center, sigma=correction["sigma"])
            except KeyError as exc:
                raise ValueError(
                    "Parameter 'sigma' required for correction type 'Gaussian', but not present!",
                ) from exc

            sigma_slider = ipw.FloatSlider(
                value=correction["sigma"],
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

            def apply_func(apply: bool):
                common_apply_func(apply)
                self.correction["sigma"] = correction["sigma"]
                sigma_slider.close()

        elif correction["correction_type"] == "Lorentzian_asymmetric":
            try:
                if "amplitude2" not in correction:
                    correction["amplitude2"] = correction["amplitude"]
                if "sigma2" not in correction:
                    correction["gamma2"] = correction["gamma"]
                update(
                    correction["amplitude"],
                    x_center,
                    y_center,
                    gamma=correction["gamma"],
                    amplitude2=correction["amplitude2"],
                    gamma2=correction["gamma2"],
                )
            except KeyError as exc:
                raise ValueError(
                    "Parameter 'gamma' required for correction type 'Lorentzian_asymmetric', ",
                    "but not present!",
                ) from exc

            gamma_slider = ipw.FloatSlider(
                value=correction["gamma"],
                min=0,
                max=2000,
                step=1,
            )

            amplitude2_slider = ipw.FloatSlider(
                value=correction["amplitude2"],
                min=0,
                max=10,
                step=0.1,
            )

            gamma2_slider = ipw.FloatSlider(
                value=correction["gamma2"],
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

            def apply_func(apply: bool):
                common_apply_func(apply)
                self.correction["gamma"] = correction["gamma"]
                self.correction["amplitude2"] = correction["amplitude2"]
                self.correction["gamma2"] = correction["gamma2"]
                gamma_slider.close()
                amplitude2_slider.close()
                gamma2_slider.close()

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
        tof_column: str = None,
        new_tof_column: str = None,
        correction_type: str = None,
        amplitude: float = None,
        correction: dict = None,
        verbose: bool = True,
        **kwds,
    ) -> Tuple[Union[pd.DataFrame, dask.dataframe.DataFrame], dict]:
        """Apply correction to the time-of-flight (TOF) axis of single-event data.

        Args:
            df (Union[pd.DataFrame, dask.dataframe.DataFrame]): The dataframe where
                to apply the energy correction to.
            tof_column (str, optional): Name of the source column to convert.
                Defaults to config["dataframe"]["tof_column"].
            new_tof_column (str, optional): Name of the destination column to convert.
                Defaults to config["dataframe"]["corrected_tof_column"].
            correction_type (str, optional): Type of correction to apply to the TOF
                axis. Valid values are:

                - 'spherical'
                - 'Lorentzian'
                - 'Gaussian'
                - 'Lorentzian_asymmetric'

                Defaults to config["energy"]["correction_type"].
            amplitude (float, optional): Amplitude of the time-of-flight correction
                term. Defaults to config["energy"]["correction"]["correction_type"].
            correction (dict, optional): Correction dictionary containing paramters
                for the correction. Defaults to self.correction or
                config["energy"]["correction"].
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to True.
            **kwds: Additional parameters to use for the correction:

                - **x_column** (str): Name of the x column.
                - **y_column** (str): Name of the y column.
                - **d** (float): Field-free drift distance.
                - **gamma** (float): Linewidth value for correction using a 2D
                  Lorentz profile.
                - **sigma** (float): Standard deviation for correction using a 2D
                  Gaussian profile.
                - **gamma2** (float): Linewidth value for correction using an
                  asymmetric 2D Lorentz profile, X-direction.
                - **amplitude2** (float): Amplitude value for correction using an
                  asymmetric 2D Lorentz profile, X-direction.

        Returns:
            Union[pd.DataFrame, dask.dataframe.DataFrame]: dataframe with added column
            and Energy correction metadata dictionary.
        """
        if correction is None:
            correction = deepcopy(self.correction)

        x_column = kwds.pop("x_column", self.x_column)
        y_column = kwds.pop("y_column", self.y_column)

        if tof_column is None:
            tof_column = self.tof_column

        if new_tof_column is None:
            new_tof_column = self.corrected_tof_column

        if correction_type is not None or amplitude is not None or len(kwds) > 0:
            if correction_type is not None:
                correction["correction_type"] = correction_type

            if amplitude is not None:
                correction["amplitude"] = amplitude

            for key, value in kwds.items():
                correction[key] = value

            correction["creation_date"] = datetime.now().timestamp()

        elif "creation_date" in correction and verbose:
            datestring = datetime.fromtimestamp(correction["creation_date"]).strftime(
                "%m/%d/%Y, %H:%M:%S",
            )
            print(f"Using energy correction parameters generated on {datestring}")

        missing_keys = {"correction_type", "center", "amplitude"} - set(correction.keys())
        if missing_keys:
            raise ValueError(f"Required correction parameters '{missing_keys}' missing!")

        df[new_tof_column] = df[tof_column] + correction_function(
            x=df[x_column],
            y=df[y_column],
            **correction,
        )
        metadata = self.gather_correction_metadata(correction=correction)

        return df, metadata

    def gather_correction_metadata(self, correction: dict = None) -> dict:
        """Collect meta data for energy correction

        Args:
            correction (dict, optional): Dictionary with energy correction parameters.
                Defaults to None.

        Returns:
            dict: Generated metadata dictionary.
        """
        if correction is None:
            correction = self.correction
        metadata: Dict[Any, Any] = {}
        metadata["applied"] = True
        metadata["correction"] = deepcopy(correction)

        return metadata

    def align_dld_sectors(
        self,
        df: dask.dataframe.DataFrame,
        tof_column: str = None,
        sector_id_column: str = None,
        sector_delays: np.ndarray = None,
    ) -> Tuple[dask.dataframe.DataFrame, dict]:
        """Aligns the time-of-flight axis of the different sections of a detector.

        Args:
            df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to use.
            tof_column (str, optional): Name of the column containing the time-of-flight values.
                Defaults to config["dataframe"]["tof_column"].
            sector_id_column (str, optional): Name of the column containing the sector id values.
                Defaults to config["dataframe"]["sector_id_column"].
            sector_delays (np.ndarray, optional): Array containing the sector delays. Defaults to
                config["dataframe"]["sector_delays"].

        Returns:
            dask.dataframe.DataFrame: Dataframe with the new columns.
            dict: Metadata dictionary.
        """
        if sector_delays is None:
            sector_delays = self.sector_delays
        if sector_id_column is None:
            sector_id_column = self.sector_id_column

        if sector_delays is None or sector_id_column is None:
            raise ValueError(
                "No value for sector_delays or sector_id_column found in config."
                "Config file is not properly configured for dld sector correction.",
            )
        tof_column = tof_column or self.tof_column

        # align the 8s sectors
        sector_delays_arr = dask.array.from_array(sector_delays)

        def align_sector(x):
            val = x[tof_column] - sector_delays_arr[x[sector_id_column].values.astype(int)]
            return val.astype(np.float32)

        df[tof_column] = df.map_partitions(align_sector, meta=(tof_column, np.float32))
        metadata: Dict[str, Any] = {
            "applied": True,
            "sector_delays": sector_delays,
        }
        return df, metadata

    def add_offsets(
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame] = None,
        offsets: Dict[str, Any] = None,
        constant: float = None,
        columns: Union[str, Sequence[str]] = None,
        weights: Union[float, Sequence[float]] = None,
        preserve_mean: Union[bool, Sequence[bool]] = False,
        reductions: Union[str, Sequence[str]] = None,
        energy_column: str = None,
        verbose: bool = True,
    ) -> Tuple[Union[pd.DataFrame, dask.dataframe.DataFrame], dict]:
        """Apply an offset to the energy column by the values of the provided columns.

        If no parameter is passed to this function, the offset is applied as defined in the
        config file. If parameters are passed, they are used to generate a new offset dictionary
        and the offset is applied using the ``dfops.apply_offset_from_columns()`` function.

        Args:
            df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to use.
            offsets (Dict, optional): Dictionary of energy offset parameters.
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
            energy_column (str, optional): Name of the column containing the energy values.
            verbose (bool, optional): Option to print out diagnostic information.
                Defaults to True.

        Returns:
            dask.dataframe.DataFrame: Dataframe with the new columns.
            dict: Metadata dictionary.
        """
        if offsets is None:
            offsets = deepcopy(self.offsets)

        if energy_column is None:
            energy_column = self.energy_column

        metadata: Dict[str, Any] = {
            "applied": True,
        }

        # flip sign for binding energy scale
        energy_scale = self.calibration.get("energy_scale", None)
        if energy_scale is None:
            raise ValueError("Energy scale not set. Cannot interpret the sign of the offset.")
        if energy_scale not in ["binding", "kinetic"]:
            raise ValueError(f"Invalid energy scale: {energy_scale}")
        scale_sign: Literal[-1, 1] = -1 if energy_scale == "binding" else 1

        if columns is not None or constant is not None:
            # pylint:disable=duplicate-code
            # use passed parameters, overwrite config
            offsets = {}
            offsets["creation_date"] = datetime.now().timestamp()
            # column-based offsets
            if columns is not None:
                if weights is None:
                    weights = 1
                if isinstance(weights, (int, float, np.integer, np.floating)):
                    weights = [weights]
                if len(weights) == 1:
                    weights = [weights[0]] * len(columns)
                if not isinstance(weights, Sequence):
                    raise TypeError(f"Invalid type for weights: {type(weights)}")
                if not all(isinstance(s, (int, float, np.integer, np.floating)) for s in weights):
                    raise TypeError(f"Invalid type for weights: {type(weights)}")

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
                    offsets[col] = {
                        "weight": weight,
                        "preserve_mean": pmean,
                        "reduction": red,
                    }

            # constant offset
            if isinstance(constant, (int, float, np.integer, np.floating)):
                offsets["constant"] = constant
            elif constant is not None:
                raise TypeError(f"Invalid type for constant: {type(constant)}")

        elif "creation_date" in offsets and verbose:
            datestring = datetime.fromtimestamp(offsets["creation_date"]).strftime(
                "%m/%d/%Y, %H:%M:%S",
            )
            print(f"Using energy offset parameters generated on {datestring}")

        if len(offsets) > 0:
            # unpack dictionary
            # pylint: disable=duplicate-code
            columns = []
            weights = []
            preserve_mean = []
            reductions = []
            if verbose:
                print("Energy offset parameters:")
            for k, v in offsets.items():
                if k == "creation_date":
                    continue
                if k == "constant":
                    # flip sign if binding energy scale
                    constant = v * scale_sign
                    if verbose:
                        print(f"   Constant: {constant} ")
                else:
                    columns.append(k)
                    try:
                        weight = v["weight"]
                    except KeyError:
                        weight = 1
                    if not isinstance(weight, (int, float, np.integer, np.floating)):
                        raise TypeError(f"Invalid type for weight of column {k}: {type(weight)}")
                    # flip sign if binding energy scale
                    weight = weight * scale_sign
                    weights.append(weight)
                    pm = v.get("preserve_mean", False)
                    if str(pm).lower() in ["false", "0", "no"]:
                        pm = False
                    elif str(pm).lower() in ["true", "1", "yes"]:
                        pm = True
                    preserve_mean.append(pm)
                    red = v.get("reduction", None)
                    if str(red).lower() in ["none", "null"]:
                        red = None
                    reductions.append(red)
                    if verbose:
                        print(
                            f"   Column[{k}]: Weight={weight}, Preserve Mean: {pm}, ",
                            f"Reductions: {red}.",
                        )

            if len(columns) > 0:
                df = dfops.offset_by_other_columns(
                    df=df,
                    target_column=energy_column,
                    offset_columns=columns,
                    weights=weights,
                    preserve_mean=preserve_mean,
                    reductions=reductions,
                )

        # apply constant
        if constant:
            if not isinstance(constant, (int, float, np.integer, np.floating)):
                raise TypeError(f"Invalid type for constant: {type(constant)}")
            df[energy_column] = df.map_partitions(
                lambda x: x[energy_column] + constant,
                meta=(energy_column, np.float64),
            )

        self.offsets = offsets
        metadata["offsets"] = offsets

        return df, metadata


def extract_bias(files: List[str], bias_key: str) -> np.ndarray:
    """Read bias values from hdf5 files

    Args:
        files (List[str]): List of filenames
        bias_key (str): hdf5 path to the bias value

    Returns:
        np.ndarray: Array of bias values.
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
    """Calculate the TOF correction based on the given X/Y coordinates and a model.

    Args:
        x (float): x coordinate
        y (float): y coordinate
        correction_type (str): type of correction. One of
            "spherical", "Lorentzian", "Gaussian", or "Lorentzian_asymmetric"
        center (Tuple[int, int]): center position of the distribution (x,y)
        amplitude (float): Amplitude of the correction
        **kwds: Keyword arguments:

            - **diameter** (float): Field-free drift distance.
            - **gamma** (float): Linewidth value for correction using a 2D
              Lorentz profile.
            - **sigma** (float): Standard deviation for correction using a 2D
              Gaussian profile.
            - **gamma2** (float): Linewidth value for correction using an
              asymmetric 2D Lorentz profile, X-direction.
            - **amplitude2** (float): Amplitude value for correction using an
              asymmetric 2D Lorentz profile, X-direction.

    Returns:
        float: calculated correction value
    """
    if correction_type == "spherical":
        try:
            diameter = kwds.pop("diameter")
        except KeyError as exc:
            raise ValueError(
                f"Parameter 'diameter' required for correction type '{correction_type}' "
                "but not provided!",
            ) from exc
        correction = -(
            (
                1
                - np.sqrt(
                    1 - ((x - center[0]) ** 2 + (y - center[1]) ** 2) / diameter**2,
                )
            )
            * 100
            * amplitude
        )

    elif correction_type == "Lorentzian":
        try:
            gamma = kwds.pop("gamma")
        except KeyError as exc:
            raise ValueError(
                f"Parameter 'gamma' required for correction type '{correction_type}' "
                "but not provided!",
            ) from exc
        correction = (
            100000
            * amplitude
            / (gamma * np.pi)
            * (gamma**2 / ((x - center[0]) ** 2 + (y - center[1]) ** 2 + gamma**2) - 1)
        )

    elif correction_type == "Gaussian":
        try:
            sigma = kwds.pop("sigma")
        except KeyError as exc:
            raise ValueError(
                f"Parameter 'sigma' required for correction type '{correction_type}' "
                "but not provided!",
            ) from exc
        correction = (
            20000
            * amplitude
            / np.sqrt(2 * np.pi * sigma**2)
            * (
                np.exp(
                    -((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma**2),
                )
                - 1
            )
        )

    elif correction_type == "Lorentzian_asymmetric":
        try:
            gamma = kwds.pop("gamma")
        except KeyError as exc:
            raise ValueError(
                f"Parameter 'gamma' required for correction type '{correction_type}' "
                "but not provided!",
            ) from exc
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
    """Normalize a series of 1D signals.

    Args:
        specs (np.ndarray): Collection of 1D signals.
        smooth (bool, optional): Option to smooth the signals before normalization.
            Defaults to False.
        span (int, optional): Smoothing span parameters of the LOESS method
            (see ``scipy.signal.savgol_filter()``). Defaults to 7.
        order (int, optional): Smoothing order parameters of the LOESS method
            (see ``scipy.signal.savgol_filter()``).. Defaults to 1.

    Returns:
        np.ndarray: The matrix assembled from a list of maximum-normalized signals.
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
    """Determine the correspondence between two 1D traces by alignment using a
    time-warp algorithm.

    Args:
        sig_still (np.ndarray): Reference 1D signals.
        sig_mov (np.ndarray): 1D signal to be aligned.
        **kwds: keyword arguments for ``fastdtw.fastdtw()``

    Returns:
        np.ndarray: Pixel-wise path correspondences between two input 1D arrays
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
    from time warping algorithm).

    Args:
        x (np.ndarray): Values of the x axis (e.g. time-of-flight values).
        xrng (Tuple): Boundary value range on the x axis.
        pathcorr (np.ndarray): Path correspondence between two 1D arrays in the
            following form,
            [(id_1_trace_1, id_1_trace_2), (id_2_trace_1, id_2_trace_2), ...]

    Returns:
        Tuple: Transformed range according to the path correspondence.
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
    """Find the value closest to a given one in a 1D array.

    Args:
        val (float): Value of interest.
        narray (np.ndarray):  The array to look for the nearest value.

    Returns:
        int: Array index of the value nearest to the given one.
    """
    return int(np.argmin(np.abs(narray - val)))


def peaksearch(
    traces: np.ndarray,
    tof: np.ndarray,
    ranges: List[Tuple] = None,
    pkwindow: int = 3,
    plot: bool = False,
) -> np.ndarray:
    """Detect a list of peaks in the corresponding regions of multiple spectra.

    Args:
        traces (np.ndarray): Collection of 1D spectra.
        tof (np.ndarray): Time-of-flight values.
        ranges (List[Tuple], optional): List of ranges for peak detection in the format
        [(LowerBound1, UpperBound1), (LowerBound2, UpperBound2), ....].
            Defaults to None.
        pkwindow (int, optional): Window width of a peak (amounts to lookahead in
            ``peakdetect1d``). Defaults to 3.
        plot (bool, optional): Specify whether to display a custom plot of the peak
            search results. Defaults to False.

    Returns:
        np.ndarray: Collection of peak positions.
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
    """Input format checking for 1D peakdtect algorithm

    Args:
        x_axis (np.ndarray): x-axis array
        y_axis (np.ndarray): y-axis array

    Raises:
        ValueError: Raised if x and y values don't have the same length.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of checked (x/y) arrays.
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
    """Function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively

    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    Args:
        y_axis (np.ndarray): A list containing the signal over which to find peaks.
        x_axis (np.ndarray, optional): A x-axis whose values correspond to the y_axis
            list and is used in the return to specify the position of the peaks. If
            omitted an index of the y_axis is used.
        lookahead (int, optional): distance to look ahead from a peak candidate to
            determine if it is the actual peak
            '(samples / period) / f' where '4 >= f >= 1.25' might be a good value.
            Defaults to 200.
        delta (int, optional): this specifies a minimum difference between a peak and
            the following points, before a peak may be considered a peak. Useful
            to hinder the function from picking up false peaks towards to end of
            the signal. To work well delta should be set to delta >= RMSnoise * 5.
            Defaults to 0.

    Raises:
        ValueError: Raised if lookahead and delta are out of range.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of positions of the positive peaks,
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

    if not (np.ndim(delta) == 0 and delta >= 0):
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


def fit_energy_calibration(
    pos: Union[List[float], np.ndarray],
    vals: Union[List[float], np.ndarray],
    binwidth: float,
    binning: int,
    ref_id: int = 0,
    ref_energy: float = None,
    t: Union[List[float], np.ndarray] = None,
    energy_scale: str = "kinetic",
    verbose: bool = True,
    **kwds,
) -> dict:
    """Energy calibration by nonlinear least squares fitting of spectral landmarks on
    a set of (energy dispersion curves (EDCs). This is done here by fitting to the
    function d/(t-t0)**2.

    Args:
        pos (Union[List[float], np.ndarray]): Positions of the spectral landmarks
            (e.g. peaks) in the EDCs.
        vals (Union[List[float], np.ndarray]): Bias voltage value associated with
            each EDC.
        binwidth (float): Time width of each original TOF bin in ns.
        binning (int): Binning factor of the TOF values.
        ref_id (int, optional): Reference dataset index. Defaults to 0.
        ref_energy (float, optional): Energy value of the feature in the refence
            trace (eV). required to output the calibration. Defaults to None.
        t (Union[List[float], np.ndarray], optional): Array of TOF values. Required
            to calculate calibration trace. Defaults to None.
        energy_scale (str, optional): Direction of increasing energy scale.

            - **'kinetic'**: increasing energy with decreasing TOF.
            - **'binding'**: increasing energy with increasing TOF.
        verbose (bool, optional): Option to print out diagnostic information.
            Defaults to True.
        **kwds: keyword arguments:

            - **t0** (float): constrains and initial values for the fit parameter t0,
              corresponding to the time of flight offset. Defaults to 1e-6.
            - **E0** (float): constrains and initial values for the fit parameter E0,
              corresponding to the energy offset. Defaults to min(vals).
            - **d** (float): constrains and initial values for the fit parameter d,
              corresponding to the drift distance. Defaults to 1.

    Returns:
        dict: A dictionary of fitting parameters including the following,

        - "coeffs": Fitted function coefficents.
        - "axis": Fitted energy axis.
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
    d_pars = kwds.pop("d", {})
    pars.add(
        name="d",
        value=d_pars.get("value", 1),
        min=d_pars.get("min", -np.inf),
        max=d_pars.get("max", np.inf),
        vary=d_pars.get("vary", True),
    )
    t0_pars = kwds.pop("t0", {})
    pars.add(
        name="t0",
        value=t0_pars.get("value", 1e-6),
        min=t0_pars.get("min", -np.inf),
        max=t0_pars.get(
            "max",
            (min(pos) - 1) * binwidth * 2**binning,
        ),
        vary=t0_pars.get("vary", True),
    )
    E0_pars = kwds.pop("E0", {})  # pylint: disable=invalid-name
    pars.add(
        name="E0",
        value=E0_pars.get("value", min(vals)),
        min=E0_pars.get("min", -np.inf),
        max=E0_pars.get("max", np.inf),
        vary=E0_pars.get("vary", True),
    )
    fit = Minimizer(
        residual,
        pars,
        fcn_args=(pos, vals, binwidth, binning, energy_scale),
    )
    result = fit.leastsq()
    if verbose:
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
    energy_scale: str = "kinetic",
    **kwds,
) -> dict:
    """Energy calibration by nonlinear least squares fitting of spectral landmarks on
    a set of (energy dispersion curves (EDCs). This amounts to solving for the
    coefficient vector, a, in the system of equations T.a = b. Here T is the
    differential drift time matrix and b the differential bias vector, and
    assuming that the energy-drift-time relationship can be written in the form,
    E = sum_n (a_n * t**n) + E0


    Args:
        pos (Union[List[float], np.ndarray]): Positions of the spectral landmarks
            (e.g. peaks) in the EDCs.
        vals (Union[List[float], np.ndarray]): Bias voltage value associated with
            each EDC.
        order (int, optional): Polynomial order of the fitting function. Defaults to 3.
        ref_id (int, optional): Reference dataset index. Defaults to 0.
        ref_energy (float, optional): Energy value of the feature in the refence
            trace (eV). required to output the calibration. Defaults to None.
        t (Union[List[float], np.ndarray], optional): Array of TOF values. Required
            to calculate calibration trace. Defaults to None.
        aug (int, optional): Fitting dimension augmentation
            (1=no change, 2=double, etc). Defaults to 1.
        method (str, optional): Method for determining the energy calibration.

            - **'lmfit'**: Energy calibration using lmfit and 1/t^2 form.
            - **'lstsq'**, **'lsqr'**: Energy calibration using polynomial form..

            Defaults to "lstsq".
        energy_scale (str, optional): Direction of increasing energy scale.

            - **'kinetic'**: increasing energy with decreasing TOF.
            - **'binding'**: increasing energy with increasing TOF.

    Returns:
        dict: A dictionary of fitting parameters including the following,

        - "coeffs": Fitted polynomial coefficients (the a's).
        - "offset": Minimum time-of-flight corresponding to a peak.
        - "Tmat": the T matrix (differential time-of-flight) in the equation Ta=b.
        - "bvec": the b vector (differential bias) in the fitting Ta=b.
        - "axis": Fitted energy axis.
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
    ecalibdict["energy_scale"] = energy_scale

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
    """(d/(t-t0))**2 expression of the time-of-flight to electron volt
    conversion formula.

    Args:
        tof_distance (float): Drift distance in meter.
        time_offset (float): time offset in ns.
        binwidth (float): Time width of each original TOF bin in ns.
        binning (int): Binning factor of the TOF values.
        energy_scale (str, optional): Direction of increasing energy scale.

            - **'kinetic'**: increasing energy with decreasing TOF.
            - **'binding'**: increasing energy with increasing TOF.

        energy_offset (float): Energy offset in eV.
        t (float): TOF value in bin number.

    Returns:
        float: Converted energy in eV
    """
    sign = 1 if energy_scale == "kinetic" else -1

    #         m_e/2 [eV]                      bin width [s]
    energy = (
        2.84281e-12 * sign * (tof_distance / (t * binwidth * 2**binning - time_offset)) ** 2
        + energy_offset
    )

    return energy


def tof2evpoly(
    poly_a: Union[List[float], np.ndarray],
    energy_offset: float,
    t: float,
) -> float:
    """Polynomial approximation of the time-of-flight to electron volt
    conversion formula.

    Args:
        poly_a (Union[List[float], np.ndarray]): Polynomial coefficients.
        energy_offset (float): Energy offset in eV.
        t (float): TOF value in bin number.

    Returns:
        float: Converted energy.
    """
    odr = len(poly_a)  # Polynomial order
    poly_a = poly_a[::-1]
    energy = 0.0

    for i, order in enumerate(range(1, odr + 1)):
        energy += poly_a[i] * t**order
    energy += energy_offset

    return energy


def tof2ns(
    binwidth: float,
    binning: int,
    t: float,
) -> float:
    """Converts the time-of-flight steps to time-of-flight in nanoseconds.

    designed for use with dask.dataframe.DataFrame.map_partitions.

    Args:
        binwidth (float): Time step size in seconds.
        binning (int): Binning of the time-of-flight steps.
        t (float): TOF value in bin number.
    Returns:
        float: Converted time in nanoseconds.
    """
    val = t * 1e9 * binwidth * 2.0**binning
    return val
