"""sed.calibrator.energy module. Code for energy calibration and
correction. Mostly ported from https://github.com/mpes-kit/mpes.
"""
# pylint: disable=too-many-lines
import itertools as it
import os
import pickle
import warnings as wn
from copy import deepcopy
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import bokeh.plotting as pbk
import dask.dataframe
import deepdish.io as dio
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from bokeh.io import output_notebook
from bokeh.palettes import Category10 as ColorCycle
from fastdtw import fastdtw
from lmfit import Minimizer
from lmfit import Parameters
from lmfit.printfuncs import report_fit
from mpes import fprocessing as fp
from numpy.linalg import lstsq
from scipy.signal import savgol_filter
from scipy.sparse.linalg import lsqr
from silx.io import dictdump

from sed.binning import bin_dataframe


class EnergyCalibrator:  # pylint: disable=too-many-instance-attributes
    """
    Electron binding energy calibration workflow.
    """

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        biases: np.ndarray = None,
        traces: np.ndarray = None,
        tof: np.ndarray = None,
        config: dict = {},
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
        self._config = config

        self.featranges: List[Tuple] = []  # Value ranges for feature detection
        self.peaks: np.ndarray = np.asarray([])
        self.calibration: Dict[Any, Any] = {}

        self.tof_column = "t"
        self.energy_column = "E"
        self.x_column = "X"
        self.y_column = "Y"
        self.binwidth: float = 4.125e-12
        self.binning: int = 1

        self.center = (650, 650)
        self.amplitude = -1

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
        ranges: List[Tuple[int, int]] = None,
        biases: np.ndarray = None,
        bias_key: str = None,
        **kwds,
    ):
        """Load and bin data from single-event files

        Parameters:
            data_files: list of file names to bin
            axes: bin axes | _config["energy"]["axes"]
            bins: number of bins | _config["energy"]["bins"]
            ranges: bin ranges | _config["energy"]["ranges"]
            biases: Bias voltages used
            bias_key: hdf5 path where bias values are stored.
                    | _config["energy"]["bias_key"]
            tof: TOF-values for the data traces
        """
        if axes is None:
            axes = self._config["energy"]["axes"]
        if bins is None:
            bins = self._config["energy"]["bins"]
        if ranges is None:
            ranges = self._config["energy"]["ranges"]

        #        hist_mode = kwds.pop("hist_mode", self._config["binning"]["hist_mode"])
        #        mode = kwds.pop("mode", self._config["binning"]["mode"])
        #        pbar = kwds.pop("pbar", self._config["binning"]["pbar"])
        #        num_cores = kwds.pop("num_cores", self._config["binning"]["num_cores"])
        #        threads_per_worker = kwds.pop(
        #            "threads_per_worker",
        #            self._config["binning"]["threads_per_worker"],
        #        )
        #        threadpool_API = kwds.pop(
        #            "threadpool_API",
        #            self._config["binning"]["threadpool_API"],
        #        )

        traces = []
        read_biases = False
        if biases is None:
            biases = []
            read_biases = True

        for file in data_files:
            folder = os.path.dirname(file)
            dfp = fp.dataframeProcessor(datafolder=folder, datafiles=[file])
            dfp.read(source="files", ftype="h5")
            data = bin_dataframe(
                dfp.edf,
                bins=bins,
                axes=axes,
                ranges=ranges,
                # histMode=hist_mode,
                # mode=mode,
                # pbar=pbar,
                # nCores=num_cores,
                # nThreadsPerWorker=threads_per_worker,
                # threadpoolAPI=threadpool_API,
                **kwds,
            )
            traces.append(data.data)
            if read_biases:
                biases.append(extract_bias(file, bias_key))
        tof = data.coords[(axes[0])]
        self.traces = np.asarray(traces)
        self.tof = np.asarray(tof)
        self.biases = np.asarray(biases)

    def normalize(self, **kwds):
        """Normalize the spectra along an axis.

        **Parameters**\n
        **kwds: keyword arguments
            See the keywords for ``mpes.utils.normspec()``.
        """

        self.traces_normed = normspec(self.traces, **kwds)

    def add_features(  # pylint: disable=too-many-arguments
        self,
        ranges: Union[List[Tuple], Tuple],
        refid: int = 0,
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
        refid: int | 0
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
                    traces[refid, :],
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
            Method for determining the energy calibration. "lmfit" or "poly"
        **kwds: keyword arguments
            See available keywords for ``poly_energy_calibration()``.
        """

        landmarks = kwds.pop("landmarks", self.peaks[:, 0])
        biases = kwds.pop("biases", self.biases)
        if method == "lmfit":
            self.calibration = fit_energy_calibation(
                landmarks,
                biases,
                ref_id=ref_id,
                **kwds,
            )
        elif method in ("lstsq", "lsqr"):
            self.calibration = poly_energy_calibration(
                landmarks,
                biases,
                ref_id=ref_id,
                aug=self.dup,
                method=method,
                **kwds,
            )
        else:
            raise NotImplementedError()

        return self.calibration

    def view(  # pylint: disable=W0102, R0912, R0913, R0914
        self,
        traces: np.ndarray,
        segs: List[Tuple[float, float]] = None,
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

        if backend == "matplotlib":

            figsize = kwds.pop("figsize", (12, 4))
            fig, ax = plt.subplots(figsize=figsize)
            for itr, trace in enumerate(traces):
                if align:
                    ax.plot(
                        xaxis
                        - (
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
                        - (
                            self.biases[itr]
                            - self.biases[self.calibration["refid"]]
                        ),
                        trace,
                        color=color,
                        line_dash="solid",
                        line_width=1,
                        line_alpha=1,
                        legend=lbs[itr],
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
                        legend=lbs[itr],
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
            calib_type = "fit"

        elif "a" in kwds and "E0" in kwds:
            poly_a = kwds.pop("a")
            energy_offset = kwds.pop("E0")
            calib_type = "poly"

        elif (
            "t0" in self.calibration
            and "d" in self.calibration
            and "E0" in self.calibration
        ):
            time_offset = self.calibration["t0"]
            drift_distance = self.calibration["d"]
            energy_offset = self.calibration["E0"]
            calib_type = "fit"
        elif "coeffs" in self.calibration and "E0" in self.calibration:
            poly_a = self.calibration["coeffs"]
            energy_offset = self.calibration["E0"]
            calib_type = "poly"

        if calib_type == "fit":
            df[energy_column] = tof2ev(
                drift_distance,
                time_offset,
                energy_offset,
                df[tof_column].astype("float64"),
                binwidth=binwidth,
                binning=binning,
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

    def apply_energy_correction(  # pylint: disable=R0913, R0914
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        tof_column: str = None,
        x_column: str = None,
        y_column: str = None,
        correction_type: str = "Lorentzian",
        center: Tuple[int, int] = None,
        amplitude: float = None,
        **kwds,
    ) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
        """Apply correction to the time-of-flight (TOF) axis of single-event data.

        :Parameters:
            type: str
                Type of correction to apply to the TOF axis.
            **kwds: keyword arguments
                Additional parameters to use for the correction.
                :corraxis: str | 't'
                    String name of the axis to correct.
                :center: list/tuple | (650, 650)
                    Image center pixel positions in (row, column) format.
                :amplitude: numeric | -1
                    Amplitude of the time-of-flight correction term
                    (negative sign meaning subtracting the curved wavefront).
                :d: numeric | 0.9
                    Field-free drift distance.
                :t0: numeric | 0.06
                    Time zero position corresponding to the tip of the valence band.
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

        :Return:

        """

        if tof_column is None:
            tof_column = self.tof_column
        if x_column is None:
            x_column = self.x_column
        if y_column is None:
            y_column = self.y_column
        if center is None:
            center = self.center

        if amplitude is None:
            amplitude = self.amplitude

        if correction_type == "spherical":
            diameter = kwds.pop("d", 0.9)
            time_offset = kwds.pop("t0", 0.06)
            df[tof_column] += (
                (
                    np.sqrt(
                        1
                        + (
                            (df[x_column] - center[0]) ** 2
                            + (df[y_column] - center[1]) ** 2
                        )
                        / diameter**2,
                    )
                    - 1
                )
                * time_offset
                * amplitude
            )

        elif correction_type == "Lorentzian":
            gam = kwds.pop("gamma", 300)
            df[tof_column] += (
                amplitude
                / (gam * np.pi)
                * (
                    gam**2
                    / (
                        (df[x_column] - center[0]) ** 2
                        + (df[y_column] - center[1]) ** 2
                        + gam**2
                    )
                )
            ) - amplitude / (gam * np.pi)

        elif correction_type == "Gaussian":
            sigma = kwds.pop("sigma", 300)
            df[tof_column] += (
                amplitude
                / np.sqrt(2 * np.pi * sigma**2)
                * np.exp(
                    -(
                        (df[x_column] - center[0]) ** 2
                        + (df[y_column] - center[1]) ** 2
                    )
                    / (2 * sigma**2),
                )
            )

        elif correction_type == "Lorentzian_asymmetric":
            gamma = kwds.pop("gamma", 300)
            gamma2 = kwds.pop("gamma2", 300)
            amplitude2 = kwds.pop("amplitude2", -1)
            df[tof_column] += (
                amplitude
                / (gamma * np.pi)
                * (gamma**2 / ((df[y_column] - center[1]) ** 2 + gamma**2))
            )
            df[tof_column] += (
                amplitude2
                / (gamma2 * np.pi)
                * (
                    gamma2**2
                    / ((df[x_column] - center[0]) ** 2 + gamma2**2)
                )
            )

        else:
            raise NotImplementedError

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
        fh = open(save_addr, "wb")
        pickle.dump(clss, fh)
        fh.close()
    elif form in ("h5", "hdf5"):
        dictcopy = deepcopy(clss.__dict__)
        dictcopy["featranges"] = np.asarray(dictcopy["featranges"])
        try:
            dictdump.dicttoh5(dictcopy, save_addr)
        except KeyError:
            dio.save(save_addr, dictcopy, compression=None)

    else:
        raise NotImplementedError


def extract_bias(file: str, bias_key: str) -> float:
    """
    Read bias value from hdf5 file

    Parameters:
        file: filename
        bias_key: hdf5 path to the bias value

    Returns:
        bias value
    """
    with h5py.File(file, "r") as f:
        if bias_key[0] == "@":
            bias = f.attrs[bias_key[1:]]
        else:
            bias = f[bias_key]

    return -round(bias, 2)


def normspec(
    specs: np.ndarray,
    smooth: bool = False,
    span: int = 7,
    order: int = 1,
) -> np.ndarray:
    """
    Normalize a series of 1D signals.

    **Parameters**\n
    *specs: list/2D array
        Collection of 1D signals.
    smooth: bool | False
        Option to smooth the signals before normalization.
    span, order: int, int | 13, 1
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


def peakdetect1d(  # pylint: disable=too-many-branches
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


def fit_energy_calibation(  # pylint: disable=too-many-locals
    pos: Union[List[float], np.ndarray],
    vals: Union[List[float], np.ndarray],
    ref_id: int = 0,
    ref_energy: float = None,
    t: Union[List[float], np.ndarray] = None,
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

    **Returns**\n
    ecalibdict: dict
        A dictionary of fitting parameters including the following,
        :coeffs: Fitted function coefficents.
        :axis: Fitted energy axis.
    """
    binwidth = kwds.pop("binwidth", 4.125e-12)
    binning = kwds.pop("binning", 1)

    vals = np.asarray(vals)
    nvals = vals.size

    if ref_id >= nvals:
        wn.warn(
            "Reference index (refid) cannot be larger than the number of traces!\
                Reset to the largest allowed number.",
        )
        ref_id = nvals - 1

    def residual(pars, time, data, binwidth=binwidth, binning=binning):
        model = tof2ev(
            pars["d"],
            pars["t0"],
            pars["E0"],
            time,
            binwidth=binwidth,
            binning=binning,
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
    fit = Minimizer(residual, pars, fcn_args=(pos, vals))
    result = fit.leastsq()
    report_fit(result)

    # Construct the calibrating function
    pfunc = partial(
        tof2ev,
        result.params["d"].value,
        result.params["t0"].value,
        binwidth=binwidth,
        binning=binning,
    )

    # Return results according to specification
    ecalibdict = {}
    ecalibdict["d"] = result.params["d"].value
    ecalibdict["t0"] = result.params["t0"].value
    ecalibdict["E0"] = result.params["E0"].value

    if (ref_energy is not None) and (t is not None):
        energy_offset = pfunc(-1 * ref_energy, pos[ref_id])
        ecalibdict["axis"] = pfunc(-energy_offset, t)
        ecalibdict["E0"] = -energy_offset
        ecalibdict["refid"] = ref_id

    return ecalibdict


def poly_energy_calibration(  # pylint: disable=R0913, R0914
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


def tof2ev(  # pylint: disable=too-many-arguments
    tof_distance: float,
    time_offset: float,
    energy_offset: float,
    t: float,
    binwidth: float = 4.125e-12,
    binning: int = 1,
) -> float:
    """
    d/(t-t0) expression of the time-of-flight to electron volt
    conversion formula.

    **Parameters**\n
    d: float
        Drift distance
    t0: float
        time offset
    E0: float
        Energy offset.
    t: numeric array
        Drift time of electron.

    **Return**\n
    E: numeric array
        Converted energy
    """
    #         m_e/2 [eV]                      bin width [s]
    energy = (
        2.84281e-12
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
