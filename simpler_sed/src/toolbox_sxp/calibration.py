"""
Calibration functions for the toolbox_sxp package.

Inspired by the SED library:
<https://github.com/OpenCOMPES/sed>

Author: David Doblas Jiménez <david.doblas-jimenez@xfel.eu>
Copyright (c) 2023, European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""
import math
from collections import namedtuple
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dtaidistance import dtw
from numpy.linalg import norm
from scipy.linalg import lstsq
from scipy.signal import find_peaks

from .utils import _range_convert


def _generate_momentum_config(arr, point, k_distance):
    """Get config from detector for momentum calibration

    Args:
        arr: numpy_ndarray
            2D array
        point: numpy.ndarray
            Pixel (y, x) coordinates of a high symmetry point
        k_distance: float
            Momentum distance between point a and b

    Returns:
        namedtuple
            Tuple with detector fields accessible by attribute lookup
    """
    k_dist = 4 / 3 * np.pi / k_distance

    point_b = np.array([256, 256])

    pixel_dist = norm(point - point_b)
    # Use the same conversion factor along both x and y directions
    x_ratio = y_ratio = k_dist / pixel_dist

    k_coord_b = [0.0, 0.0]
    r_center = point_b[0] - k_coord_b[0] / x_ratio
    c_center = point_b[1] - k_coord_b[1] / y_ratio

    r_start = -256.0
    c_start = -256.0

    rows, cols = arr.shape
    r_step = (1792.0 - (-256.0)) / rows
    c_step = (1792.0 - (-256.0)) / cols

    det_config = namedtuple(
        "Det_config",
        [
            "row_start",
            "col_start",
            "row_step",
            "col_step",
            "row_centre",
            "col_centre",
            "row_conversion",
            "col_conversion",
        ],
    )

    return det_config(
        r_start,
        c_start,
        r_step,
        c_step,
        r_center,
        c_center,
        x_ratio,
        y_ratio,
    )


def momentum_cal(
    arr,
    point,
    k_distance,
    df,
):
    """Calibrate momenum axes from detector to momentum coordinates

    Args:
        arr: numpy_ndarray
            2D array
        point: numpy.ndarray
            Pixel (y, x) coordinates of a high symmetry point
        k_distance: float
            Momentum distance between point a and b
        df: polars.DataFrame

    Returns:
        polars.DataFrame
            Dataframe with `kx` and `ky` columns resulting from the conversion
            of cartesian coordinates to "momentum" coordinates
    """
    det_config = _generate_momentum_config(arr, point, k_distance)
    r_det, c_det = df[["Xm", "Ym"]]

    r_det0 = det_config.row_start + det_config.row_step * det_config.row_centre
    c_det0 = det_config.col_start + det_config.col_step * det_config.col_centre
    k_r = det_config.row_conversion * ((r_det - r_det0) / det_config.row_step)
    k_c = det_config.col_conversion * ((c_det - c_det0) / det_config.col_step)

    df = pl.DataFrame(
        {
            "kx": k_r,
            "ky": k_c,
        },
    )

    return df


def peaks_search(
    x_arr,
    y_arr,
    initial_guess_range,
    plot=True,
):
    """Find peaks in a series of curves

    Args:
        x_arr: numpy.array
            1D array
        y_arr: numpy.ndarray
            2D array where to find peaks. Dimensions must macth with `x_arr`
        initial_guess_range: (int, int)
            Range containing the peak
        plot: bool
            True (default) will return a figure with `x_arr` vs. `y_arr`s where
            the first peak has been identified
    Returns:
        numpy.ndarray
            Array with pairs of (peak position, intensity)

    """
    if plot:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        cmap = plt.get_cmap("jet_r")

    # https://stackoverflow.com/questions/44994866/\
    # efficient-pairwise-dtw-calculation-using-numpy-or-cython
    dist_and_paths = [
        dtw.warping_paths_fast(s1, s2)
        for s1, s2 in zip(repeat(y_arr[0], len(y_arr)), y_arr)
    ]

    paths = np.array([np.array(dtw.best_path(p)) for _, p in dist_and_paths])
    peak_ranges = [
        _range_convert(x_arr, initial_guess_range, p) for p in paths
    ]

    peaks_found = []
    for idx, (rg, trace) in enumerate(zip(peak_ranges, y_arr)):
        cond = (x_arr > rg[0]) & (x_arr <= rg[1])
        x, y = x_arr[cond], trace[cond]

        _peaks, _ = find_peaks(y, height=0.9)

        # use maximum as the only peak
        peaks = max(_peaks)
        peaks_found.append((x[peaks], y[peaks]))

        if plot:
            color = cmap(float(idx) / len(y_arr))
            ax.plot(x, y, c=color, alpha=0.7)
            ax.plot(x[peaks], y[peaks], "x", c=color)
            ax.plot(x_arr, trace, "--k", linewidth=0.7)
            ax.set_xlabel("Time of flight", fontsize=15)
            ax.set_ylabel("Normalized intensity", fontsize=15)
            ax.grid()

    return np.array(peaks_found)


def tof_to_ev_poly(a, E0, t):
    """Convert from tof to ev

    Polynomial approximation of the time-of-flight to electron volt conversion
    formula.

    Args:
        a: numpy.ndarray
            Polynomial coefficients
        E0: float
            Energy offset
        t: float or numpy.ndarray
            Drift time of electron

    Returns:
        E: float or numpy.ndarray
            Converted energy
    """
    energy = a[0] * t
    for coeff in a[1:]:
        energy = (energy + coeff) * t
    energy += E0

    return energy


def energy_cal(x_arr, y_arr, my_pks, voltages, plot):
    """

    Args:


    Returns:
        numpy.ndarray

    """
    Eref = -0.5
    refid = 4
    order = 3
    _poly_order = list(range(order, 0, -1))
    _term_order = [n for n in range(voltages.size) if n != refid]

    _Tsec = np.array(
        [
            math.pow(x[0], p)
            for p in _poly_order
            for i, x in enumerate(my_pks)
            if i != refid
        ],
    ).reshape((len(_term_order), len(_poly_order)), order="F")

    _Tmain = np.array([math.pow(my_pks[refid][0], p) for p in _poly_order])

    _Tmat = _Tmain - _Tsec

    _bvec = np.delete(voltages[refid] - voltages, refid)

    _p, _res, _rnk, _s = lstsq(_Tmat, _bvec)

    _E0 = -tof_to_ev_poly(_p, -Eref, my_pks[refid][0])

    _energy_calibration = tof_to_ev_poly(_p, _E0, x_arr)

    if plot:
        fig1, ax1 = plt.subplots(figsize=(9, 4), constrained_layout=True)
        fig1.suptitle("Quality of Calibration")
        for idx, voltage in enumerate(voltages):
            ax1.plot(
                _energy_calibration - (voltage - voltages[refid]),
                y_arr[idx],
            )
        ax1.set_xlim([-23, 15])

        ax1.set_xlabel("Energy (eV)", fontsize=15)
        ax1.set_ylabel("Normalized intensity", fontsize=15)
        ax1.grid()

        sign = 1
        fig2, ax2 = plt.subplots(figsize=(9, 4), constrained_layout=True)
        fig2.suptitle("E/TOF relationship")
        ax2.scatter(
            [x[0] for x in my_pks],
            sign * (voltages - voltages[refid]) + Eref,
            s=30,
            c="k",
            zorder=2.5,
        )  # move dots on top of line
        ax2.plot(x_arr, _energy_calibration, "--")
        ax2.set_xlabel("Time-of-flight", fontsize=15)
        ax2.set_ylabel("Energy (eV)", fontsize=15)
        ax2.set_ylim([-18, 9])
        ax2.set_xlim([63500, 75500])

    return _energy_calibration
