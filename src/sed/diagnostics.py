"""This module contains diagnostic output functions for the sed module

"""
from __future__ import annotations

from collections.abc import Sequence

import bokeh.plotting as pbk
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import output_notebook
from bokeh.layouts import gridplot


def plot_single_hist(
    histvals: np.ndarray,
    edges: np.ndarray,
    legend: str = None,
    **kwds,
) -> pbk.figure:
    """Bokeh-based plotting of a single histogram with legend and tooltips.

    Args:
        histvals (np.ndarray): Histogram counts (e.g. vertical axis).
        edges (np.ndarray): Histogram edge values (e.g. horizontal axis).
        legend (str, optional): Text for the plot legend. Defaults to None.
        **kwds:
            - *tooltip*: Tooltip formatting tuple. Defaults to [("(x, y)", "($x, $y)")]

            Additional keyword arguments are passed to ``bokeh.plotting.figure().quad()``.

    Returns:
        pbk.figure: An instance of 'bokeh.plotting.figure' as a plot handle.
    """
    ttp = kwds.pop("tooltip", [("(x, y)", "($x, $y)")])

    fig = pbk.figure(background_fill_color="white", tooltips=ttp)
    fig.quad(
        top=histvals,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        line_color="white",
        alpha=0.8,
        legend_label=legend,
        **kwds,
    )

    fig.y_range.start = 0  # type: ignore
    fig.legend.location = "top_right"
    fig.grid.grid_line_color = "lightgrey"

    return fig


def grid_histogram(
    dct: dict,
    ncol: int,
    rvs: Sequence,
    rvbins: Sequence,
    rvranges: Sequence[tuple[float, float]],
    backend: str = "matplotlib",
    legend: bool = True,
    histkwds: dict = None,
    legkwds: dict = None,
    **kwds,
):
    """Grid plot of multiple 1D histograms.

    Args:
        dct (dict): Dictionary containing the name and values of the random variables.
        ncol (int): Number of columns in the plot grid.
        rvs (Sequence): List of names for the random variables (rvs).
        rvbins (Sequence): Bin values for all random variables.
        rvranges (Sequence[tuple[float, float]]): Value ranges of all random variables.
        backend (str, optional): Backend for making the plot ("matplotlib" or "bokeh").
            Defaults to "matplotlib".
        legend (bool, optional): Option to include a legend in each histogram plot.
            Defaults to True.
        histkwds (dict, optional): Keyword arguments for histogram plots.
            Defaults to None.
        legkwds (dict, optional): Keyword arguments for legends. Defaults to None.
        **kwds:
            - *figsize*: Figure size. Defaults to (6, 4)
    """
    if histkwds is None:
        histkwds = {}
    if legkwds is None:
        legkwds = {}

    figsz = kwds.pop("figsize", (3, 2))  # figure size of each panel

    if len(kwds) > 0:
        raise TypeError(f"grid_histogram() got unexpected keyword arguments {kwds.keys()}.")

    if backend == "matplotlib":
        nrv = len(rvs)
        nrow = int(np.ceil(nrv / ncol))
        histtype = kwds.pop("histtype", "bar")

        figsize = [figsz[0] * ncol, figsz[1] * nrow]
        fig, ax = plt.subplots(nrow, ncol, figsize=figsize)
        otherax = ax.copy()
        for i, zipped in enumerate(zip(rvs, rvbins, rvranges)):
            # Make each histogram plot
            rvname, rvbin, rvrg = zipped
            try:
                axind = np.unravel_index(i, (nrow, ncol))
                plt.setp(ax[axind].get_xticklabels(), fontsize=8)
                plt.setp(ax[axind].get_yticklabels(), fontsize=8)
                ax[axind].hist(
                    dct[rvname],
                    bins=rvbin,
                    range=rvrg,
                    label=rvname,
                    histtype=histtype,
                    **histkwds,
                )
                if legend:
                    ax[axind].legend(fontsize=10, **legkwds)

                otherax[axind] = None

            except IndexError:
                plt.setp(ax[i].get_xticklabels(), fontsize=8)
                plt.setp(ax[i].get_yticklabels(), fontsize=8)
                ax[i].hist(
                    dct[rvname],
                    bins=rvbin,
                    range=rvrg,
                    label=rvname,
                    histtype=histtype,
                    **histkwds,
                )
                if legend:
                    ax[i].legend(fontsize=10, **legkwds)

                otherax[i] = None

        for oax in otherax.flatten():
            if oax is not None:
                fig.delaxes(oax)

        plt.tight_layout()

    elif backend == "bokeh":
        output_notebook(hide_banner=True)

        plots = []
        for i, zipped in enumerate(zip(rvs, rvbins, rvranges)):
            rvname, rvbin, rvrg = zipped
            histvals, edges = np.histogram(dct[rvname], bins=rvbin, range=rvrg)

            if legend:
                plots.append(
                    plot_single_hist(
                        histvals,
                        edges,
                        legend=rvname,
                        **histkwds,
                    ),
                )
            else:
                plots.append(
                    plot_single_hist(histvals, edges, legend=None, **histkwds),
                )

        # Make grid plot
        pbk.show(
            gridplot(
                plots,  # type: ignore
                ncols=ncol,
                width=figsz[0] * 100,
                height=figsz[1] * 100,
            ),
        )
