"""This module contains diagnostic output functions for the sed module

"""
from typing import Sequence
from typing import Tuple

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
        **kwds: Keyword arguments for ``bokeh.plotting.figure().quad()``.

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
    rvranges: Sequence[Tuple[float, float]],
    backend: str = "bokeh",
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
        rvranges (Sequence[Tuple[float, float]]): Value ranges of all random variables.
        backend (str, optional): Backend for making the plot ('matplotlib' or 'bokeh').
            Defaults to "bokeh".
        legend (bool, optional): Option to include a legend in each histogram plot.
            Defaults to True.
        histkwds (dict, optional): Keyword arguments for histogram plots.
            Defaults to None.
        legkwds (dict, optional): Keyword arguments for legends. Defaults to None.
        **kwds: Additional keyword arguments.
    """
    if histkwds is None:
        histkwds = {}
    if legkwds is None:
        legkwds = {}

    figsz = kwds.pop("figsize", (14, 8))

    if backend == "matplotlib":
        nrv = len(rvs)
        nrow = int(np.ceil(nrv / ncol))
        histtype = kwds.pop("histtype", "step")

        fig, ax = plt.subplots(nrow, ncol, figsize=figsz)
        otherax = ax.copy()
        for i, zipped in enumerate(zip(rvs, rvbins, rvranges)):
            # Make each histogram plot
            rvname, rvbin, rvrg = zipped
            try:
                axind = np.unravel_index(i, (nrow, ncol))
                ax[axind].hist(
                    dct[rvname],
                    bins=rvbin,
                    range=rvrg,
                    label=rvname,
                    histtype=histtype,
                    **histkwds,
                )
                if legend:
                    ax[axind].legend(fontsize=15, **legkwds)

                otherax[axind] = None

            except IndexError:
                ax[i].hist(
                    dct[rvname],
                    bins=rvbin,
                    range=rvrg,
                    label=rvname,
                    histtype=histtype,
                    **histkwds,
                )
                if legend:
                    ax[i].legend(fontsize=15, **legkwds)

                otherax[i] = None

        for oax in otherax.flatten():
            if oax is not None:
                fig.delaxes(oax)

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
                width=figsz[0] * 30,
                height=figsz[1] * 28,
            ),
        )
