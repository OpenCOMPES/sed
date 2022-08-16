from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple

import bokeh.plotting as pbk
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import output_notebook
from bokeh.layouts import gridplot


def plot_single_hist(
    histvals: List[int],
    edges: List[float],
    legend: str = None,
    **kwds: Any,
) -> pbk.Figure:
    """Bokeh-based plotting of a single histogram with legend and tooltips.

    Args:
        histvals: Histogram counts (e.g. vertical axis).
        edges: Histogram edge values (e.g. horizontal axis).
        legend: Text for the plot legend.
        **kwds: Keyword arguments for 'bokeh.plotting.figure().quad()'.

    Returns:
        An instance of 'bokeh.plotting.Figure' as a plot handle.
    """

    ttp = kwds.pop("tooltip", [("(x, y)", "($x, $y)")])

    p = pbk.figure(background_fill_color="white", tooltips=ttp)
    p.quad(
        top=histvals,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        line_color="white",
        alpha=0.8,
        legend=legend,
        **kwds,
    )

    p.y_range.start = 0
    p.legend.location = "top_right"
    p.grid.grid_line_color = "lightgrey"

    return p


def grid_histogram(
    dct: dict,
    ncol: int,
    rvs: Sequence,
    rvbins: Sequence,
    rvranges: Sequence[Tuple[float, float]],
    backend: str = "bokeh",
    legend: bool = True,
    histkwds: dict = {},
    legkwds: dict = {},
    **kwds: Any,
):
    """
    Grid plot of multiple 1D histograms.

    Args:
        dct: Dictionary containing the name and values of the random variables.
        ncol: Number of columns in the plot grid.
        rvs: List of names for the random variables (rvs).
        rvbins: Bin values for all random variables.
        rvranges: Value ranges of all random variables.
        backend: Backend for making the plot ('matplotlib' or 'bokeh').
        legend: Option to include a legend in each histogram plot.
        histkwds: Keyword arguments for histogram plots.
        legkwds: Keyword arguments for legends.
        **kwds: Additional keyword arguments.
    """

    figsz = kwds.pop("figsize", (14, 8))

    if backend == "matplotlib":

        nrv = len(rvs)
        nrow = int(np.ceil(nrv / ncol))
        histtype = kwds.pop("histtype", "step")

        f, ax = plt.subplots(nrow, ncol, figsize=figsz)
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
                f.delaxes(oax)

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
                plots,
                ncols=ncol,
                plot_width=figsz[0] * 30,
                plot_height=figsz[1] * 28,
            ),
        )
