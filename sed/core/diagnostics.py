from typing import Any
from typing import Sequence
from typing import Tuple


def view_event_histogram(  # TODO: check that this works, its just copy-pasted
    workflow,
    dfpid: int,
    ncol: int = 2,
    bins: Sequence[int] = None,
    axes: Sequence[str] = None,
    ranges: Sequence[Tuple[float, float]] = None,
    backend: str = "bokeh",
    legend: bool = True,
    histkwds: dict = None,
    legkwds: dict = None,
    **kwds: Any,
) -> None:
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
        bins = workflow._config["histogram"]["bins"]
    if axes is None:
        axes = workflow._config["histogram"]["axes"]
    if ranges is None:
        ranges = workflow._config["histogram"]["ranges"]

    input_types = map(type, [axes, bins, ranges])
    allowed_types = [list, tuple]

    df = workflow._dataframe

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
    from ..diagnostics import grid_histogram

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
