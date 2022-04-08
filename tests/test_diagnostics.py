import random

import numpy as np
import pytest
import xarray as xr

from sed.diagnostics import simulate_binned_data

shapes = []
for n in range(4):
    shapes.append(tuple(np.random.randint(10) + 1 for i in range(n + 1)))
axes_names = ["x", "y", "t", "e"]
random.shuffle(axes_names)
args = [(s, axes_names[: len(s)]) for s in shapes]


@pytest.mark.parametrize(
    "args",
    args,
    ids=lambda x: f"ndims:{len(x[0])}",
)
def test_simulated_binned_data_is_xarray(args):
    sim = simulate_binned_data(*args)
    assert type(sim) == xr.DataArray
