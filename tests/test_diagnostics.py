import random

import numpy as np
import pytest
import xarray as xr

from sed.diagnostics import simulate_binned_data

shapes = []
for n in range(4):
    shapes.append([np.random.randint[10] + 1 for i in range(n)])
axes_names = random.shuffle(["x", "y", "t", "e"])


@pytest.mark.parametrize(
    "_shape",
    shapes,
    ids=lambda x: f"shapes:{x.shape}",
)
def test_simulated_binned_data_is_xarray(_shape):
    sim = simulate_binned_data(_shape, axes_names)
    assert isinstance(sim, xr.DataArray)
