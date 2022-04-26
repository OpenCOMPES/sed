import numpy as np
import xarray as xr


def simulate_binned_data(shape: tuple, dims: list):
    assert len(dims) == len(
        shape,
    ), "number of dimesions and data shape must coincide"

    return xr.DataArray(
        data=np.random.rand(*shape),
        coords={d: np.linspace(-1, 1, s) for d, s in zip(dims, shape)},
        attrs={
            "list": [1, 2, 3],
            "string": "asdf",
            "int": 1,
            "float": 1.0,
            "bool": True,
        },
    )
