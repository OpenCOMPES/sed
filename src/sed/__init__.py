"""sed module easy access APIs."""
import importlib.metadata

import dask

dask.config.set({"dataframe.query-planning": False})

from .core.processor import SedProcessor  # noqa: E402

__version__ = importlib.metadata.version("sed-processor")
__all__ = ["SedProcessor"]
