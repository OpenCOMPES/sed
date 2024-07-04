"""sed module easy access APIs."""
import importlib.metadata

from .core.processor import SedProcessor

__version__ = importlib.metadata.version("sed-processor")
__all__ = ["SedProcessor"]
