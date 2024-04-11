"""sed module easy access APIs

"""
import toml

from .core.processor import SedProcessor

config = toml.load("pyproject.toml")
__version__ = config["tool"]["poetry"]["version"]
__all__ = ["SedProcessor"]
