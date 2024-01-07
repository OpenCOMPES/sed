"""This file contains code that performs several tests for the sed package
"""
from sed import __version__


def test_version() -> None:
    """This function tests for the version of the package"""
    assert __version__ == "0.1.0"
