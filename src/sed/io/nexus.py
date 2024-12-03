"""This module contains NuXus file input/output functions for the sed.io module.
The conversion is based on the pynxtools from the FAIRmat NFDI consortium.
For details, see https://github.com/nomad-coe/nomad-parser-nexus

"""
from __future__ import annotations

from collections.abc import Sequence

import xarray as xr
from pynxtools.dataconverter.convert import convert


def to_nexus(
    data: xr.DataArray,
    faddr: str,
    reader: str,
    definition: str,
    input_files: str | Sequence[str],
    **kwds,
):
    """Saves the x-array provided to a NeXus file at faddr, using the provided reader,
    NeXus definition and configuration file.

    Args:
        data (xr.DataArray): The data to save, containing metadata definitions in
            data._attrs["metadata"].
        faddr (str): The file path to save to.
        reader (str): The name of the NeXus reader to use.
        definition (str): The NeXus definition to use.
        input_files (str | Sequence[str]): The file path or paths to the additional files to use.
        **kwds: Keyword arguments for ``pynxtools.dataconverter.convert.convert()``.
    """

    if isinstance(input_files, str):
        input_files = tuple([input_files])
    else:
        input_files = tuple(input_files)

    convert(
        input_file=input_files,
        objects=(data),
        reader=reader,
        nxdl=definition,
        output=faddr,
        **kwds,
    )
