"""This module contains NuXus file input/output functions for the sed.io module.
The conversion is based on the nexusutils from the FAIRmat NFDI consortium.
For details, see https://github.com/nomad-coe/nomad-parser-nexus

"""
from typing import Sequence
from typing import Union

import xarray as xr
from pynxtools.dataconverter.convert import convert


def to_nexus(
    data: xr.DataArray,
    faddr: str,
    reader: str,
    definition: str,
    input_files: Union[str, Sequence[str]],
    **kwds,
):
    """Saves the x-array provided to a NeXus file at faddr, using the provided reader,
    NeXus definition and configuration file.

    Args:
        data (xr.DataArray): The data to save, containing metadata definitions in
            data._attrs["metadata"].
        faddr (str): The file path to save to.
        reader (str): The name of the NeXus reader to use.
        definition (str): The NeXus definiton to use.
        config_file (str): The file path to the configuration file to use.
        **kwds: Keyword arguments for ``nexusutils.dataconverter.convert``.
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
