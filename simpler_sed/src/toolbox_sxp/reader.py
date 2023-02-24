"""
Read functions for the toolbox_sxp package.

Inspired by the SED library:
<https://github.com/OpenCOMPES/sed>

Author: David Doblas Jiménez <david.doblas-jimenez@xfel.eu>
Copyright (c) 2023, European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pasha as psh
import polars as pl

from .utils import concat_dfs
from .utils import jitter

__all__ = ["get_df"]


# Groups in the h5 files
raw_hdf = {
    "X": "Stream_0",
    "Y": "Stream_1",
    "t": "Stream_2",
    "ADC": "Stream_4",
}


def _get_chunks(files):
    """
    Get chunks of rows from a bunch of files
    """
    offset = {}
    for idx, f in enumerate(files):
        offset[idx] = h5py.File(f)["Stream_0"].shape[0]
    return offset


def _sort_dir(files_dir):
    """Sort files from a directory

    Files are sorted by ascending order from a directory with the following
    naming convention:
        File_1, File_10, File_100, File_2, File_3, ...

    Args:
        files_dir: str
            Path of the directory where the files are located. Accepts also a
            Path object

    Returns:
        list
            List of files sorted in ascending natural order
    """
    return [
        f
        for f in sorted(
            files_dir.iterdir(),
            key=lambda path: int(path.stem.rsplit("_", maxsplit=1)[1]),
        )
    ]


def get_df(files_dir, apply_jitter=True, concat_raw=True):
    """Read raw data and return a dataframe

    Read raw data from a directory using multiple process and shared memory via
    `pasha`. The output is a `polars` dataframe.

    Args:
        files_dir: str
            Path of the directory where the files are located. Accepts also a
            Path object
        apply_jitter: bool
            True (default) will add a randomly distributed jitter to raw data
        concat_raw: bool
            True (default) will return jittered columns next to raw data

    Returns:
        polars.DataFrame
            Dataframe with events (electrons) in rows and columns defined in
            the `raw_hdf` dictionary
    """
    sorted_files = _sort_dir(files_dir)
    chunks = np.cumsum(list(_get_chunks(sorted_files).values()))

    # work is distributed across processes between 60 workers
    psh.set_default_context("processes", num_workers=60)
    length = sum(h5py.File(f)["Stream_0"].shape[0] for f in sorted_files)
    outp = psh.alloc(shape=(len(raw_hdf), length), dtype=np.uint32)

    # pasha kernel functions
    def _read_h5f(f, s, e):
        with h5py.File(f, "r") as hf:
            for idx, val in enumerate(raw_hdf.values()):
                hf[val].read_direct(outp[idx, s:e])

    def _fill_pasha_array(wid, index, value):
        start = chunks[index - 1] if index != 0 else 0
        end = chunks[index]
        _read_h5f(value, start, end)

    psh.map(_fill_pasha_array, sorted_files)

    # pandas creates zero-copy of the allocated numpy array
    _pandas = pd.DataFrame(outp.T, columns=list(raw_hdf))
    # but a polars df is returned
    df = pl.DataFrame(_pandas)

    if apply_jitter:
        _jit = jitter(size=length, distribution="random")
        df_jit = df.select(
            [(pl.all() + _jit).cast(pl.Float32).suffix("_jitter")],
        )
        if not concat_raw:
            return df_jit

    if concat_raw and apply_jitter:
        df = concat_dfs([df, df_jit])

    return df
