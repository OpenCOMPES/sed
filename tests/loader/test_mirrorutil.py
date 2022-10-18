"""Module tests.loader.mirrorutil, tests for the sed.load.mirrorutil file
"""
import glob
import io
import os
import shutil
import tempfile
from contextlib import redirect_stdout

import pytest

import sed
from sed.loader.mirrorutil import CopyTool

# import numpy as np

package_dir = os.path.dirname(sed.__file__)
source_folder = package_dir + "/../"
folder = package_dir + "/../tests/data/loader"
file = package_dir + "/../tests/data/loader/Scan0030_2.h5"


def test_copy_tool_folder():
    """Test the folder copy functionalty of the CopyTool"""
    dest_folder = tempfile.mkdtemp()
    gid = os.getgid()
    ct = CopyTool(
        source_folder,
        dest_folder,
        safetyMargin=0.1 * 2**30,
        gid=gid,
    )
    copied = ct.copy(folder)
    assert os.path.realpath(dest_folder) in copied
    source_content = os.listdir(folder).sort()
    dest_content = os.listdir(copied).sort()
    assert source_content == dest_content
    assert os.stat(copied).st_gid == gid
    assert oct(os.stat(copied).st_mode & 0o777) == "0o775"
    for copied_, source_ in zip(glob.glob(copied), glob.glob(folder)):
        if os.path.isfile(copied_):
            assert os.stat(copied_).st_gid == gid
            assert oct(os.stat(copied_).st_mode & 0o777) == "0o664"
            assert os.path.getsize(copied_) == os.path.getsize(source_)

    shutil.rmtree(dest_folder)


def test_copy_tool_file():
    """Test the file copy functionality of the copy tool"""
    dest_folder = tempfile.mkdtemp()
    gid = os.getgid()
    ct = CopyTool(
        source_folder,
        dest_folder,
        safetyMargin=0.1 * 2**30,
        gid=gid,
    )
    copied = ct.copy(file)
    assert os.path.realpath(dest_folder) in copied
    assert os.stat(copied).st_gid == gid
    assert oct(os.stat(copied).st_mode & 0o777) == "0o664"
    assert os.path.getsize(copied) == os.path.getsize(file)

    shutil.rmtree(dest_folder)


def test_copy_tool_cleanup():
    """Test the file cleanup functionality of the copy tool"""
    dest_folder = tempfile.mkdtemp()
    gid = os.getgid()
    ct = CopyTool(
        source_folder,
        dest_folder,
        safetyMargin=0.1 * 2**30,
        gid=gid,
    )
    copied = ct.copy(folder)
    f = io.StringIO()
    with redirect_stdout(f):
        ct.cleanup_oldest_scan(force=True)
    assert copied in f.getvalue()
    with pytest.raises(FileNotFoundError):
        ct.cleanup_oldest_scan()

    shutil.rmtree(dest_folder)
