"""Utilities for loaders
"""
from glob import glob
from typing import cast
from typing import List

from natsort import natsorted


def gather_files(
    folder: str,
    extension: str = "h5",
    f_start: int = None,
    f_end: int = None,
    f_step: int = 1,
    file_sorting: bool = True,
) -> List[str]:
    """
    Collects and sorts files with specified extension from a given folder.

    Parameters:
        folder: str
            The folder to search
        extension: str | r'/*.h5'
            File extension used for glob.glob().
        f_start, f_end, f_step: int, int, int | None, None, 1
            Starting, ending file id and the step. Used to construct a file selector.
        file_sorting: bool | True
            Option to sort the files by their names.
    """

    try:
        files = glob(folder + "/*." + extension)

        if file_sorting:
            files = cast(List[str], natsorted(files))

        if f_start is not None and f_end is not None:
            files = files[slice(f_start, f_end, f_step)]

    except FileNotFoundError:
        print("No legitimate folder address is specified for file retrieval!")
        raise

    return files
