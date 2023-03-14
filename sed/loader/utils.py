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
    """Collects and sorts files with specified extension from a given folder.

    Args:
        folder (str): The folder to search
        extension (str, optional):  File extension used for glob.glob().
            Defaults to "h5".
        f_start (int, optional): Start file id used to construct a file selector.
            Defaults to None.
        f_end (int, optional): End file id used to construct a file selector.
            Defaults to None.
        f_step (int, optional): Step of file id incrementation, used to construct
            a file selector. Defaults to 1.
        file_sorting (bool, optional): Option to sort the files by their names.
            Defaults to True.

    Returns:
        List[str]: List of collected file names.
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
