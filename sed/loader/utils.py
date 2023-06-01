"""Utilities for loaders
"""
from glob import glob
from typing import cast
from typing import List

from h5py import File
from h5py import Group
from natsort import natsorted


def gather_files(
    folder: str,
    extension: str,
    f_start: int = None,
    f_end: int = None,
    f_step: int = 1,
    file_sorting: bool = True,
) -> List[str]:
    """Collects and sorts files with specified extension from a given folder.

    Args:
        folder (str): The folder to search
        extension (str):  File extension used for glob.glob().
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


def parse_h5_keys(h5_file: File, prefix: str = "") -> List[str]:
    """Helper method which parses the channels present in the h5 file
    Args:
        h5_file (h5py.File): The H5 file object.
        prefix (str, optional): The prefix for the channel names.
        Defaults to an empty string.

    Returns:
        List[str]: A list of channel names in the H5 file.

    Raises:
        Exception: If an error occurs while parsing the keys.
    """

    # Initialize an empty list to store the channels
    file_channel_list = []

    # Iterate over the keys in the H5 file
    for key in h5_file.keys():
        try:
            # Check if the object corresponding to the key is a group
            if isinstance(h5_file[key], Group):
                # If it's a group, recursively call the function on the group object
                # and append the returned channels to the file_channel_list
                [
                    file_channel_list.append(s)
                    for s in parse_h5_keys(
                        h5_file[key],
                        prefix=prefix + "/" + key,
                    )
                ]
            else:
                # If it's not a group (i.e., it's a dataset), append the key
                # to the file_channel_list
                file_channel_list.append(prefix + "/" + key)
        except Exception as exception:
            # If an exception occurs, raise a new exception with an error message
            raise Exception(
                f"Error parsing key: {prefix}/{key}",
            ) from exception

    # Return the list of channels
    return file_channel_list
