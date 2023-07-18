"""Interface to select a specified loader
"""
import glob
import importlib.util
import os
from typing import List

from sed.loader.base.loader import BaseLoader


def get_loader(
    loader_name: str,
    config: dict = None,
) -> BaseLoader:
    """Helper function to get the loader object from it's given name.

    Args:
        loader_name (str): Name of the loader
        config (dict, optional): Configuration dictionary. Defaults to None.

    Raises:
        ValueError: Raised if the loader cannot be found.

    Returns:
        BaseLoader: The loader object.
    """

    if config is None:
        config = {}

    path_prefix = f"{os.path.dirname(__file__)}{os.sep}" if os.path.dirname(__file__) else ""
    path = os.path.join(path_prefix, loader_name, "loader.py")
    if not os.path.exists(path):
        error_str = f"Invalid loader {loader_name}. Available loaders are: ["
        for loader in get_names_of_all_loaders():
            error_str += f"{loader}, "
        error_str += "]."
        raise ValueError(error_str)

    spec = importlib.util.spec_from_file_location("loader.py", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LOADER(config=config)


def get_names_of_all_loaders() -> List[str]:
    """Helper function to populate a list of all available loaders.

    Returns:
        List[str]: List of all detected loader names.
    """
    path_prefix = f"{os.path.dirname(__file__)}{os.sep}" if os.path.dirname(__file__) else ""
    files = glob.glob(os.path.join(path_prefix, "*", "loader.py"))
    all_loaders = []
    for file in files:
        if f"{os.sep}base{os.sep}" not in file:
            index_of_loaders_folder_name = file.rindex(
                f"loader{os.sep}",
            ) + len(f"loader{os.sep}")
            index_of_last_path_sep = file.rindex(os.sep)
            all_loaders.append(
                file[index_of_loaders_folder_name:index_of_last_path_sep],
            )
    return all_loaders
