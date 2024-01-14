""" This module contains fixtures for the FEL module tests.
"""
import os
import shutil
from importlib.util import find_spec

import h5py
import pytest

from sed.core.config import parse_config
from sed.loader.flash.config_model import DataFrameConfig
from sed.loader.flash.config_model import FlashLoaderConfig

package_dir = os.path.dirname(find_spec("sed").origin)
config_path = os.path.join(package_dir, "../tests/data/loader/flash/config.yaml")
H5_PATH = "FLASH1_USER3_stream_2_run43878_file1_20230130T153807.1.h5"
H5_PATHS = [H5_PATH, "FLASH1_USER3_stream_2_run43879_file1_20230130T153807.1.h5"]


@pytest.fixture(name="config_raw")
def fixture_config_raw_file() -> dict:
    """Fixture providing a configuration file for FlashLoader tests.

    Returns:
        dict: The parsed configuration file.
    """
    return parse_config(config_path)


@pytest.fixture(name="config")
def fixture_config_file(config_raw) -> FlashLoaderConfig:
    """Fixture providing a configuration file for FlashLoader tests.

    Returns:
        dict: The parsed configuration file.
    """
    return FlashLoaderConfig(**config_raw)


@pytest.fixture(name="config_dataframe")
def fixture_config_file_dataframe(config) -> DataFrameConfig:
    """Fixture providing a configuration file for FlashLoader tests.

    Returns:
        dict: The parsed configuration file.
    """
    return config.dataframe


@pytest.fixture(name="h5_file")
def fixture_h5_file():
    """Fixture providing an open h5 file.

    Returns:
        h5py.File: The open h5 file.
    """
    return h5py.File(os.path.join(package_dir, f"../tests/data/loader/flash/{H5_PATH}"), "r")


@pytest.fixture(name="h5_file_copy")
def fixture_h5_file_copy(tmp_path):
    """Fixture providing a copy of an open h5 file.

    Returns:
        h5py.File: The open h5 file copy.
    """
    # Create a copy of the h5 file in a temporary directory
    original_file_path = os.path.join(package_dir, f"../tests/data/loader/flash/{H5_PATH}")
    copy_file_path = tmp_path / "copy.h5"
    shutil.copyfile(original_file_path, copy_file_path)

    # Open the copy in 'read-write' mode and return it
    return h5py.File(copy_file_path, "r+")


@pytest.fixture(name="h5_paths")
def fixture_h5_paths():
    """Fixture providing a list of h5 file paths.

    Returns:
        list: A list of h5 file paths.
    """
    return [os.path.join(package_dir, f"../tests/data/loader/flash/{path}") for path in H5_PATHS]


# @pytest.fixture(name="pulserSignAdc_channel_array")
# def get_pulse_channel_from_h5(config_dataframe, h5_file):
#     df = DataFrameCreator(config_dataframe)
#     df.h5_file = h5_file
#     train_id, pulse_id = df.get_dataset_array("pulserSignAdc")
#     return train_id, pulse_id


# @pytest.fixture(name="multiindex_electron")
# def fixture_multi_index_electron(config_dataframe, h5_file):
#     """Fixture providing multi index for electron resolved data"""
#     df = DataFrameCreator(config_dataframe)
#     df.h5_file = h5_file
#     pulse_index, indexer = df.pulse_index(config_dataframe["ubid_offset"])

#     return pulse_index, indexer


# @pytest.fixture(name="fake_data")
# def fake_data_electron():
#     # Creating manageable fake data, but not used currently
#     num_trains = 5
#     max_pulse_id = 100
#     nan_threshold = 50
#     ubid_offset = 5
#     seed = 42
#     np.random.seed(seed)
#     train_ids = np.arange(1600000000, 1600000000 + num_trains)
#     fake_data = []

#     for _ in train_ids:
#         pulse_ids = []
#         while len(pulse_ids) < nan_threshold:
#             random_pulse_ids = np.random.choice(
#                 np.arange(ubid_offset, nan_threshold), size=np.random.randint(0, 10))
#             pulse_ids = np.concatenate([pulse_ids, random_pulse_ids])

#         pulse_ids = np.concatenate([pulse_ids, np.full(max_pulse_id-len(pulse_ids), np.nan)])

#         fake_data.append(np.sort(pulse_ids))
#     return Series(train_ids, name="trainId"), np.array(fake_data), ubid_offset
