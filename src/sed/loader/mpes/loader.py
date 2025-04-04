"""
module sed.loader.mpes, code for loading hdf5 files delayed into a dask dataframe.
Mostly ported from https://github.com/mpes-kit/mpes.
@author: L. Rettig
"""
from __future__ import annotations

import datetime
import glob
import io
import os
from collections.abc import Sequence
from typing import Any

import dask
import dask.array as da
import dask.dataframe as ddf
import h5py
import numpy as np
import scipy.interpolate as sint
from natsort import natsorted

from sed.core.logging import set_verbosity
from sed.core.logging import setup_logging
from sed.loader.base.loader import BaseLoader
from sed.loader.mpes.metadata import MetadataRetriever


# Configure logging
logger = setup_logging("mpes_loader")


def load_h5_in_memory(file_path):
    """
    Load an HDF5 file entirely into memory and open it with h5py.

    Parameters:
        file_path (str): Path to the .h5 file.

    Returns:
        h5py.File: An h5py File object representing the in-memory HDF5 file.
    """
    # Read the entire file into memory
    with open(file_path, "rb") as f:
        file_content = f.read()

    # Load the content into a BytesIO object
    file_buffer = io.BytesIO(file_content)

    # Open the HDF5 file using h5py from the in-memory buffer
    h5_file = h5py.File(file_buffer, "r")

    return h5_file


def hdf5_to_dataframe(
    files: Sequence[str],
    channels: dict[str, Any] = None,
    time_stamps: bool = False,
    time_stamp_alias: str = "timeStamps",
    ms_markers_key: str = "msMarkers",
    first_event_time_stamp_key: str = "FirstEventTimeStamp",
    test_fid: int = 0,
) -> ddf.DataFrame:
    """Function to read a selection of hdf5-files, and generate a delayed dask
    dataframe from provided groups in the files. Optionally, aliases can be defined.

    Args:
        files (List[str]): A list of the file paths to load.
        channels (dict[str, str], optional): hdf5 channels names to load. Each entry in the dict
            should contain the keys "format" and "dataset_key". Defaults to load all groups
            containing "Stream", and to read the attribute "Name" from each group.
        time_stamps (bool, optional): Option to calculate time stamps. Defaults to
            False.
        time_stamp_alias (str): Alias name for the timestamp column.
            Defaults to "timeStamps".
        ms_markers_key (str): hdf5 path containing timestamp information.
            Defaults to "msMarkers".
        first_event_time_stamp_key (str): h5 attribute containing the start
            timestamp of a file. Defaults to "FirstEventTimeStamp".
        test_fid(int, optional): File ID to use for extracting shape information.

    Returns:
        ddf.DataFrame: The delayed Dask DataFrame
    """
    # Read a file to parse the file structure
    test_proc = load_h5_in_memory(files[test_fid])

    if channels is None:
        channels = get_datasets_and_aliases(
            h5file=test_proc,
            search_pattern="Stream",
        )

    electron_channels = []
    column_names = []
    for name, channel in channels.items():
        if channel["format"] == "per_electron":
            if channel["dataset_key"] in test_proc:
                electron_channels.append(channel)
                column_names.append(name)
            else:
                logger.warning(
                    f"Entry \"{channel['dataset_key']}\" for channel \"{name}\" not found. "
                    "Skipping the channel.",
                )
        elif channel["format"] != "per_file":
            error_msg = f"Invalid 'format':{channel['format']} for channel {name}."
            logger.error(error_msg)
            raise ValueError(error_msg)

    if not electron_channels:
        error_msg = "No valid 'per_electron' channels found."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if time_stamps:
        column_names.append(time_stamp_alias)

    test_array = hdf5_to_array(
        h5filename=files[test_fid],
        channels=electron_channels,
        time_stamps=time_stamps,
        ms_markers_key=ms_markers_key,
        first_event_time_stamp_key=first_event_time_stamp_key,
    )

    # Delay-read all files
    arrays = []
    for f in files:
        try:
            arrays.append(
                da.from_delayed(
                    dask.delayed(hdf5_to_array)(
                        h5filename=f,
                        channels=electron_channels,
                        time_stamps=time_stamps,
                        ms_markers_key=ms_markers_key,
                        first_event_time_stamp_key=first_event_time_stamp_key,
                    ),
                    dtype=test_array.dtype,
                    shape=(test_array.shape[0], np.nan),
                ),
            )
        except OSError as exc:
            if "Unable to synchronously open file" in str(exc):
                logger.warning(
                    f"Unable to open file {f}: {str(exc)}. Most likely the file is incomplete.",
                )
                pass

    array_stack = da.concatenate(arrays, axis=1).T

    dataframe = ddf.from_dask_array(array_stack, columns=column_names)

    for name, channel in channels.items():
        if channel["format"] == "per_file":
            if channel["dataset_key"] in test_proc.attrs:
                values = []
                for f in files:
                    try:
                        values.append(float(get_attribute(h5py.File(f), channel["dataset_key"])))
                    except OSError:
                        pass
                delayeds = [
                    add_value(partition, name, value)
                    for partition, value in zip(dataframe.partitions, values)
                ]
                dataframe = ddf.from_delayed(delayeds)

            else:
                logger.warning(
                    f"Entry \"{channel['dataset_key']}\" for channel \"{name}\" not found. "
                    "Skipping the channel.",
                )

    test_proc.close()

    return dataframe


def hdf5_to_timed_dataframe(
    files: Sequence[str],
    channels: dict[str, Any] = None,
    time_stamps: bool = False,
    time_stamp_alias: str = "timeStamps",
    ms_markers_key: str = "msMarkers",
    first_event_time_stamp_key: str = "FirstEventTimeStamp",
    test_fid: int = 0,
) -> ddf.DataFrame:
    """Function to read a selection of hdf5-files, and generate a delayed dask
    dataframe from provided groups in the files. Optionally, aliases can be defined.
    Returns a dataframe for evenly spaced time intervals.

    Args:
        files (List[str]): A list of the file paths to load.
        channels (dict[str, str], optional): hdf5 channels names to load. Each entry in the dict
            should contain the keys "format" and "groupName". Defaults to load all groups
            containing "Stream", and to read the attribute "Name" from each group.
        time_stamps (bool, optional): Option to calculate time stamps. Defaults to
            False.
        time_stamp_alias (str): Alias name for the timestamp column.
            Defaults to "timeStamps".
        ms_markers_key (str): hdf5 dataset containing timestamp information.
            Defaults to "msMarkers".
        first_event_time_stamp_key (str): h5 attribute containing the start
            timestamp of a file. Defaults to "FirstEventTimeStamp".
        test_fid(int, optional): File ID to use for extracting shape information.

    Returns:
        ddf.DataFrame: The delayed Dask DataFrame
    """
    # Read a file to parse the file structure
    test_proc = load_h5_in_memory(files[test_fid])

    if channels is None:
        channels = get_datasets_and_aliases(
            h5file=test_proc,
            search_pattern="Stream",
        )

    electron_channels = []
    column_names = []
    for name, channel in channels.items():
        if channel["format"] == "per_electron":
            if channel["dataset_key"] in test_proc:
                electron_channels.append(channel)
                column_names.append(name)
        elif channel["format"] != "per_file":
            error_msg = f"Invalid 'format':{channel['format']} for channel {name}."
            logger.error(error_msg)
            raise ValueError(error_msg)

    if not electron_channels:
        error_msg = "No valid 'per_electron' channels found."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if time_stamps:
        column_names.append(time_stamp_alias)

    test_array = hdf5_to_timed_array(
        h5filename=files[test_fid],
        channels=electron_channels,
        time_stamps=time_stamps,
        ms_markers_key=ms_markers_key,
        first_event_time_stamp_key=first_event_time_stamp_key,
    )

    # Delay-read all files
    arrays = []
    for f in files:
        try:
            arrays.append(
                da.from_delayed(
                    dask.delayed(hdf5_to_timed_array)(
                        h5filename=f,
                        channels=electron_channels,
                        time_stamps=time_stamps,
                        ms_markers_key=ms_markers_key,
                        first_event_time_stamp_key=first_event_time_stamp_key,
                    ),
                    dtype=test_array.dtype,
                    shape=(test_array.shape[0], np.nan),
                ),
            )
        except OSError as exc:
            if "Unable to synchronously open file" in str(exc):
                pass

    array_stack = da.concatenate(arrays, axis=1).T

    dataframe = ddf.from_dask_array(array_stack, columns=column_names)

    for name, channel in channels.items():
        if channel["format"] == "per_file":
            if channel["dataset_key"] in test_proc.attrs:
                values = []
                for f in files:
                    try:
                        values.append(float(get_attribute(h5py.File(f), channel["dataset_key"])))
                    except OSError:
                        pass
                delayeds = [
                    add_value(partition, name, value)
                    for partition, value in zip(dataframe.partitions, values)
                ]
                dataframe = ddf.from_delayed(delayeds)

    test_proc.close()

    return dataframe


@dask.delayed
def add_value(partition: ddf.DataFrame, name: str, value: float) -> ddf.DataFrame:
    """Dask delayed helper function to add a value to each dataframe partition

    Args:
        partition (ddf.DataFrame): Dask dataframe partition
        name (str): Name of the column to add
        value (float): value to add to this partition

    Returns:
        ddf.DataFrame: Dataframe partition with added column
    """
    partition[name] = value
    return partition


def get_datasets_and_aliases(
    h5file: h5py.File,
    search_pattern: str = None,
    alias_key: str = "Name",
) -> dict[str, Any]:
    """Read datasets and aliases from a provided hdf5 file handle

    Args:
        h5file (h5py.File):
            The hdf5 file handle
        search_pattern (str, optional):
            Search pattern to select groups. Defaults to include all groups.
        alias_key (str, optional):
            Attribute key where aliases are stored. Defaults to "Name".

    Returns:
        dict[str, Any]:
        A dict of aliases and groupnames parsed from the file
    """
    # get group names:
    dataset_names = list(h5file)

    # Filter the group names
    if search_pattern is None:
        filtered_dataset_names = dataset_names
    else:
        filtered_dataset_names = [name for name in dataset_names if search_pattern in name]

    alias_dict = {}
    for name in filtered_dataset_names:
        alias_dict[name] = get_attribute(h5file[name], alias_key)

    return {
        alias_dict[name]: {"format": "per_electron", "dataset_key": name}
        for name in filtered_dataset_names
    }


def hdf5_to_array(
    h5filename: str,
    channels: Sequence[dict[str, Any]],
    time_stamps=False,
    ms_markers_key: str = "msMarkers",
    first_event_time_stamp_key: str = "FirstEventTimeStamp",
) -> np.ndarray:
    """Reads the content of the given groups in an hdf5 file, and returns a
    2-dimensional array with the corresponding values.

    Args:
        h5filename (str): hdf5 file name to read from
        channels (Sequence[dict[str, any]]):
            channel dicts containing group names and types to read.
        time_stamps (bool, optional):
            Option to calculate time stamps. Defaults to False.
        ms_markers_group (str): hdf5 dataset containing timestamp information.
            Defaults to "msMarkers".
        first_event_time_stamp_key (str): h5 attribute containing the start
            timestamp of a file. Defaults to "FirstEventTimeStamp".

    Returns:
        np.ndarray: The 2-dimensional data array containing the values of the groups.
    """

    # Delayed array for loading an HDF5 file of reasonable size (e.g. < 1GB)

    h5file = load_h5_in_memory(h5filename)
    # Read out groups:
    data_list = []
    for channel in channels:
        if channel["format"] == "per_electron":
            g_dataset = np.asarray(h5file[channel["dataset_key"]])
        else:
            raise ValueError(
                f"Invalid 'format':{channel['format']} for channel {channel['dataset_key']}.",
            )
        if "dtype" in channel.keys():
            g_dataset = g_dataset.astype(channel["dtype"])
        else:
            g_dataset = g_dataset.astype("float32")
        data_list.append(g_dataset)

    # calculate time stamps
    if time_stamps:
        # create target array for time stamps
        time_stamp_data = np.zeros(len(data_list[0]))
        # the ms marker contains a list of events that occurred at full ms intervals.
        # It's monotonically increasing, and can contain duplicates
        ms_marker = np.asarray(h5file[ms_markers_key])

        # try to get start timestamp from "FirstEventTimeStamp" attribute
        try:
            start_time_str = get_attribute(h5file, first_event_time_stamp_key)
            start_time = datetime.datetime.strptime(
                start_time_str,
                "%Y-%m-%dT%H:%M:%S.%f%z",
            ).timestamp()
        except KeyError:
            # get the start time of the file from its modification date if the key
            # does not exist (old files)
            start_time = os.path.getmtime(h5filename)  # convert to ms
            # the modification time points to the time when the file was finished, so we
            # need to correct for the time it took to write the file
            start_time -= len(ms_marker) / 1000

        # fill in range before 1st marker
        time_stamp_data[0 : ms_marker[0]] = start_time
        for i in range(len(ms_marker) - 1):
            # linear interpolation between ms: Disabled, because it takes a lot of
            # time, and external signals are anyway not better synchronized than 1 ms
            # time_stamp_data[ms_marker[n] : ms_marker[n + 1]] = np.linspace(
            #     start_time + n,
            #     start_time + n + 1,
            #     ms_marker[n + 1] - ms_marker[n],
            # )
            time_stamp_data[ms_marker[i] : ms_marker[i + 1]] = start_time + (i + 1) / 1000
        # fill any remaining points
        time_stamp_data[ms_marker[len(ms_marker) - 1] : len(time_stamp_data)] = (
            start_time + len(ms_marker) / 1000
        )

        data_list.append(time_stamp_data)

    h5file.close()

    return np.asarray(data_list)


def hdf5_to_timed_array(
    h5filename: str,
    channels: Sequence[dict[str, Any]],
    time_stamps=False,
    ms_markers_key: str = "msMarkers",
    first_event_time_stamp_key: str = "FirstEventTimeStamp",
) -> np.ndarray:
    """Reads the content of the given groups in an hdf5 file, and returns a
    timed version of a 2-dimensional array with the corresponding values.

    Args:
        h5filename (str): hdf5 file name to read from
        channels (Sequence[dict[str, any]]):
            channel dicts containing group names and types to read.
        time_stamps (bool, optional):
            Option to calculate time stamps. Defaults to False.
        ms_markers_group (str): hdf5 dataset containing timestamp information.
            Defaults to "msMarkers".
        first_event_time_stamp_key (str): h5 attribute containing the start
            timestamp of a file. Defaults to "FirstEventTimeStamp".

    Returns:
        np.ndarray: the array of the values at evenly spaced timing obtained from
        the ms_markers.
    """

    # Delayed array for loading an HDF5 file of reasonable size (e.g. < 1GB)

    h5file = load_h5_in_memory(h5filename)
    # Read out groups:
    data_list = []
    ms_marker = np.asarray(h5file[ms_markers_key])
    for channel in channels:
        if channel["format"] == "per_electron":
            g_dataset = np.asarray(h5file[channel["dataset_key"]])
            timed_dataset = g_dataset[np.maximum(ms_marker - 1, 0)]
        else:
            raise ValueError(
                f"Invalid 'format':{channel['format']} for channel {channel['dataset_key']}.",
            )
        if "dtype" in channel.keys():
            timed_dataset = timed_dataset.astype(channel["dtype"])
        else:
            timed_dataset = timed_dataset.astype("float32")

        data_list.append(timed_dataset)

    # calculate time stamps
    if time_stamps:
        # try to get start timestamp from "FirstEventTimeStamp" attribute
        try:
            start_time_str = get_attribute(h5file, first_event_time_stamp_key)
            start_time = datetime.datetime.strptime(
                start_time_str,
                "%Y-%m-%dT%H:%M:%S.%f%z",
            ).timestamp()
        except KeyError:
            # get the start time of the file from its modification date if the key
            # does not exist (old files)
            start_time = os.path.getmtime(h5filename)  # convert to ms
            # the modification time points to the time when the file was finished, so we
            # need to correct for the time it took to write the file
            start_time -= len(ms_marker) / 1000

        time_stamp_data = start_time + np.arange(len(ms_marker)) / 1000

        data_list.append(time_stamp_data)

    h5file.close()

    return np.asarray(data_list)


def get_attribute(h5group: h5py.Group, attribute: str) -> str:
    """Reads, decodes and returns an attribute from an hdf5 group

    Args:
        h5group (h5py.Group):
            The hdf5 group to read from
        attribute (str):
            The name of the attribute

    Returns:
        str: The parsed attribute data
    """
    try:
        content = h5group.attrs[attribute].decode("utf-8")
    except AttributeError:  # No need to decode
        content = h5group.attrs[attribute]
    except KeyError as exc:  # No such attribute
        raise KeyError(f"Attribute '{attribute}' not found!") from exc

    return content


def get_count_rate(
    h5file: h5py.File,
    ms_markers_key: str = "msMarkers",
) -> tuple[np.ndarray, np.ndarray]:
    """Create count rate in the file from the msMarker column.

    Args:
        h5file (h5py.File): The h5file from which to get the count rate.
        ms_markers_key (str, optional): The hdf5 path where the millisecond markers
            are stored. Defaults to "msMarkers".

    Returns:
        tuple[np.ndarray, np.ndarray]: The count rate in Hz and the seconds into the
        scan.
    """
    ms_markers = np.asarray(h5file[ms_markers_key])
    secs = np.arange(0, len(ms_markers)) / 1000
    msmarker_spline = sint.InterpolatedUnivariateSpline(secs, ms_markers, k=1)
    rate_spline = msmarker_spline.derivative()
    count_rate = rate_spline(secs)

    return (count_rate, secs)


def get_elapsed_time(
    h5file: h5py.File,
    ms_markers_key: str = "msMarkers",
) -> float:
    """Return the elapsed time in the file from the msMarkers wave

    Args:
        h5file (h5py.File): The h5file from which to get the count rate.
        ms_markers_key (str, optional): The hdf5 path where the millisecond markers
            are stored. Defaults to "msMarkers".

    Return:
        float: The acquisition time of the file in seconds.
    """
    secs = h5file[ms_markers_key].len() / 1000

    return secs


class MpesLoader(BaseLoader):
    """Mpes implementation of the Loader. Reads from h5 files or folders of the
    SPECS Metis 1000 (FHI Berlin)

    Args:
        config (dict, optional): Config dictionary. Defaults to None.
        verbose (bool, optional): Option to print out diagnostic information.
            Defaults to True.
    """

    __name__ = "mpes"

    supported_file_types = ["h5"]

    def __init__(
        self,
        config: dict = None,
        verbose: bool = True,
    ):
        super().__init__(config=config, verbose=verbose)

        set_verbosity(logger, self._verbose)

        self.read_timestamps = self._config.get("dataframe", {}).get(
            "read_timestamps",
            False,
        )

    @property
    def verbose(self) -> bool:
        """Accessor to the verbosity flag.

        Returns:
            bool: Verbosity flag.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """Setter for the verbosity.

        Args:
            verbose (bool): Option to turn on verbose output. Sets loglevel to INFO.
        """
        self._verbose = verbose
        set_verbosity(logger, self._verbose)

    def read_dataframe(
        self,
        files: str | Sequence[str] = None,
        folders: str | Sequence[str] = None,
        runs: str | Sequence[str] = None,
        ftype: str = "h5",
        metadata: dict = None,
        collect_metadata: bool = False,
        time_stamps: bool = False,
        **kwds,
    ) -> tuple[ddf.DataFrame, ddf.DataFrame, dict]:
        """Read stored hdf5 files from a list or from folder and returns a dask
        dataframe and corresponding metadata.

        Args:
            files (str | Sequence[str], optional): File path(s) to process.
                Defaults to None.
            folders (str | Sequence[str], optional): Path to folder(s) where files
                are stored. Path has priority such that if it's specified, the specified
                files will be ignored. Defaults to None.
            runs (str | Sequence[str], optional): Run identifier(s). Corresponding
                files will be located in the location provided by ``folders``. Takes
                precedence over ``files`` and ``folders``. Defaults to None.
            ftype (str, optional): File extension to use. If a folder path is given,
                all files with the specified extension are read into the dataframe
                in the reading order. Defaults to "h5".
            metadata (dict, optional): Manual meta data dictionary. Auto-generated
                meta data are added to it. Defaults to None.
            collect_metadata (bool): Option to collect metadata from files. Requires
                a valid config dict. Defaults to False.
            time_stamps (bool, optional): Option to create a time_stamp column in
                the dataframe from ms-Markers in the files. Defaults to False.
            **kwds: Keyword parameters.

                - **channels** : Dict of channel informations.
                - **time_stamp_alias**: Alias for the timestamp column
                - **ms_markers_key**: HDF5 path of the millisecond marker column.
                - **first_event_time_stamp_key**: Attribute name containing the start
                  timestamp of the file.

                Additional keywords are passed to ``hdf5_to_dataframe``.

        Raises:
            ValueError: raised if neither files or folder provided.
            FileNotFoundError: Raised if a file or folder is not found.

        Returns:
            tuple[ddf.DataFrame, ddf.DataFrame, dict]: Dask dataframe, timed Dask
            dataframe and metadata read from specified files.
        """
        # if runs is provided, try to locate the respective files relative to the provided folder.
        if runs is not None:
            files = []
            if isinstance(runs, (str, int)):
                runs = [runs]
            for run in runs:
                files.extend(
                    self.get_files_from_run_id(run_id=run, folders=folders, extension=ftype),
                )
            self.runs = list(runs)
            super().read_dataframe(
                files=files,
                ftype=ftype,
                metadata=metadata,
            )
        else:
            super().read_dataframe(
                files=files,
                folders=folders,
                runs=runs,
                ftype=ftype,
                metadata=metadata,
            )

        token = kwds.pop("token", None)
        channels = kwds.pop(
            "channels",
            self._config.get("dataframe", {}).get("channels", None),
        )
        time_stamp_alias = kwds.pop(
            "time_stamp_alias",
            self._config.get("dataframe", {}).get(
                "time_stamp_alias",
                "timeStamps",
            ),
        )
        ms_markers_key = kwds.pop(
            "ms_markers_key",
            self._config.get("dataframe", {}).get(
                "ms_markers_key",
                "msMarkers",
            ),
        )
        first_event_time_stamp_key = kwds.pop(
            "first_event_time_stamp_key",
            self._config.get("dataframe", {}).get(
                "first_event_time_stamp_key",
                "FirstEventTimeStamp",
            ),
        )
        df = hdf5_to_dataframe(
            files=self.files,
            channels=channels,
            time_stamps=time_stamps,
            time_stamp_alias=time_stamp_alias,
            ms_markers_key=ms_markers_key,
            first_event_time_stamp_key=first_event_time_stamp_key,
            **kwds,
        )
        timed_df = hdf5_to_timed_dataframe(
            files=self.files,
            channels=channels,
            time_stamps=time_stamps,
            time_stamp_alias=time_stamp_alias,
            ms_markers_key=ms_markers_key,
            first_event_time_stamp_key=first_event_time_stamp_key,
            **kwds,
        )

        if collect_metadata:
            metadata = self.gather_metadata(
                files=self.files,
                metadata=self.metadata,
                token=token,
            )
        else:
            metadata = self.metadata

        return df, timed_df, metadata

    def get_files_from_run_id(
        self,
        run_id: str,
        folders: str | Sequence[str] = None,
        extension: str = "h5",
        **kwds,
    ) -> list[str]:
        """Locate the files for a given run identifier.

        Args:
            run_id (str): The run identifier to locate.
            folders (str | Sequence[str], optional): The directory(ies) where the raw
                data is located. Defaults to config["core"]["base_folder"]
            extension (str, optional): The file extension. Defaults to "h5".
            kwds: Keyword arguments, not used in this loader.

        Return:
            list[str]: List of file path strings to the location of run data.
        """
        if len(kwds) > 0:
            raise TypeError(
                f"get_files_from_run_id() got unexpected keyword arguments {kwds.keys()}.",
            )

        if folders is None:
            folders = str(self._config["core"]["paths"]["raw"])

        if isinstance(folders, str):
            folders = [folders]

        files: list[str] = []
        for folder in folders:
            run_files = natsorted(
                glob.glob(
                    folder + "/**/Scan" + str(run_id).zfill(4) + "_*." + extension,
                    recursive=True,
                ),
            )
            # Compatibility for old scan format
            if not run_files:
                run_files = natsorted(
                    glob.glob(
                        folder + "/**/Scan" + str(run_id).zfill(3) + "_*." + extension,
                        recursive=True,
                    ),
                )
            files.extend(run_files)

        # Check if any files are found
        if not files:
            raise FileNotFoundError(
                f"No files found for run {run_id} in directory {str(folders)}",
            )

        # Return the list of found files
        return files

    def get_start_and_end_time(self) -> tuple[float, float]:
        """Extract the start and end time stamps from the loaded files

        Returns:
            tuple[float, float]: A tuple containing the start and end time stamps
        """
        h5filename = self.files[0]
        channels = []
        for channel in self._config["dataframe"]["channels"].values():
            if channel["format"] == "per_electron":
                channels = [channel]
                break
        if not channels:
            raise ValueError("No valid 'per_electron' channels found.")
        timestamps = hdf5_to_array(
            h5filename=h5filename,
            channels=channels,
            time_stamps=True,
        )
        ts_from = timestamps[-1][1]
        h5filename = self.files[-1]
        try:
            timestamps = hdf5_to_array(
                h5filename=h5filename,
                channels=channels,
                time_stamps=True,
            )
        except OSError:
            try:
                h5filename = self.files[-2]
                timestamps = hdf5_to_array(
                    h5filename=h5filename,
                    channels=channels,
                    time_stamps=True,
                )
            except OSError:
                ts_to = ts_from
                logger.warning("Could not read end time, using start time as end time!")
        ts_to = timestamps[-1][-1]
        return (ts_from, ts_to)

    def gather_metadata(
        self,
        files: Sequence[str],
        metadata: dict = None,
        token: str = None,
    ) -> dict:
        """Collect meta data from files

        Args:
            files (Sequence[str]): List of files loaded
            metadata (dict, optional): Manual meta data dictionary. Auto-generated
                meta data are added to it. Defaults to None.
            token (str, optional):: The elabFTW api token to use for fetching metadata

        Returns:
            dict: The completed metadata dictionary.
        """

        if metadata is None:
            metadata = {}
        logger.info("Gathering metadata from different locations")
        # Read events in with ms time stamps
        logger.info("Collecting time stamps...")
        (ts_from, ts_to) = self.get_start_and_end_time()

        metadata["timing"] = {
            "acquisition_start": datetime.datetime.utcfromtimestamp(ts_from)
            .replace(tzinfo=datetime.timezone.utc)
            .isoformat(),
            "acquisition_stop": datetime.datetime.utcfromtimestamp(ts_to)
            .replace(tzinfo=datetime.timezone.utc)
            .isoformat(),
            "acquisition_duration": int(ts_to - ts_from),
            "collection_time": float(ts_to - ts_from),
        }

        # import meta data from data file
        if "file" not in metadata:  # If already present, the value is assumed to be a dictionary
            metadata["file"] = {}

        logger.info("Collecting file metadata...")
        with h5py.File(files[0], "r") as h5file:
            for key, value in h5file.attrs.items():
                key = key.replace("VSet", "V")
                metadata["file"][key] = value

        metadata["entry_identifier"] = os.path.dirname(
            os.path.realpath(files[0]),
        )

        metadata_retriever = MetadataRetriever(self._config["metadata"], token)

        metadata = metadata_retriever.fetch_epics_metadata(
            ts_from=ts_from,
            ts_to=ts_to,
            metadata=metadata,
        )

        if self.runs:
            metadata = metadata_retriever.fetch_elab_metadata(
                runs=self.runs,
                metadata=metadata,
            )
        else:
            logger.warning('Fetching elabFTW metadata only supported for loading from "runs"')

        return metadata

    def get_count_rate(
        self,
        fids: Sequence[int] = None,
        **kwds,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create count rate from the msMarker column for the files specified in
        ``fids``.

        Args:
            fids (Sequence[int], optional): fids (Sequence[int]): the file ids to
                include. Defaults to list of all file ids.
            kwds: Keyword arguments:

                - **ms_markers_key**: HDF5 path of the ms-markers

        Returns:
            tuple[np.ndarray, np.ndarray]: Arrays containing countrate and seconds
            into the scan.
        """
        if fids is None:
            fids = range(0, len(self.files))

        ms_markers_key = kwds.pop(
            "ms_markers_key",
            self._config.get("dataframe", {}).get(
                "ms_markers_key",
                "msMarkers",
            ),
        )

        if len(kwds) > 0:
            raise TypeError(f"get_count_rate() got unexpected keyword arguments {kwds.keys()}.")

        secs_list = []
        count_rate_list = []
        accumulated_time = 0
        for fid in fids:
            try:
                count_rate_, secs_ = get_count_rate(
                    h5py.File(self.files[fid]),
                    ms_markers_key=ms_markers_key,
                )
                secs_list.append((accumulated_time + secs_).T)
                count_rate_list.append(count_rate_.T)
                accumulated_time += secs_[-1]
            except OSError as exc:
                if "Unable to synchronously open file" in str(exc):
                    logger.warning(
                        f"Unable to open file {fid}: {str(exc)}. "
                        "Most likely the file is incomplete.",
                    )
                    pass

        count_rate = np.concatenate(count_rate_list)
        secs = np.concatenate(secs_list)

        return count_rate, secs

    def get_elapsed_time(self, fids: Sequence[int] = None, **kwds) -> float:
        """Return the elapsed time in the files specified in ``fids`` from
        the msMarkers column.

        Args:
            fids (Sequence[int], optional): fids (Sequence[int]): the file ids to
                include. Defaults to list of all file ids.
            kwds: Keyword arguments:

                - **ms_markers_key**: HDF5 path of the millisecond marker column.

        Return:
            float: The elapsed time in the files in seconds.
        """
        if fids is None:
            fids = range(0, len(self.files))

        ms_markers_key = kwds.pop(
            "ms_markers_key",
            self._config.get("dataframe", {}).get(
                "ms_markers_key",
                "msMarkers",
            ),
        )

        if len(kwds) > 0:
            raise TypeError(f"get_elapsed_time() got unexpected keyword arguments {kwds.keys()}.")

        secs = 0.0
        for fid in fids:
            try:
                secs += get_elapsed_time(
                    h5py.File(self.files[fid]),
                    ms_markers_key=ms_markers_key,
                )
            except OSError as exc:
                if "Unable to synchronously open file" in str(exc):
                    logger.warning(
                        f"Unable to open file {fid}: {str(exc)}. "
                        "Most likely the file is incomplete.",
                    )
                    pass

        return secs


LOADER = MpesLoader
