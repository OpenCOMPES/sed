"""
module sed.loader.mpes, code for loading hdf5 files delayed into a dask dataframe.
Mostly ported from https://github.com/mpes-kit/mpes.
@author: L. Rettig
"""
import datetime
import os
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import dask
import dask.array as da
import dask.dataframe as ddf
import h5py
import numpy as np
import scipy.interpolate as sint

from sed.core.metadata import MetaHandler
from sed.loader.base.loader import BaseLoader
from sed.loader.utils import gather_files


def hdf5_to_dataframe(
    files: Sequence[str],
    group_names: Sequence[str] = None,
    alias_dict: Dict[str, str] = None,
    time_stamps: bool = False,
    time_stamp_alias: str = "timeStamps",
    ms_markers_group: str = "msMarkers",
    first_event_time_stamp_key: str = "FirstEventTimeStamp",
    **kwds,
) -> ddf.DataFrame:
    """Function to read a selection of hdf5-files, and generate a delayed dask
    dataframe from provided groups in the files. Optionally, aliases can be defined.

    Args:
        files (List[str]):
            A list of the file paths to load.
        group_names (List[str], optional):
            hdf5 group names to load. Defaults to load all groups containing "Stream"
        alias_dict (Dict[str, str], optional):
            dictionary of aliases for the dataframe columns. Keys are the hdf5
            groupnames, and values the aliases. If an alias is not found, its group
            name is used. Defaults to read the attribute "Name" from each group.
        time_stamps (bool, optional):
            Option to calculate time stamps. Defaults to False.

    Returns:
        ddf.DataFrame: The delayed Dask DataFrame
    """
    if group_names is None:
        group_names = []
    if alias_dict is None:
        alias_dict = {}

    # Read a file to parse the file structure
    test_fid = kwds.pop("test_fid", 0)
    test_proc = h5py.File(files[test_fid])
    if group_names == []:
        group_names, alias_dict = get_groups_and_aliases(
            h5file=test_proc,
            seach_pattern="Stream",
        )

    column_names = [alias_dict.get(group, group) for group in group_names]

    if time_stamps:
        column_names.append(time_stamp_alias)

    test_array = hdf5_to_array(
        h5file=test_proc,
        group_names=group_names,
        time_stamps=time_stamps,
        ms_markers_group=ms_markers_group,
        first_event_time_stamp_key=first_event_time_stamp_key,
    )

    # Delay-read all files
    arrays = [
        da.from_delayed(
            dask.delayed(hdf5_to_array)(
                h5file=h5py.File(f),
                group_names=group_names,
                time_stamps=time_stamps,
                ms_markers_group=ms_markers_group,
                first_event_time_stamp_key=first_event_time_stamp_key,
            ),
            dtype=test_array.dtype,
            shape=(test_array.shape[0], np.nan),
        )
        for f in files
    ]
    array_stack = da.concatenate(arrays, axis=1).T

    return ddf.from_dask_array(array_stack, columns=column_names)


def get_groups_and_aliases(
    h5file: h5py.File,
    seach_pattern: str = None,
    alias_key: str = "Name",
) -> Tuple[List[str], Dict[str, str]]:
    """Read groups and aliases from a provided hdf5 file handle

    Args:
        h5file (h5py.File):
            The hdf5 file handle
        seach_pattern (str, optional):
            Search pattern to select groups. Defaults to include all groups.
        alias_key (str, optional):
            Attribute key where aliases are stored. Defaults to "Name".

    Returns:
        Tuple[List[str], Dict[str, str]]:
            The list of groupnames and the alias dictionary parsed from the file
    """
    # get group names:
    group_names = list(h5file)

    # Filter the group names
    if seach_pattern is None:
        filtered_group_names = group_names
    else:
        filtered_group_names = [
            name for name in group_names if seach_pattern in name
        ]

    alias_dict = {}
    for name in filtered_group_names:
        alias_dict[name] = get_attribute(h5file[name], alias_key)

    return filtered_group_names, alias_dict


def hdf5_to_array(
    h5file: h5py.File,
    group_names: Sequence[str],
    data_type: str = "float32",
    time_stamps=False,
    ms_markers_group: str = "msMarkers",
    first_event_time_stamp_key: str = "FirstEventTimeStamp",
) -> np.ndarray:
    """Reads the content of the given groups in an hdf5 file, and returns a
    2-dimensional array with the corresponding values.

    Args:
        h5file (h5py.File):
            hdf5 file handle to read from
        group_names (str):
            group names to read
        data_type (str, optional):
            Data type of the output data. Defaults to "float32".
        time_stamps (bool, optional):
            Option to calculate time stamps. Defaults to False.

    Returns:
        np.ndarray: The 2-dimensional data array containing the values of the groups.
    """

    # Delayed array for loading an HDF5 file of reasonable size (e.g. < 1GB)

    # Read out groups:
    data_list = []
    for group in group_names:

        g_dataset = np.asarray(h5file[group])
        if bool(data_type):
            g_dataset = g_dataset.astype(data_type)
        data_list.append(g_dataset)

    # calculate time stamps
    if time_stamps:
        # create target array for time stamps
        time_stamp_data = np.zeros(len(data_list[0]))
        # the ms marker contains a list of events that occurred at full ms intervals.
        # It's monotonically increasing, and can contain duplicates
        ms_marker = np.asarray(h5file[ms_markers_group])

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
            start_time = os.path.getmtime(h5file.filename)  # convert to ms
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
            time_stamp_data[ms_marker[i] : ms_marker[i + 1]] = (
                start_time + i / 1000
            )
        # fill any remaining points
        time_stamp_data[
            ms_marker[len(ms_marker) - 1] : len(time_stamp_data)
        ] = start_time + len(ms_marker)

        data_list.append(time_stamp_data)

    return np.asarray(data_list)


def get_attribute(h5group: h5py.Group, attribute: str) -> str:
    """Reads, decodes and returns an attrubute from an hdf5 group

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


def parse_metadata(
    files: Sequence[str],  # pylint: disable=unused-argument
) -> dict:
    """Dummy

    Args:
        files (Sequence[str]): _description_

    Returns:
        dict: _description_
    """
    return {}


def get_count_rate(
    h5file: h5py.File,
    ms_markers_group: str = "msMarkers",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create count rate trace from the msMarker field in the hdf5 file.

    Parameters:
        h5file: The h5file from which to get the count rate
        ms_marker_group: The hdf5 group where the millisecond markers are stored

    Return:
        countRate: ndarray
            The count rate in Hz.
        secs: ndarray
            The seconds into the scan.

    """

    ms_markers = np.asarray(h5file[ms_markers_group])
    secs = np.arange(0, len(ms_markers)) / 1000
    msmarker_spline = sint.InterpolatedUnivariateSpline(secs, ms_markers, k=1)
    rate_spline = msmarker_spline.derivative()
    count_rate = rate_spline(secs)

    return (count_rate, secs)


def get_elapsed_time(
    h5file: h5py.File,
    ms_markers_group: str = "msMarkers",
) -> float:
    """
    Return the elapsed time in the file from the msMarkers wave

        return: secs: the acquision time of the file in seconds.
    """

    secs = h5file[ms_markers_group].len() / 1000

    return secs


class MpesLoader(BaseLoader):
    """Mpes implementation of the Loader. Reads from h5 files or folders of the
    SPECS Metis 1000 (FHI Berlin)"""

    __name__ = "mpes"

    supported_file_types = ["h5"]

    def __init__(
        self,
        config: dict = None,
        meta_handler: MetaHandler = None,
    ):
        self._config = config if config is not None else {}

        self._meta_handler = (
            meta_handler if meta_handler is not None else MetaHandler()
        )

        self.read_timestamps = self._config.get("dataframe", {}).get(
            "read_timestamps",
            False,
        )

        self.files: List[str] = []

    def read_dataframe(
        self,
        files: Sequence[str] = None,
        folder: str = None,
        ftype: str = "h5",
        time_stamps: bool = False,
        **kwds,
    ) -> Tuple[ddf.DataFrame, dict]:
        """Read stored hdf5 files from a list or from folder into a dataframe.

        Parameters:
        folder, files: str, list/tuple | None, None
            Folder path of the files or a list of file paths. The folder path has
            the priority such that if it's specified, the specified files will
            be ignored.
        extension: str | 'h5'
            File extension to use.
            If a folder path is given, all files with the specified extension are read
            into the dataframe in the reading order.
        time_stamps: bool | False
            Option to create a time_stamp column in the dataframe from ms-Markers
            in the files.

        **kwds: Keyword parameters for gather_files.

        **Return**\n
            Dask dataframe read from specified files.
        """
        # pylint: disable=duplicate-code
        if folder is not None:
            folder = os.path.realpath(folder)
            files = gather_files(
                folder=folder,
                extension=ftype,
                file_sorting=True,
                **kwds,
            )

        elif folder is None:
            if files is None:
                raise ValueError(
                    "Either the folder or file path should be provided!",
                )
            files = [os.path.realpath(file) for file in files]

        self.files = files

        if not files:
            raise FileNotFoundError("No valid files found!")

        hdf5_groupnames = kwds.pop(
            "hdf5_groupnames",
            self._config.get("dataframe", {}).get("hdf5_groupnames", []),
        )
        hdf5_aliases = kwds.pop(
            "hdf5_aliases",
            self._config.get("dataframe", {}).get("hdf5_aliases", {}),
        )
        time_stamp_alias = kwds.pop(
            "time_stamp_alias",
            self._config.get("dataframe", {}).get(
                "time_stamp_alias",
                "timeStamps",
            ),
        )
        ms_markers_group = kwds.pop(
            "ms_markers_group",
            self._config.get("dataframe", {}).get(
                "ms_markers_group",
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
            files=files,
            group_names=hdf5_groupnames,
            alias_dict=hdf5_aliases,
            time_stamps=time_stamps,
            time_stamp_alias=time_stamp_alias,
            ms_markers_group=ms_markers_group,
            first_event_time_stamp_key=first_event_time_stamp_key,
            **kwds,
        )
        metadata = parse_metadata(files=files)

        return df, metadata

    def get_count_rate(
        self,
        fids: Sequence[int] = None,
        **kwds,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create count rate data for the files specified in ``fids``.

        Parameters:
            fids: the file ids to include. None | list of file ids.
            kwds: Keyword arguments:
                ms_markers_group: Name of the hdf5 group containing the ms-markers
        """

        if fids is None:
            fids = range(0, len(self.files))

        ms_markers_group = kwds.pop(
            "ms_markers_group",
            self._config.get("dataframe", {}).get(
                "ms_markers_group",
                "msMarkers",
            ),
        )

        secs_list = []
        count_rate_list = []
        accumulated_time = 0
        for fid in fids:
            count_rate_, secs_ = get_count_rate(
                h5py.File(self.files[fid]),
                ms_markers_group=ms_markers_group,
            )
            secs_list.append((accumulated_time + secs_).T)
            count_rate_list.append(count_rate_.T)
            accumulated_time += secs_[-1]

        count_rate = np.concatenate(count_rate_list)
        secs = np.concatenate(secs_list)

        return count_rate, secs

    def get_elapsed_time(self, fids: Sequence[int] = None, **kwds):
        """
        Return the elapsed time in the file from the msMarkers wave.

        **Return**\n
            The length of the the file in seconds.
        """

        if fids is None:
            fids = range(0, len(self.files))

        ms_markers_group = kwds.pop(
            "ms_markers_group",
            self._config.get("dataframe", {}).get(
                "ms_markers_group",
                "msMarkers",
            ),
        )

        secs = 0.0
        for fid in fids:
            secs += get_elapsed_time(
                h5py.File(self.files[fid]),
                ms_markers_group=ms_markers_group,
            )

        return secs


LOADER = MpesLoader
