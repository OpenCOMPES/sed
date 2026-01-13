"""
This module implements the flash data loader.
This loader currently supports hextof, wespe and instruments with similar structure.
The raw hdf5 data is combined and saved into buffer files and loaded as a dask dataframe.
The dataframe is an amalgamation of all h5 files for a combination of runs, where the NaNs are
automatically forward-filled across different files.
This can then be saved as a parquet for out-of-sed processing and reread back to access other
sed functionality.
"""
from __future__ import annotations

import re
import time
from collections.abc import Sequence
from pathlib import Path

import dask.dataframe as dd
import h5py
import numpy as np
import scipy.interpolate as sint
from natsort import natsorted

from sed.core.logging import set_verbosity
from sed.core.logging import setup_logging
from sed.loader.base.loader import BaseLoader
from sed.loader.cfel.buffer_handler import BufferHandler
from sed.loader.flash.metadata import MetadataRetriever

import pandas as pd

# Configure logging
logger = setup_logging("flash_loader")


class CFELLoader(BaseLoader):
    """
    The class generates multiindexed multidimensional pandas dataframes from the new FLASH
    dataformat resolved by both macro and microbunches alongside electrons.
    Only the read_dataframe (inherited and implemented) method is accessed by other modules.

    Args:
        config (dict, optional): Config dictionary. Defaults to None.
        verbose (bool, optional): Option to print out diagnostic information.
            Defaults to True.
    """

    __name__ = "cfel"

    supported_file_types = ["h5"]

    def __init__(self, config: dict, verbose: bool = True) -> None:
        """
        Initializes the FlashLoader.

        Args:
            config (dict): Configuration dictionary.
            verbose (bool, optional): Option to print out diagnostic information.
        """
        super().__init__(config=config, verbose=verbose)

        set_verbosity(logger, self._verbose)

        self.instrument: str = self._config["core"].get("instrument", "hextof")  # default is hextof
        self.beamtime_dir: str = None
        self.raw_dir: str = None
        self.processed_dir: str = None
        self.meta_dir: str = None

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

    def __len__(self) -> int:
        """
        Returns the total number of rows in the electron resolved dataframe.

        Returns:
            int: Total number of rows.
        """
        try:
            file_statistics = self.metadata["file_statistics"]["electron"]
        except KeyError as exc:
            raise KeyError("File statistics missing. Use 'read_dataframe' first.") from exc

        total_rows = sum(stats["num_rows"] for stats in file_statistics.values())
        return total_rows


    def _initialize_dirs(self) -> None:
        """
        Initializes the directories on Maxwell based on configuration. If paths is provided in
        the configuration, the raw data directory and parquet data directory are taken from there.
        Otherwise, the beamtime_id and year are used to locate the data directories.
        The first path that has either online- or express- prefix, or the daq name is taken as the
        raw data directory.

        Raises:
            ValueError: If required values are missing from the configuration.
            FileNotFoundError: If the raw data directories are not found.
        """
        # Parses to locate the raw beamtime directory from config file
        # Only raw_dir is necessary, processed_dir can be based on raw_dir, if not provided
        if "paths" in self._config["core"]:
            raw_dir = Path(self._config["core"]["paths"].get("raw", ""))
            print(raw_dir)
            processed_dir = Path(
                self._config["core"]["paths"].get("processed", raw_dir.joinpath("processed")),
            )
            meta_dir = Path(
                self._config["core"]["paths"].get("meta", raw_dir.joinpath("meta")),
            )
            beamtime_dir = Path(raw_dir).parent

        else:
            try:
                beamtime_id = self._config["core"]["beamtime_id"]
                year = self._config["core"]["year"]

            except KeyError as exc:
                raise ValueError(
                    "The beamtime_id and year are required.",
                ) from exc

            beamtime_dir = Path(
                self._config["core"]["beamtime_dir"][self._config["core"]["beamline"]],
            )
            beamtime_dir = beamtime_dir.joinpath(f"{year}/data/{beamtime_id}/")

            # Use pathlib walk to reach the raw data directory
            raw_paths: list[Path] = []

            for path in beamtime_dir.joinpath("raw").glob("**/*"):
                if path.is_dir():
                    dir_name = path.name
                    if dir_name.startswith(("online-", "express-")):
                        raw_paths.append(path.joinpath(self._config["dataframe"]["daq"]))
                    elif dir_name == self._config["dataframe"]["daq"].upper():
                        raw_paths.append(path)

            if not raw_paths:
                raise FileNotFoundError("Raw data directories not found.")

            raw_dir = raw_paths[0].resolve()

            processed_dir = beamtime_dir.joinpath("processed")
            meta_dir = beamtime_dir.joinpath("meta/fabtrack/")  # cspell:ignore fabtrack

        processed_dir.mkdir(parents=True, exist_ok=True)

        self.beamtime_dir = str(beamtime_dir)
        self.raw_dir = str(raw_dir)
        self.processed_dir = str(processed_dir)
        self.meta_dir = str(meta_dir)

    @property
    def available_runs(self) -> list[int]:
        # Get all files in raw_dir with "run" in their names
        files = list(Path(self.raw_dir).glob("*run*"))

        # Extract run IDs from filenames
        run_ids = set()
        for file in files:
            match = re.search(r"run(\d+)", file.name)
            if match:
                run_ids.add(int(match.group(1)))

        # Return run IDs in sorted order
        return sorted(list(run_ids))

    def get_files_from_run_id(  # type: ignore[override]
        self,
        run_id: str | int,
        folders: str | Sequence[str] = None,
        extension: str = "h5",
    ) -> list[str]:
        """
        Returns a list of filenames for a given run located in the specified directory
        for the specified data acquisition (daq).

        Args:
            run_id (str | int): The run identifier to locate.
            folders (str | Sequence[str], optional): The directory(ies) where the raw
                data is located. Defaults to config["core"]["base_folder"].
            extension (str, optional): The file extension. Defaults to "h5".

        Returns:
            list[str]: A list of path strings representing the collected file names.

        Raises:
            FileNotFoundError: If no files are found for the given run in the directory.
        """
        # Define the stream name prefixes based on the data acquisition identifier
        stream_name_prefixes = self._config["core"].get("stream_name_prefixes")

        if folders is None:
            folders = self._config["core"]["base_folder"]

        if isinstance(folders, str):
            folders = [folders]

        daq = self._config["dataframe"]["daq"]

        # Generate the file patterns to search for in the directory
        if stream_name_prefixes:
            file_pattern = f"{stream_name_prefixes[daq]}_run{run_id}_*." + extension
        else:
            file_pattern = f"*{run_id}*." + extension

        files: list[Path] = []
        # Use pathlib to search for matching files in each directory
        for folder in folders:
            files.extend(
                natsorted(
                    Path(folder).glob(file_pattern),
                    key=lambda filename: str(filename).rsplit("_", maxsplit=1)[-1],
                ),
            )

        # Check if any files are found
        if not files:
            raise FileNotFoundError(
                f"No files found for run {run_id} in directory {str(folders)}",
            )

        # Return the list of found files
        return [str(file.resolve()) for file in files]

    def _resolve_fids(
        self,
        fids: Sequence[int] | None = None,
        runs: Sequence[int] | None = None,
        first_files: int | None = None,
    ) -> list[int]:
        """
        Resolve run IDs or file IDs into a list of file indices into self.files.
        Ensures consistent ordering in acquisition time.
    
        Parameters
        ----------
        fids : Sequence[int] | None
            Specific file indices to use.
        runs : Sequence[int] | None
            Run IDs to include.
        first_files : int | None
            If given, limits the result to the first N files.
    
        Returns
        -------
        list[int]
            List of file indices in acquisition order.
        """
        if runs is not None:
            fids_resolved = []
            for run_id in runs:
                if self.raw_dir is None:
                    self._initialize_dirs()
                files_in_run = self.get_files_from_run_id(run_id=run_id, folders=self.raw_dir)
                fids_resolved.extend([self.files.index(f) for f in files_in_run])
        elif fids is not None:
            fids_resolved = list(fids)
        else:
            fids_resolved = list(range(len(self.files)))
    
        if first_files is not None:
            fids_resolved = fids_resolved[:first_files]
    
        return fids_resolved


    def parse_scicat_metadata(self, token: str = None) -> dict:
        """Uses the MetadataRetriever class to fetch metadata from scicat for each run.

        Returns:
            dict: Metadata dictionary
            token (str, optional):: The scicat token to use for fetching metadata
        """
        if "metadata" not in self._config:
            return {}
            
        metadata_retriever = MetadataRetriever(self._config["metadata"], token)
        metadata = metadata_retriever.get_metadata(
            beamtime_id=self._config["core"]["beamtime_id"],
            runs=self.runs,
            metadata=self.metadata,
        )

        return metadata

    def parse_local_metadata(self) -> dict:
        """Uses the MetadataRetriever class to fetch metadata from local folder for each run.

        Returns:
            dict: Metadata dictionary
        """
        if "metadata" not in self._config:
            return {}
            
        metadata_retriever = MetadataRetriever(self._config["metadata"])
        metadata = metadata_retriever.get_local_metadata(
            beamtime_id=self._config["core"]["beamtime_id"],
            beamtime_dir=self.beamtime_dir,
            meta_dir=self.meta_dir,
            runs=self.runs,
            metadata=self.metadata,
        )

        return metadata

    # -------------------------------
    # Count rate with millisecCounter
    # -------------------------------
    def get_count_rate_ms(
        self,
        fids: Sequence[int] | None = None,
        *,
        mode: str = "file",       # "file" or "point"
        first_files: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Count-rate calculation using millisecCounter and NumOfEvents.

        Parameters
        ----------
        fids : Sequence[int] or None
            File IDs to include. Default: all.
        mode : {"file", "point"}
            - "point": rate per acquisition window
            - "file" : one average rate per file
        first_files : int or None
            If given, only the first N files are used.

        Returns
        -------
        rates : np.ndarray
            Count rate in Hz.
        times : np.ndarray
            Time in seconds (window end time for point mode, last time per file for file mode)
        """
        millis_key = self._config.get("millis_counter_key", "/DLD/millisecCounter")
        counts_key = self._config.get("num_events_key", "/DLD/NumOfEvents")

        fids_resolved = self._resolve_fids(fids=fids, first_files=first_files)

        # -------------------------------
        # 1) Load and concatenate
        # -------------------------------
        ms_all = []
        counts_all = []
        file_sizes = []

        for fid in fids_resolved:
            with h5py.File(self.files[fid], "r") as h5:
                ms = np.asarray(h5[millis_key], dtype=np.float64)
                c = np.asarray(h5[counts_key], dtype=np.float64) if counts_key in h5 else np.ones_like(ms)

                if len(ms) != len(c):
                    raise ValueError(f"Length mismatch in file {self.files[fid]}")

                ms_all.append(ms)
                counts_all.append(c)
                file_sizes.append(len(ms))

        ms = np.concatenate(ms_all)
        counts = np.concatenate(counts_all)

        # -------------------------------
        # 2) Ensure global time order
        # -------------------------------
        order = np.argsort(ms)
        ms = ms[order]
        counts = counts[order]

        # -------------------------------
        # 3) Compute point-resolved rates
        # -------------------------------
        dt = np.diff(ms) * 1e-3
        if np.any(dt <= 0):
            raise ValueError("Non-positive time step detected in millisecCounter")

        rates_point = counts[1:] / dt
        times_point = ms[1:] * 1e-3

        if mode == "point":
            return rates_point, times_point

        # -------------------------------
        # 4) Compute file-resolved rates
        # -------------------------------
        rates_file = []
        times_file = []

        idx = 0
        for n in file_sizes:
            if n < 2:
                idx += n
                continue
            ms_f = ms[idx:idx + n]
            c_f = counts[idx:idx + n]

            dt_f = np.diff(ms_f) * 1e-3
            rate = c_f[1:].sum() / dt_f.sum()
            time = ms_f[-1] * 1e-3

            rates_file.append(rate)
            times_file.append(time)
            idx += n

        return np.asarray(rates_file), np.asarray(times_file)

    # -------------------------------
    # File-based count rate
    # -------------------------------
    def get_count_rate(
        self,
        fids: Sequence[int] | None = None,
        runs: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns count rate per file using the total number of events and elapsed time.
        Calculates the count rate using the number of rows and elapsed time for each file.
        Hence the resolution is not very high, but this method is very fast.

        Args:
            fids (Sequence[int]): A sequence of file IDs. Defaults to all files.

        Keyword Args:
            runs: A sequence of run IDs.

        Returns:
            tuple[np.ndarray, np.ndarray]: The count rate and elapsed time in seconds.

        Raises:
            KeyError: If the file statistics are missing.
        """
        fids_resolved = self._resolve_fids(fids=fids, runs=runs)

        all_counts = [self.metadata["file_statistics"]["electron"][str(fid)]["num_rows"] for fid in fids_resolved]
        elapsed_times = [self.get_elapsed_time(fids=[fid]) for fid in fids_resolved]

        count_rate = np.array(all_counts) / np.array(elapsed_times)
        times = np.cumsum(elapsed_times)
        return count_rate, times

    # -------------------------------
    # Time-resolved count rate (binned)
    # -------------------------------
    def get_count_rate_time_resolved(
        self,
        fids: Sequence[int] | None = None,
        time_bin_size: float = 1.0,
        runs: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns count rate in time bins using metadata timestamps.
        Calculates the count rate over time within each file using timestamp binning.
        
        Args:
            fids (Sequence[int]): A sequence of file IDs. Defaults to all files.
            time_bin_size (float): Time bin size in seconds for rate calculation. Defaults to 1.0.
            
        Keyword Args:
            runs: A sequence of run IDs.
            
        Returns:
            tuple[np.ndarray, np.ndarray]: The count rate array and time array in seconds.
            
        Raises:
            KeyError: If the file statistics are missing.
        """
        fids_resolved = self._resolve_fids(fids=fids, runs=runs)

        all_rates = []
        all_times = []
        cumulative_time = 0.0

        for fid in fids_resolved:
            file_statistics = self.metadata["file_statistics"]["timed"]
            time_stamp_alias = self._config["dataframe"]["columns"].get("timestamp", "timeStamp")
            time_stamps = file_statistics[str(fid)]["columns"][time_stamp_alias]

            t_min = float(getattr(time_stamps["min"], "total_seconds", lambda: time_stamps["min"])())
            t_max = float(getattr(time_stamps["max"], "total_seconds", lambda: time_stamps["max"])())
            total_counts = self.metadata["file_statistics"]["electron"][str(fid)]["num_rows"]
            file_duration = t_max - t_min

            n_bins = max(int(file_duration / time_bin_size), 1)
            counts_per_bin = total_counts / n_bins
            rate_per_bin = counts_per_bin / time_bin_size

            bin_centers = np.linspace(
                cumulative_time + time_bin_size / 2,
                cumulative_time + file_duration - time_bin_size / 2,
                n_bins,
            )

            rates = np.full(n_bins, rate_per_bin)
            all_rates.extend(rates)
            all_times.extend(bin_centers)

            cumulative_time += file_duration

        return np.array(all_rates), np.array(all_times)

    def get_elapsed_time(self, fids: Sequence[int] = None, **kwds) -> float | list[float]:  # type: ignore[override]
        """
        Calculates the elapsed time.

        Args:
            fids (Sequence[int]): A sequence of file IDs. Defaults to all files.

        Keyword Args:
            runs: A sequence of run IDs. Takes precedence over fids.
            aggregate: Whether to return the sum of the elapsed times across
                    the specified files or the elapsed time for each file. Defaults to True.

        Returns:
            float | list[float]: The elapsed time(s) in seconds.

        Raises:
            KeyError: If a file ID in fids or a run ID in 'runs' does not exist in the metadata.
        """
        try:
            file_statistics = self.metadata["file_statistics"]["timed"]
        except Exception as exc:
            raise KeyError(
                "File statistics missing. Use 'read_dataframe' first.",
            ) from exc
        time_stamp_alias = self._config["dataframe"]["columns"].get("timestamp", "timeStamp")

        def get_elapsed_time_from_fid(fid):
            try:
                fid_str = str(fid)  # Ensure the key is a string
                filename = Path(self.files[fid]).name if fid < len(self.files) else f"file_{fid}"
                time_stamps = file_statistics[fid_str]["columns"][time_stamp_alias]
                elapsed_time = time_stamps["max"] - time_stamps["min"]
                
                # Convert to seconds if it's a Timedelta object
                if hasattr(elapsed_time, 'total_seconds'):
                    elapsed_time = elapsed_time.total_seconds()
                elif hasattr(elapsed_time, 'seconds'):
                    elapsed_time = float(elapsed_time.seconds)
                else:
                    elapsed_time = float(elapsed_time)
                    
            except KeyError as exc:
                filename = Path(self.files[fid]).name if fid < len(self.files) else f"file_{fid}"
                raise KeyError(
                    f"Timestamp metadata missing in file {filename} (fid: {fid_str}). "
                    "Add timestamp column and alias to config before loading.",
                ) from exc

            return elapsed_time

        def get_elapsed_time_from_run(run_id):
            if self.raw_dir is None:
                self._initialize_dirs()
            files = self.get_files_from_run_id(run_id=run_id, folders=self.raw_dir)
            fids = [self.files.index(file) for file in files]
            return sum(get_elapsed_time_from_fid(fid) for fid in fids)

        elapsed_times = []
        runs = kwds.pop("runs", None)
        aggregate = kwds.pop("aggregate", True)

        if len(kwds) > 0:
            raise TypeError(f"get_elapsed_time() got unexpected keyword arguments {kwds.keys()}.")

        if runs is not None:
            elapsed_times = [get_elapsed_time_from_run(run) for run in runs]
        else:
            if fids is None:
                fids = range(len(self.files))
            elapsed_times = [get_elapsed_time_from_fid(fid) for fid in fids]

        if aggregate:
            elapsed_times = sum(elapsed_times)

        return elapsed_times

    def read_dataframe(
        self,
        files: str | Sequence[str] = None,
        folders: str | Sequence[str] = None,
        runs: str | int | Sequence[str | int] = None,
        ftype: str = "h5",
        metadata: dict | None = None,
        collect_metadata: bool = False,
        **kwds,
    ) -> tuple[dd.DataFrame, dd.DataFrame, dict]:
        """
        Read express data from the DAQ, generating a parquet in between.

        Args:
            files (str | Sequence[str], optional): File path(s) to process. Defaults to None.
            folders (str | Sequence[str], optional): Path to folder(s) where files are stored
                Path has priority such that if it's specified, the specified files will be ignored.
                Defaults to None.
            runs (str | int | Sequence[str | int], optional): Run identifier(s).
                Corresponding files will be located in the location provided by ``folders``.
                Takes precedence over ``files`` and ``folders``. Defaults to None.
            ftype (str, optional): The file extension type. Defaults to "h5".
            metadata (dict, optional): Additional metadata. Defaults to None.
            collect_metadata (bool, optional): Whether to collect metadata. Defaults to False.

        Keyword Args:
            detector (str, optional): The detector to use. Defaults to "".
            force_recreate (bool, optional): Whether to force recreation of the buffer files.
                Defaults to False.
            processed_dir (str, optional): The directory to save the processed files.
                Defaults to None.
            debug (bool, optional): Whether to run buffer creation in serial. Defaults to False.
            remove_invalid_files (bool, optional): Whether to exclude invalid files.
                Defaults to False.
            token (str, optional): The scicat token to use for fetching metadata. If provided,
                will be saved to .env file for future use. If not provided, will check environment
                variables when collect_metadata is True.
            filter_timed_by_electron (bool, optional): When True, the timed dataframe will only
                contain data points where valid electron events were detected. When False, all
                timed data points are included regardless of electron detection. Defaults to True.

        Returns:
            tuple[dd.DataFrame, dd.DataFrame, dict]: A tuple containing the concatenated DataFrame
            and metadata.

        Raises:
            ValueError: If neither 'runs' nor 'files'/'raw_dir' is provided.
            FileNotFoundError: If the conversion fails for some files or no data is available.
            ValueError: If collect_metadata is True and no token is available.
        """
        if metadata is None:
            metadata = {}
        
        detector = kwds.pop("detector", "")
        force_recreate = kwds.pop("force_recreate", False)
        processed_dir = kwds.pop("processed_dir", None)
        debug = kwds.pop("debug", False)
        remove_invalid_files = kwds.pop("remove_invalid_files", False)
        token = kwds.pop("token", None)
        filter_timed_by_electron = kwds.pop("filter_timed_by_electron", True)

        if len(kwds) > 0:
            raise ValueError(f"Unexpected keyword arguments: {kwds.keys()}")
        t0 = time.time()

        self._initialize_dirs()
        # Prepare a list of names for the runs to read and parquets to write
        if runs is not None:
            files = []
            runs_ = [str(runs)] if isinstance(runs, (str, int)) else list(map(str, runs))
            for run in runs_:
                run_files = self.get_files_from_run_id(
                    run_id=run,
                    folders=self.raw_dir,
                )
                files.extend(run_files)
            self.runs = runs_
            super().read_dataframe(files=files, ftype=ftype)
        else:
            # This call takes care of files and folders. As we have converted runs into files
            # already, they are just stored in the class by this call.
            super().read_dataframe(
                files=files,
                folders=folders,
                ftype=ftype,
                metadata=metadata,
            )

        bh = BufferHandler(
            config=self._config,
        )

        # if processed_dir is None, use self.processed_dir
        processed_dir = processed_dir or self.processed_dir
        processed_dir = Path(processed_dir)

        # Obtain the parquet filenames, metadata, and schema from the method
        # which handles buffer file creation/reading
        h5_paths = [Path(file) for file in self.files]
        df, df_timed = bh.process_and_load_dataframe(
            h5_paths=h5_paths,
            folder=processed_dir,
            force_recreate=force_recreate,
            suffix=detector,
            debug=debug,
            remove_invalid_files=remove_invalid_files,
            filter_timed_by_electron=filter_timed_by_electron,
        )

        if len(self.parse_scicat_metadata(token)) == 0:
            logger.warning("No SciCat metadata available, checking local folder")
            self.metadata.update(self.parse_local_metadata())
        else:
            logger.warning("Metadata taken from SciCat")
            self.metadata.update(self.parse_scicat_metadata(token) if collect_metadata else {})
        self.metadata.update(bh.metadata)

        print(f"loading complete in {time.time() - t0: .2f} s")

        return df, df_timed, self.metadata




LOADER = CFELLoader
