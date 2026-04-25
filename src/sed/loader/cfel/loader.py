"""
This module implements the cfel data loader (for hextof's lab data).
This loader currently supports hextof, wespe and instruments with a similar structure.
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
import pandas as pd
from natsort import natsorted

from sed.core.logging import set_verbosity
from sed.core.logging import setup_logging
from sed.loader.base.loader import BaseLoader
from sed.loader.cfel.buffer_handler import BufferHandler
from sed.loader.flash.metadata import MetadataRetriever
# import scipy.interpolate as sint

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

    @staticmethod
    def _file_index(path: Path) -> int:
        """
        Extract file index from filename.
        Returns 0 for single-file runs.
        """
        stem = path.stem  # no extension
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])

        return 0

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
        stream_name_prefixes = self._config["core"].get("stream_name_prefixes")

        if folders is None:
            folders = self._config["core"]["base_folder"]

        if isinstance(folders, str):
            folders = [folders]

        daq = self._config["dataframe"]["daq"]

        if stream_name_prefixes:
            file_pattern = f"{stream_name_prefixes[daq]}_run{run_id}*.{extension}"
        else:
            file_pattern = f"*{run_id}*.{extension}"

        def file_index(path: Path) -> int:
            stem = path.stem
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return int(parts[1])
            return 0  # single-file run

        files: list[Path] = []
        for folder in folders:
            found = list(Path(folder).glob(file_pattern))
            # Use the static method directly
            files.extend(natsorted(found, key=self._file_index))

        if not files:
            raise FileNotFoundError(
                f"No files found for run {run_id} in directory {folders}",
            )

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
    def _unwrap_millis(self, ms: np.ndarray) -> np.ndarray:
        """
        Unwrap monotonic millisecCounter that resets to 0 at each acquisition point.
        
        Args:
            ms (np.ndarray): MillisecCounter array.
            
        Returns:
            np.ndarray: Unwrapped monotonic counter in milliseconds.
        """
        ms = np.asarray(ms, dtype=np.float64)
        if len(ms) < 2:
            return ms
        
        # Detect negative jumps (resets)
        dt = np.diff(ms)
        negative_jumps = np.where(dt < 0)[0]
        
        if len(negative_jumps) == 0:
            return ms
            
        unwrapped = ms.copy()
        offset = 0.0
        for idx in negative_jumps:
            # We assume a reset to 0 happened. We add the last value 
            # (plus 1ms to ensure strictly monotonic) to the offset.
            offset += ms[idx] + 1.0
            unwrapped[idx + 1 :] = ms[idx + 1 :] + offset
            
        return unwrapped

    def get_count_rate_ms(
        self,
        fids: Sequence[int] | None = None,
        *,
        mode: str = "file",  # "file" or "point"
        time_unit: str = "relative",  # "relative" (stitched) or "absolute" (with gaps)
        first_files: int | None = None,
        **kwds,
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
        time_unit : {"relative", "absolute"}
            - "relative": contiguous timeline, files stitched together.
            - "absolute": global timeline preserving actual gaps between runs/files.
        first_files : int or None
            If given, only the first N files are used.

        Returns
        -------
        rates : np.ndarray
            Count rate in Hz.
        times : np.ndarray
            Time in seconds.
        """
        millis_key = self._config.get("millis_counter_key", "/DLD/millisecCounter")
        counts_key = self._config.get("num_events_key", "/DLD/NumOfEvents")
        start_time_key = self._config.get("first_event_time_stamp_key", "/ScanParam/StartTime")

        fids_resolved = self._resolve_fids(fids=fids, first_files=first_files)

        # -------------------------------
        # 1) Load and process per file
        # -------------------------------
        rates_all = []
        times_all = []
        file_stats = []

        cumulative_offset = 0.0  # used for 'relative' mode
        global_start_time = None  # used for 'absolute' mode

        for fid in fids_resolved:
            with h5py.File(self.files[fid], "r") as h5:
                # Get local millisec info
                ms = np.asarray(h5[millis_key], dtype=np.float64)
                if len(ms) < 2:
                    logger.warning(f"Insufficient data in millisecCounter in file {fid}, skipping.")
                    continue

                c = (
                    np.asarray(h5[counts_key], dtype=np.float64)
                    if counts_key in h5
                    else np.ones_like(ms)
                )

                if len(ms) != len(c):
                    raise ValueError(f"Length mismatch in file {self.files[fid]}")

                # Establish temporal anchor for THIS file
                ms_unwrapped = self._unwrap_millis(ms)
                t = ms_unwrapped * 1e-3
                t_normalized = t - t[0]

                # 1) Establish temporal offset for this file
                if time_unit == "absolute":
                    if start_time_key in h5:
                        start_time_raw = np.reshape(h5[start_time_key], -1)[0]
                        current_start = pd.to_datetime(start_time_raw.decode())
                        if global_start_time is None:
                            global_start_time = current_start
                        
                        # file_offset is the gap since the VERY FIRST file in the sequence (in seconds)
                        file_offset = (current_start - global_start_time).total_seconds()
                    else:
                        logger.warning(f"StartTime missing in {fid}. Falling back to relative stitching.")
                        file_offset = cumulative_offset
                else:
                    file_offset = cumulative_offset

                # 2) Handle multiple frames in the same millisecond by grouping
                # We use the unwrapped ms to ensure chronological order is preserved
                unique_ms, index = np.unique(ms_unwrapped, return_inverse=True)
                grouped_counts = np.bincount(index, weights=c)
                
                # t_grouped is in seconds (usually starting at 0 for the file or run)
                t_grouped = unique_ms * 1e-3
                
                # We want durations BETWEEN unique ms markers
                # Note: if ms resets each file, t_grouped[0] is 0.
                # If it's monotonic across files, t_grouped[0] is > 0.
                t_normalized = t_grouped - t_grouped[0]
                dt = np.diff(t_normalized)
                
                # Avoid division by zero (should be impossible after unique() but being safe)
                valid_mask = dt > 1e-9 
                
                if mode == "point":
                    # Assign the counts that arrived in the interval [i-1, i] to the timestamp at i
                    # Note: We ignore the very first counts (grouped_counts[0]) as they often contain
                    # DAQ start-up artifacts/bursts.
                    rates_file = grouped_counts[1:][valid_mask] / dt[valid_mask]
                    times_file = t_normalized[1:][valid_mask] + file_offset
                    
                    # Apply rolling average if requested
                    bin_size = kwds.get("bin_size", 1)
                    if bin_size > 1:
                        rates_file = (
                            pd.Series(rates_file)
                            .rolling(window=bin_size, center=True, min_periods=1)
                            .mean()
                            .values
                        )
                    
                    rates_all.append(rates_file)
                    times_all.append(times_file)

                # Store stats for 'file' mode
                # We exclude grouped_counts[0] as it often contains DAQ start-up artifacts
                total_counts_file = grouped_counts[1:].sum()
                file_duration = t_normalized[-1] - t_normalized[0] if len(t_normalized) > 1 else 0
                
                print(
                    f"DEBUG: Processing file {fid} ({Path(self.files[fid]).name}). "
                    f"ST={current_start if 'current_start' in locals() else 'N/A'}, "
                    f"MS_range=({unique_ms[0]}, {unique_ms[-1]}), offset={file_offset:.2f}s, "
                    f"dur={file_duration:.2f}s, counts={total_counts_file}"
                )

                file_stats.append({
                    "rate": total_counts_file / file_duration if file_duration > 1e-9 else np.nan,
                    "time": file_offset + (file_duration / 2),
                })

                # Update relative offset for next iteration (as if they were continuous)
                cumulative_offset += file_duration

        # -------------------------------
        # 2) Assemble return values
        # -------------------------------
        if mode == "point":
            if not rates_all:
                return np.array([]), np.array([])
            return np.concatenate(rates_all), np.concatenate(times_all)

        # mode == "file"
        if not file_stats:
            return np.array([]), np.array([])
        rates_file = np.array([s["rate"] for s in file_stats])
        times_file = np.array([s["time"] for s in file_stats])
        return rates_file, times_file

    # -------------------------------
    # File-based count rate
    # -------------------------------
    def get_count_rate_simple(
        self,
        fids: Sequence[int] | None = None,
        runs: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns count rate per file using file statistics (coarse, fast method).
        Calculates the count rate using the number of rows and elapsed time for each file.
        This is a simple, fast method for coarse count rate evaluation.

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

        all_counts = [
            self.metadata["file_statistics"]["electron"][str(fid)]["num_rows"]
            for fid in fids_resolved
        ]
        # elapsed_times = self.get_elapsed_time(fids=fids_resolved)
        elapsed_times = self.get_elapsed_time_per_file(fids=fids_resolved)
        count_rate = np.array(all_counts) / np.array(elapsed_times)
        times = np.cumsum(elapsed_times)
        return count_rate, times

    def get_count_rate(
        self,
        fids: Sequence[int] | None = None,
        runs: Sequence[int] | None = None,
        method: str = "fast",  # "fast" (metadata) or "precise" (h5)
        mode: str = "file",  # "file" (1 pt/file) or "point" (intra-file)
        time_unit: str = "relative",  # "relative" or "absolute"
        **kwds,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the count rate for specified files or runs.

        By default, calculates a fast, file-resolved count rate using metadata.
        Supports high-resolution, hardware-timed rates within files when
        method='precise' is used.

        Parameters
        ----------
        fids : Sequence[int], optional
            File indices to include. Defaults to all loaded files.
        runs : Sequence[int], optional
            Run IDs to include. If provided, overrides `fids`.
        method : {"fast", "precise"}, default "fast"
            Calculation methodology:
            - "fast": Uses pre-collected metadata (very quick, low RAM).
            - "precise": Reads hardware 'millisecCounter' from H5 files.
        mode : {"file", "point"}, default "file"
            Temporal resolution:
            - "file": One average rate per file.
            - "point": Intra-file time-resolved rates (hardware or statistical).
        time_unit : {"relative", "absolute"}, default "relative"
            Temporal scale:
            - "relative": Stitched timeline (no gaps).
            - "absolute": Preserves gaps between runs using StartTime metadata.
        **kwds : dict
            Additional arguments:
            - time_bin_size (float): Binning for "fast" + "point" mode (default: 1.0s).
            - bin_size (int): Rolling average window for "precise" + "point" mode.

        Returns
        -------
        count_rate : np.ndarray
            Array of count rates in Hz.
        time : np.ndarray
            Array of global times in seconds.
        """
        fids_resolved = self._resolve_fids(fids=fids, runs=runs)

        if method == "fast":
            if mode == "file":
                all_counts = [
                    self.metadata["file_statistics"]["electron"][str(fid)]["num_rows"]
                    for fid in fids_resolved
                ]
                # Subtract the first millisecond counts from each file's total
                c0_list = []
                for fid in fids_resolved:
                    with h5py.File(self.files[fid], "r") as h5:
                        c0_list.append(h5["/DLD/NumOfEvents"][0] if "/DLD/NumOfEvents" in h5 else 0)
                # Establish file durations
                durations = self.get_elapsed_time_per_file(fids=fids_resolved)
                rates = (np.array(all_counts) - np.array(c0_list)) / np.array(durations)
                
                # Establish timeline
                if time_unit == "absolute":
                    times = []
                    global_start_time = None
                    for fid, dur in zip(fids_resolved, durations):
                        file_stats_timed = self.metadata["file_statistics"]["timed"][str(fid)]
                        time_stamps = file_stats_timed["columns"].get("timeStamp", file_stats_timed["columns"].get("timestamp"))
                        current_start = pd.to_datetime(time_stamps["min"], unit='s')
                        if global_start_time is None:
                            global_start_time = current_start
                        file_offset = (current_start - global_start_time).total_seconds()
                        times.append(file_offset + (dur / 2))
                    times = np.array(times)
                else:
                    # Stitched timeline
                    times = np.cumsum(durations) - (np.array(durations) / 2) # Centers
                return rates, times

            elif mode == "point":
                return self.get_count_rate_time_resolved(
                    fids=fids_resolved,
                    time_bin_size=kwds.get("time_bin_size", 1.0),
                    time_unit=time_unit,
                )

        elif method == "precise":
            return self.get_count_rate_ms(
                fids=fids_resolved,
                mode=mode,
                time_unit=time_unit,
                **kwds,
            )

        raise ValueError(f"Invalid method/mode combination: {method}/{mode}")

    # -------------------------------
    # Time-resolved count rate (binned)
    # -------------------------------
    def get_count_rate_time_resolved(
        self,
        fids: Sequence[int] | None = None,
        time_bin_size: float = 1.0,
        runs: Sequence[int] | None = None,
        time_unit: str = "relative",
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
        cumulative_offset = 0.0
        global_start_time = None

        for fid in fids_resolved:
            file_statistics = self.metadata["file_statistics"]["timed"]
            time_stamp_alias = self._config["dataframe"]["columns"].get("timestamp", "timeStamp")
            time_stamps = file_statistics[str(fid)]["columns"][time_stamp_alias]

            t_min = float(
                getattr(time_stamps["min"], "total_seconds", lambda: time_stamps["min"])(),
            )
            t_max = float(
                getattr(time_stamps["max"], "total_seconds", lambda: time_stamps["max"])(),
            )
            # Subtract the first millisecond counts from the total counts
            with h5py.File(self.files[fid], "r") as h5:
                c0 = h5["/DLD/NumOfEvents"][0] if "/DLD/NumOfEvents" in h5 else 0
            
            total_counts = (self.metadata["file_statistics"]["electron"][str(fid)]["num_rows"] - c0)
            file_duration = t_max - t_min
            
            # Establish temporal offset for this file
            if time_unit == "absolute":
                file_stats_timed = self.metadata["file_statistics"]["timed"][str(fid)]
                time_stamps_full = file_stats_timed["columns"].get("timeStamp", file_stats_timed["columns"].get("timestamp"))
                current_start = pd.to_datetime(time_stamps_full["min"], unit='s')
                if global_start_time is None:
                    global_start_time = current_start
                file_offset = (current_start - global_start_time).total_seconds()
            else:
                file_offset = cumulative_offset

            print(f"DEBUG: Processing file {fid} (Fast method). duration={file_duration}, offset={file_offset}")

            n_bins = max(int(file_duration / time_bin_size), 1)
            rate_per_bin = total_counts / file_duration if file_duration > 1e-9 else 0.0

            bin_centers = np.linspace(
                file_offset + time_bin_size / 2,
                file_offset + file_duration - time_bin_size / 2,
                n_bins,
            )

            rates = np.full(n_bins, rate_per_bin)
            all_rates.append(rates)
            all_times.append(bin_centers)

            cumulative_offset += file_duration

        if not all_rates:
            return np.array([]), np.array([])

        return np.concatenate(all_rates), np.concatenate(all_times)

    def get_elapsed_time(
        self,
        fids: Sequence[int] | None = None,
        aggregate: bool = True,
        **kwds,
    ) -> "float | np.ndarray":
        """
        Return elapsed time for specified files.

        Args:
            fids (Sequence[int] | None): File IDs to include. Defaults to all.
            aggregate (bool): If True (default), return the total elapsed time as a
                scalar float. If False, return a per-file numpy array.
            kwds: Optional keyword arguments:
                - runs (Sequence[int] | None)
                - precise (bool)

        Returns:
            float | np.ndarray: Total elapsed time (seconds) or per-file array.
        """
        runs = kwds.pop("runs", None)
        precise = kwds.pop("precise", False)
        # Silently ignore any remaining unrecognised kwargs for forward-compatibility
        if kwds:
            logger.warning("get_elapsed_time() ignoring unexpected keyword arguments: %s", list(kwds.keys()))

        fids_resolved = self._resolve_fids(fids=fids, runs=runs)

        elapsed_per_file = np.array(
            [self._get_elapsed_time_single(fid, precise=precise) for fid in fids_resolved],
        )

        if aggregate:
            return float(elapsed_per_file.sum())
        return elapsed_per_file

    def get_elapsed_time_per_file(
        self,
        fids: Sequence[int] | None = None,
        runs: Sequence[int] | None = None,
        precise: bool = False,
    ) -> np.ndarray:
        """Return elapsed times per file as a NumPy array (not part of BaseLoader)."""
        fids_resolved = self._resolve_fids(fids=fids, runs=runs)
        return np.array(
            [self._get_elapsed_time_single(fid, precise=precise) for fid in fids_resolved],
        )

    # -------------------------------
    # Internal helper: single file
    # -------------------------------
    def _get_elapsed_time_single(
        self,
        fid: int,
        *,
        precise: bool = False,
    ) -> float:
        """
        Compute the elapsed acquisition time for a single file.

        This method first tries to use pre-stored metadata (fast).
        If metadata is missing or `precise=True`, it reads the
        hardware millisecCounter from the HDF5 file (slower but accurate).

        Args:
            fid (int): Index of the file to process.
            precise (bool, optional): If True, forces reading the
                HDF5 hardware counter instead of using metadata.
                Default: False.

        Returns:
            float: Elapsed time in seconds for this file.
        """
        dt_s: float | None = None

        # 1. Try metadata (fast path)
        if not precise:
            try:
                file_stats = self.metadata["file_statistics"]["timed"][str(fid)]
                time_stamps = file_stats["columns"].get(
                    "timeStamp",
                    file_stats["columns"].get("timestamp"),
                )

                t_min = time_stamps["min"]
                t_max = time_stamps["max"]

                t1 = t_min.total_seconds() if hasattr(t_min, "total_seconds") else float(t_min)
                t2 = t_max.total_seconds() if hasattr(t_max, "total_seconds") else float(t_max)

                dt_s = t2 - t1
            except (KeyError, TypeError):
                logger.debug(
                    "Metadata duration missing for fid %s, falling back to H5.",
                    fid,
                )

        # 2. HDF5 millisecCounter (fallback or precise)
        if dt_s is None:
            millis_key = self._config.get("millis_counter_key", "/DLD/millisecCounter")
            try:
                with h5py.File(self.files[fid], "r") as h5:
                    ms = np.asarray(h5[millis_key], dtype=np.float64)
                    ms_unwrapped = self._unwrap_millis(ms)
                    dt_s = (ms_unwrapped[-1] - ms_unwrapped[0]) / 1000.0
            except (KeyError, IndexError):
                logger.warning(
                    "Could not determine duration for fid %s. Using 0.0",
                    fid,
                )
                dt_s = 0.0

        return float(dt_s)

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
                Path has priority, so if it's specified, the specified files will be ignored.
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
            FileNotFoundError: If the conversion fails for some files, or no data is available.
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

        scicat_metadata = self.parse_scicat_metadata(token)
        scicat_runs = scicat_metadata.get("scientificMetadata", {})

        if not any(scicat_runs.values()):
            logger.warning("No SciCat metadata available, checking local folder")
            self.metadata.update(self.parse_local_metadata())
        else:
            logger.warning("Metadata taken from SciCat")
            if collect_metadata:
                self.metadata.update(scicat_metadata)

        self.metadata.update(bh.metadata)

        print(f"loading complete in {time.time() - t0: .2f} s")

        return df, df_timed, self.metadata


LOADER = CFELLoader
