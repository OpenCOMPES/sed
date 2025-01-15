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
from natsort import natsorted

from sed.core.logging import set_verbosity
from sed.core.logging import setup_logging
from sed.loader.base.loader import BaseLoader
from sed.loader.flash.buffer_handler import BufferHandler
from sed.loader.flash.instruments import wespe_convert
from sed.loader.flash.metadata import MetadataRetriever

# Configure logging
logger = setup_logging("flash_loader")


class FlashLoader(BaseLoader):
    """
    The class generates multiindexed multidimensional pandas dataframes from the new FLASH
    dataformat resolved by both macro and microbunches alongside electrons.
    Only the read_dataframe (inherited and implemented) method is accessed by other modules.

    Args:
        config (dict, optional): Config dictionary. Defaults to None.
        verbose (bool, optional): Option to print out diagnostic information.
            Defaults to True.
    """

    __name__ = "flash"

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
        self.raw_dir: str = None
        self.processed_dir: str = None

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
            processed_dir = Path(
                self._config["core"]["paths"].get("processed", raw_dir.joinpath("processed")),
            )

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

        processed_dir.mkdir(parents=True, exist_ok=True)

        self.raw_dir = str(raw_dir)
        self.processed_dir = str(processed_dir)

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
        stream_name_prefixes = self._config["core"]["stream_name_prefixes"]

        if folders is None:
            folders = self._config["core"]["base_folder"]

        if isinstance(folders, str):
            folders = [folders]

        daq = self._config["dataframe"]["daq"]

        # Generate the file patterns to search for in the directory
        file_pattern = f"{stream_name_prefixes[daq]}_run{run_id}_*." + extension

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

    def parse_metadata(self, token: str = None) -> dict:
        """Uses the MetadataRetriever class to fetch metadata from scicat for each run.

        Returns:
            dict: Metadata dictionary
            token (str, optional):: The scicat token to use for fetching metadata
        """
        metadata_retriever = MetadataRetriever(self._config["metadata"], token)
        metadata = metadata_retriever.get_metadata(
            beamtime_id=self._config["core"]["beamtime_id"],
            runs=self.runs,
            metadata=self.metadata,
        )

        return metadata

    def get_count_rate(
        self,
        fids: Sequence[int] = None,  # noqa: ARG002
        **kwds,  # noqa: ARG002
    ):
        return None, None

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
        time_stamp_alias = self._config["dataframe"].get("time_stamp_alias", "timeStamp")

        def get_elapsed_time_from_fid(fid):
            try:
                fid = str(fid)  # Ensure the key is a string
                time_stamps = file_statistics[fid]["columns"][time_stamp_alias]
                elapsed_time = time_stamps["max"] - time_stamps["min"]
            except KeyError as exc:
                raise KeyError(
                    f"Timestamp metadata missing in file {fid}. "
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
        metadata: dict = {},
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

        if self.instrument == "wespe":
            df, df_timed = wespe_convert(df, df_timed)

        self.metadata.update(self.parse_metadata(token) if collect_metadata else {})
        self.metadata.update(bh.metadata)

        print(f"loading complete in {time.time() - t0: .2f} s")

        return df, df_timed, self.metadata


LOADER = FlashLoader
