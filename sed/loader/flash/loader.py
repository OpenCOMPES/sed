"""
This module implements the flash data loader.
This loader currently supports hextof, wespe and instruments with similar structure.
The raw hdf5 data is combined and saved into buffer files and loaded as a dask dataframe.
The dataframe is an amalgamation of all h5 files for a combination of runs, where the NaNs are
automatically forward-filled across different files.
This can then be saved as a parquet for out-of-sed processing and reread back to access other
sed functionality.
"""
import re
import time
from collections.abc import Sequence
from pathlib import Path

import dask.dataframe as dd
from natsort import natsorted

from sed.loader.base.loader import BaseLoader
from sed.loader.flash.buffer_handler import BufferHandler
from sed.loader.flash.instruments import wespe_convert
from sed.loader.flash.metadata import MetadataRetriever


class FlashLoader(BaseLoader):
    """
    The class generates multiindexed multidimensional pandas dataframes from the new FLASH
    dataformat resolved by both macro and microbunches alongside electrons.
    Only the read_dataframe (inherited and implemented) method is accessed by other modules.
    """

    __name__ = "flash"

    supported_file_types = ["h5"]

    def __init__(self, config: dict) -> None:
        """
        Initializes the FlashLoader.

        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__(config=config)
        self.instrument: str = self._config["core"].get("instrument", "hextof")  # default is hextof
        self.raw_dir: str = None
        self.parquet_dir: str = None

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
        if "paths" in self._config["core"]:
            data_raw_dir = [
                Path(self._config["core"]["paths"].get("data_raw_dir", "")),
            ]
            data_parquet_dir = Path(
                self._config["core"]["paths"].get("data_parquet_dir", ""),
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
                self._config["dataframe"]["beamtime_dir"][self._config["core"]["beamline"]],
            )
            beamtime_dir = beamtime_dir.joinpath(f"{year}/data/{beamtime_id}/")

            # Use pathlib walk to reach the raw data directory
            data_raw_dir = []
            raw_path = beamtime_dir.joinpath("raw")

            for path in raw_path.glob("**/*"):
                if path.is_dir():
                    dir_name = path.name
                    if dir_name.startswith(("online-", "express-")):
                        data_raw_dir.append(path.joinpath(self._config["dataframe"]["daq"]))
                    elif dir_name == self._config["dataframe"]["daq"].upper():
                        data_raw_dir.append(path)

            if not data_raw_dir:
                raise FileNotFoundError("Raw data directories not found.")

            parquet_path = "processed/parquet"
            data_parquet_dir = beamtime_dir.joinpath(parquet_path)

        data_parquet_dir.mkdir(parents=True, exist_ok=True)

        self.raw_dir = str(data_raw_dir[0].resolve())
        self.parquet_dir = str(data_parquet_dir)

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
            run_id (str): The run identifier to locate.
            folders (str | Sequence[str], optional): The directory(ies) where the raw
                data is located. Defaults to config["core"]["base_folder"].
            extension (str, optional): The file extension. Defaults to "h5".

        Returns:
            list[str]: A list of path strings representing the collected file names.

        Raises:
            FileNotFoundError: If no files are found for the given run in the directory.
        """
        # Define the stream name prefixes based on the data acquisition identifier
        stream_name_prefixes = self._config["dataframe"]["stream_name_prefixes"]

        if folders is None:
            folders = self._config["core"]["base_folder"]

        if isinstance(folders, str):
            folders = [folders]

        daq = self._config["dataframe"].get("daq")

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

    def parse_metadata(self, scicat_token: str = None) -> dict:
        """Uses the MetadataRetriever class to fetch metadata from scicat for each run.

        Returns:
            dict: Metadata dictionary
            scicat_token (str, optional):: The scicat token to use for fetching metadata
        """
        metadata_retriever = MetadataRetriever(self._config["metadata"], scicat_token)
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

    def get_elapsed_time(self, fids: Sequence[int] = None, **kwds):  # noqa: ARG002
        """
        Calculates the elapsed time.

        Args:
            fids (Sequence[int]): A sequence of file IDs. Defaults to all files.
            **kwds:
                - runs: A sequence of run IDs. Takes precedence over fids.
                - aggregate: Whether to return the sum of the elapsed times across
                    the specified files or the elapsed time for each file. Defaults to True.

        Returns:
            Union[float, List[float]]: The elapsed time(s) in seconds.

        Raises:
            KeyError: If a file ID in fids or a run ID in 'runs' does not exist in the metadata.
        """
        try:
            file_statistics = self.metadata["file_statistics"]
        except Exception as exc:
            raise KeyError(
                "File statistics missing. Use 'read_dataframe' first.",
            ) from exc

        def get_elapsed_time_from_fid(fid):
            try:
                time_stamps = file_statistics[fid]["time_stamps"]
                elapsed_time = max(time_stamps) - min(time_stamps)
            except KeyError as exc:
                raise KeyError(
                    f"Timestamp metadata missing in file {fid}."
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
        runs = kwds.get("runs")
        if runs is not None:
            elapsed_times = [get_elapsed_time_from_run(run) for run in runs]
        else:
            if fids is None:
                fids = range(len(self.files))
            elapsed_times = [get_elapsed_time_from_fid(fid) for fid in fids]

        if kwds.get("aggregate", True):
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
        detector: str = "",
        force_recreate: bool = False,
        parquet_dir: str | Path = None,
        debug: bool = False,
        **kwds,
    ) -> tuple[dd.DataFrame, dd.DataFrame, dict]:
        """
        Read express data from the DAQ, generating a parquet in between.

        Args:
            files (str | Sequence[str], optional): File path(s) to process. Defaults to None.
            folders (str | Sequence[str], optional): Path to folder(s) where files are stored
                Path has priority such that if it's specified, the specified files will be ignored.
                Defaults to None.
            runs (Union[str, Sequence[str]], optional): Run identifier(s). Corresponding files will
                be located in the location provided by ``folders``. Takes precedence over
                ``files`` and ``folders``. Defaults to None.
            ftype (str, optional): The file extension type. Defaults to "h5".
            metadata (dict, optional): Additional metadata. Defaults to None.
            collect_metadata (bool, optional): Whether to collect metadata. Defaults to False.

        Returns:
            tuple[dd.DataFrame, dd.DataFrame, dict]: A tuple containing the concatenated DataFrame
            and metadata.

        Raises:
            ValueError: If neither 'runs' nor 'files'/'data_raw_dir' is provided.
            FileNotFoundError: If the conversion fails for some files or no data is available.
        """
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

        # if parquet_dir is None, use self.parquet_dir
        parquet_dir = parquet_dir or self.parquet_dir
        parquet_dir = Path(parquet_dir)

        # Obtain the parquet filenames, metadata, and schema from the method
        # which handles buffer file creation/reading
        h5_paths = [Path(file) for file in self.files]
        bh.run(
            h5_paths=h5_paths,
            folder=parquet_dir,
            force_recreate=force_recreate,
            suffix=detector,
            debug=debug,
        )
        df = bh.df_electron
        df_timed = bh.df_pulse

        if self.instrument == "wespe":
            df, df_timed = wespe_convert(df, df_timed)

        self.metadata.update(self.parse_metadata(**kwds) if collect_metadata else {})
        self.metadata.update(bh.metadata)

        print(f"loading complete in {time.time() - t0: .2f} s")

        return df, df_timed, self.metadata


LOADER = FlashLoader
