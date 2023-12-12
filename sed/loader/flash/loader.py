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

import time
from pathlib import Path
from typing import Sequence

import dask.dataframe as dd
from natsort import natsorted

from sed.loader.base.loader import BaseLoader
from sed.loader.fel import BufferFileHandler
from sed.loader.fel import ParquetHandler
from sed.loader.flash.metadata import MetadataRetriever


class FlashLoader(BaseLoader):
    """
    The class generates multiindexed multidimensional pandas dataframes from the new FLASH
    dataformat resolved by both macro and microbunches alongside electrons.
    Only the read_dataframe (inherited and implemented) method is accessed by other modules.
    """

    __name__ = "flash"

    supported_file_types = ["h5", "parquet"]

    def __init__(self, config: dict) -> None:
        """
        Initializes the FlashLoader.

        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__(config=config)

    def initialize_paths(self) -> tuple[list[Path], Path]:
        """
        Initializes the paths based on the configuration.

        Returns:
            Tuple[List[Path], Path]: A tuple containing a list of raw data directories
            paths and the parquet data directory path.

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
                daq = self._config["dataframe"]["daq"]
            except KeyError as exc:
                raise ValueError(
                    "The beamtime_id, year and daq are required.",
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
                    if dir_name.startswith("express-") or dir_name.startswith(
                        "online-",
                    ):
                        data_raw_dir.append(path.joinpath(daq))
                    elif dir_name == daq.upper():
                        data_raw_dir.append(path)

            if not data_raw_dir:
                raise FileNotFoundError("Raw data directories not found.")

            parquet_path = "processed/parquet"
            data_parquet_dir = beamtime_dir.joinpath(parquet_path)

        data_parquet_dir.mkdir(parents=True, exist_ok=True)

        return data_raw_dir, data_parquet_dir

    def get_files_from_run_id(
        self,
        run_id: str,
        folders: str | Sequence[str] = None,
        extension: str = "h5",
        **kwds,
    ) -> list[str]:
        """
        Returns a list of filenames for a given run located in the specified directory
        for the specified data acquisition (daq).

        Args:
            run_id (str): The run identifier to locate.
            folders (Union[str, Sequence[str]], optional): The directory(ies) where the raw
                data is located. Defaults to config["core"]["base_folder"].
            extension (str, optional): The file extension. Defaults to "h5".
            kwds: Keyword arguments:
                - daq (str): The data acquisition identifier.

        Returns:
            List[str]: A list of path strings representing the collected file names.

        Raises:
            FileNotFoundError: If no files are found for the given run in the directory.
        """
        # Define the stream name prefixes based on the data acquisition identifier
        stream_name_prefixes = self._config["dataframe"]["stream_name_prefixes"]

        if folders is None:
            folders = self._config["core"]["base_folder"]

        if isinstance(folders, str):
            folders = [folders]

        daq = kwds.pop("daq", self._config.get("dataframe", {}).get("daq"))

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

    def parse_metadata(self) -> dict:
        """Uses the MetadataRetriever class to fetch metadata from scicat for each run.

        Returns:
            dict: Metadata dictionary
        """
        metadata_retriever = MetadataRetriever(self._config["metadata"])
        metadata = metadata_retriever.get_metadata(
            beamtime_id=self._config["core"]["beamtime_id"],
            runs=self.runs,
            metadata=self.metadata,
        )

        return metadata

    def get_count_rate(
        self,
        fids: Sequence[int] = None,
        **kwds,
    ):
        return None, None

    def get_elapsed_time(self, fids=None, **kwds):
        return None

    def read_dataframe(
        self,
        files: str | Sequence[str] = None,
        folders: str | Sequence[str] = None,
        runs: str | Sequence[str] = None,
        ftype: str = "h5",
        metadata: dict = None,
        collect_metadata: bool = False,
        converted: bool = False,
        load_parquet: bool = False,
        save_parquet: bool = False,
        detector: str = "",
        force_recreate: bool = False,
        parquet_dir: str | Path = None,
        **kwds,
    ) -> tuple[dd.DataFrame, dd.DataFrame, dict]:
        """
        Read express data from the DAQ, generating a parquet in between.

        Args:
            files (Union[str, Sequence[str]], optional): File path(s) to process. Defaults to None.
            folders (Union[str, Sequence[str]], optional): Path to folder(s) where files are stored
                Path has priority such that if it's specified, the specified files will be ignored.
                Defaults to None.
            runs (Union[str, Sequence[str]], optional): Run identifier(s). Corresponding files will
                be located in the location provided by ``folders``. Takes precedence over
                ``files`` and ``folders``. Defaults to None.
            ftype (str, optional): The file extension type. Defaults to "h5".
            metadata (dict, optional): Additional metadata. Defaults to None.
            collect_metadata (bool, optional): Whether to collect metadata. Defaults to False.

        Returns:
            Tuple[dd.DataFrame, dd.DataFrame, dict]: A tuple containing the concatenated DataFrame
                and metadata.

        Raises:
            ValueError: If neither 'runs' nor 'files'/'data_raw_dir' is provided.
            FileNotFoundError: If the conversion fails for some files or no data is available.
        """
        t0 = time.time()

        data_raw_dir, data_parquet_dir = self.initialize_paths()

        # Prepare a list of names for the runs to read and parquets to write
        if runs is not None:
            files = []
            if isinstance(runs, (str, int)):
                runs = [runs]
            for run in runs:
                run_files = self.get_files_from_run_id(
                    run_id=run,
                    folders=[str(folder.resolve()) for folder in data_raw_dir],
                    extension=ftype,
                    daq=self._config["dataframe"]["daq"],
                )
                files.extend(run_files)
            self.runs = list(runs)
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
        parquet_dir = (
            Path(parquet_dir) or data_parquet_dir
        )  # if parquet_dir is None, use data_parquet_dir
        filename = "_".join(str(run) for run in self.runs)
        converted_str = "converted" if converted else ""
        prq = ParquetHandler(filename, parquet_dir, converted_str, "run_", detector)

        # Check if load_parquet is flagged and then load the file if it exists
        if load_parquet:
            prq.read_parquet()

        else:
            # Obtain the parquet filenames, metadata, and schema from the method
            # which handles buffer file creation/reading
            h5_paths = [Path(file) for file in self.files]
            buffer = BufferFileHandler(
                self._config["dataframe"],
                h5_paths,
                data_parquet_dir,
                force_recreate,
                suffix=detector,
            )
            df = buffer.df_electron
            df_timed = buffer.df_pulse
        # Save the dataframe as parquet if requested
        if save_parquet:
            prq.save_parquet(df, drop_index=True)

        metadata = self.parse_metadata() if collect_metadata else {}
        print(f"loading complete in {time.time() - t0: .2f} s")

        return df, df_timed, metadata


LOADER = FlashLoader
