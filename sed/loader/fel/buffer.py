from __future__ import annotations

from itertools import compress
from pathlib import Path

import dask.dataframe as ddf
import numpy as np
import pyarrow.parquet as pq
from joblib import delayed
from joblib import Parallel

from sed.core.dfops import forward_fill_lazy
from sed.loader.fel.dataframe import DataFrameCreator
from sed.loader.fel.parquet import ParquetHandler


class BufferFileHandler(ParquetHandler, DataFrameCreator):
    def __init__(
        self,
        config_dataframe: dict,
        h5_paths: list[Path],
        folder: Path,
        force_recreate: bool,
        prefix: str = "",
        suffix: str = "",
    ):
        if len(h5_paths):
            raise ValueError("No data available. Probably failed reading all h5 files")

        h5_filenames = [Path(h5_path).stem for h5_path in h5_paths]
        super().__init__(config_dataframe)
        super().__init__(h5_filenames, folder, "buffer", prefix, suffix)

        if not force_recreate:
            self.schema_check()

        self.parallel_buffer_file_creation(h5_paths, force_recreate)

    def schema_check(self) -> None:
        """
        Checks the schema of the Parquet files.

        Raises:
            ValueError: If the schema of the Parquet files do not match the configuration.
        """
        existing_parquet_filenames = [file for file in self.parquet_paths if file.exists()]
        # Check if the available channels match the schema of the existing parquet files
        parquet_schemas = [pq.read_schema(file) for file in existing_parquet_filenames]
        config_schema = set(self.get_channels(formats="all", index=True))
        if self._config.get("split_sector_id_from_dld_time", False):
            config_schema.add(self._config.get("sector_id_column", False))

        for i, schema in enumerate(parquet_schemas):
            schema_set = set(schema.names)
            if schema_set != config_schema:
                missing_in_parquet = config_schema - schema_set
                missing_in_config = schema_set - config_schema

                missing_in_parquet_str = (
                    f"Missing in parquet: {missing_in_parquet}" if missing_in_parquet else ""
                )
                missing_in_config_str = (
                    f"Missing in config: {missing_in_config}" if missing_in_config else ""
                )

                raise ValueError(
                    "The available channels do not match the schema of file",
                    f"{existing_parquet_filenames[i]}",
                    f"{missing_in_parquet_str}",
                    f"{missing_in_config_str}",
                    "Please check the configuration file or set force_recreate to True.",
                )

    def parallel_buffer_file_creation(self, h5_paths: list[Path], force_recreate: bool) -> None:
        """
        Parallelizes the creation of buffer files.

        Args:
            h5_paths (List[Path]): List of paths to the input HDF5 files.

        Raises:
            ValueError: If an error occurs during the conversion process.

        """
        to_read = [
            force_recreate or not parquet_path.exists() for parquet_path in self.parquet_paths
        ]
        num_files = sum(to_read)

        h5_to_read = list(compress(h5_paths, to_read))
        parquets_to_read = list(compress(self.parquet_paths, to_read))

        print(f"Reading files: {num_files} new files of {len(h5_paths)} total.")

        # Initialize the indices for create_buffer_file conversion
        self.reset_multi_index()

        if num_files > 0:
            dataframes = Parallel(n_jobs=num_files, verbose=10)(
                delayed(self.create_dataframe_per_file)(h5_path) for h5_path in h5_to_read
            )

            # Save the dataframes to Parquet files
            self.save_parquet(dataframes, parquets_to_read)

    def get_filled_dataframe(self):
        # Read all parquet files into one dataframe using dask and reads the metadata and schema
        dataframe = ddf.read_parquet(self.parquet_paths, calculate_divisions=True)
        metadata = [pq.read_metadata(file) for file in self.parquet_paths]
        # schema = [pq.read_schema(file) for file in self.parquet_paths]

        # Channels to fill NaN values
        channels: list[str] = self.get_channels(["per_pulse", "per_train"])

        overlap = min(file.num_rows for file in metadata)

        print("Filling nan values...")
        dataframe = forward_fill_lazy(
            df=dataframe,
            columns=channels,
            before=overlap,
            iterations=self._config.get("forward_fill_iterations", 2),
        )
        # Remove the NaNs from per_electron channels
        dataframe_electron = dataframe.dropna(
            subset=self.get_channels(["per_electron"]),
        )
        dataframe_pulse = dataframe[
            self.multi_index + self.get_channels(["per_pulse", "per_train"])
        ]
        dataframe_pulse = dataframe_pulse[
            (dataframe_pulse["electronId"] == 0) | (np.isnan(dataframe_pulse["electronId"]))
        ]
        return dataframe_electron, dataframe_pulse
