"""
The ParquetHandler class allows for saving and reading Dask DataFrames to/from Parquet files.
It also provides methods for initializing paths, saving Parquet files and also reading them
into a Dask DataFrame.

Typical usage example:

    parquet_handler = ParquetHandler(parquet_names='data', folder=Path('/path/to/folder'))
    parquet_handler.save_parquet(df) # df is a uncomputed Dask DataFrame
    data = parquet_handler.read_parquet()
"""
from __future__ import annotations

from pathlib import Path

import dask.dataframe as ddf


class ParquetHandler:
    """A class for handling the creation and manipulation of Parquet files."""

    def __init__(
        self,
        parquet_names: str | list[str] = None,
        folder: Path = None,
        subfolder: str = "",
        prefix: str = "",
        suffix: str = "",
        extension: str = "parquet",
        parquet_paths: Path = None,
    ):
        """
        A handler for saving and reading Dask DataFrames to/from Parquet files.

        Args:
            parquet_names Union[str, List[str]]: The base name of the Parquet files.
            folder (Path): The directory where the Parquet file will be stored.
            subfolder (str): Optional subfolder within the main folder.
            prefix (str): Optional prefix for the Parquet file name.
            suffix (str): Optional suffix for the Parquet file name.
            parquet_path (Path): Optional custom path for the Parquet file.
        """

        self.parquet_paths: list[Path] = None

        if isinstance(parquet_names, str):
            parquet_names = [parquet_names]

        if not folder and not parquet_paths:
            raise ValueError("Please provide folder or parquet_paths.")
        if folder and not parquet_names:
            raise ValueError("With folder, please provide parquet_names.")

        # If parquet_paths is provided, use it and ignore the other arguments
        # Else, initialize the paths
        if parquet_paths:
            self.parquet_paths = (
                parquet_paths if isinstance(parquet_paths, list) else [parquet_paths]
            )
        else:
            self._initialize_paths(parquet_names, folder, subfolder, prefix, suffix, extension)

    def _initialize_paths(
        self,
        parquet_names: list[str],
        folder: Path,
        subfolder: str = None,
        prefix: str = None,
        suffix: str = None,
        extension: str = None,
    ) -> None:
        """
        Create the directory for the Parquet file.
        """
        # Create the full path for the Parquet file
        parquet_dir = folder.joinpath(subfolder)
        parquet_dir.mkdir(parents=True, exist_ok=True)

        if extension:
            extension = f".{extension}"  # to be backwards compatible
        self.parquet_paths = [
            parquet_dir.joinpath(Path(f"{prefix}{name}{suffix}{extension}"))
            for name in parquet_names
        ]

    def save_parquet(
        self,
        dfs: list[ddf.DataFrame],
        drop_index: bool = False,
    ) -> None:
        """
        Save the DataFrame to a Parquet file.

        Args:
            dfs (DataFrame | ddf.DataFrame): The pandas or Dask Dataframe to be saved.
            drop_index (bool): If True, drops the index before saving.
        """
        # Compute the Dask DataFrame, reset the index, and save to Parquet
        dfs = dfs if isinstance(dfs, list) else [dfs]
        for df, parquet_path in zip(dfs, self.parquet_paths):
            df.compute().reset_index(drop=drop_index).to_parquet(parquet_path)

    def read_parquet(self) -> ddf.DataFrame:
        """
        Read a Dask DataFrame from the Parquet file.

        Returns:
            ddf.DataFrame: The Dask DataFrame read from the Parquet file.

        Raises:
            FileNotFoundError: If the Parquet file does not exist.
        """
        try:
            return ddf.read_parquet(self.parquet_paths, calculate_divisions=True)
        except Exception as exc:
            raise FileNotFoundError(
                "The Parquet file does not exist. "
                "If it is in another location, provide the correct path as parquet_path.",
            ) from exc
