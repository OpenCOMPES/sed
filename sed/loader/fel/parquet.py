from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import dask.dataframe as ddf
from pandas import DataFrame


DFType = TypeVar("DFType", DataFrame, ddf.DataFrame)


class ParquetHandler:
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

    def __init__(
        self,
        parquet_names: str | list[str],
        folder: Path,
        subfolder: str = "",
        prefix: str = "",
        suffix: str = "",
        parquet_paths: Path | list[Path] = None,
    ):
        if parquet_paths is not None:
            self.parquet_paths = parquet_paths
        else:
            # Create the full path for the Parquet file
            parquet_dir = folder.joinpath(subfolder)
            parquet_dir.mkdir(parents=True, exist_ok=True)
            filenames = [
                Path(prefix + parquet_name + suffix).with_suffix(".parquet")
                for parquet_name in parquet_names
            ]
            self.parquet_paths = [parquet_dir.joinpath(filename) for filename in filenames]

    def save_parquet(self, dfs: DFType | list[DFType]) -> None:
        """
        Save the DataFrame to a Parquet file.

        Args:
            df (DFType | List[DFType]): The Dask DataFrame to be saved.
        """
        # Compute the Dask DataFrame, reset the index, and save to Parquet
        for df, parquet_paths in zip(dfs, self.parquet_paths):
            df.compute().reset_index(drop=True).to_parquet(parquet_paths)
            print(f"Parquet file saved: {parquet_paths}")

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

    def delete_parquet(self) -> None:
        """
        Delete the Parquet file.
        """
        for parquet_path in self.parquet_paths:
            parquet_path.unlink()
            print(f"Parquet file deleted: {parquet_path}")
