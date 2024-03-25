"""
module sed.loader.mpes, code for loading hdf5 files delayed into a dask dataframe.
Mostly ported from https://github.com/mpes-kit/mpes.
@author: L. Rettig
"""
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import dask.dataframe as ddf
import numpy as np

from sed.loader.base.loader import BaseLoader


class GenericLoader(BaseLoader):
    """Dask implementation of the Loader. Reads from various file types using the
    utilities of Dask.

    Args:
        config (dict, optional): Config dictionary. Defaults to None.
        meta_handler (MetaHandler, optional): MetaHandler object. Defaults to None.
    """

    __name__ = "generic"

    supported_file_types = ["parquet", "csv", "json"]

    def read_dataframe(
        self,
        files: Union[str, Sequence[str]] = None,
        folders: Union[str, Sequence[str]] = None,
        runs: Union[str, Sequence[str]] = None,
        ftype: str = "parquet",
        metadata: dict = None,
        collect_metadata: bool = False,
        **kwds,
    ) -> Tuple[ddf.DataFrame, ddf.DataFrame, dict]:
        """Read stored files from a folder into a dataframe.

        Args:
            files (Union[str, Sequence[str]], optional): File path(s) to process.
                Defaults to None.
            folders (Union[str, Sequence[str]], optional): Path to folder(s) where files
                are stored. Path has priority such that if it's specified, the specified
                files will be ignored. Defaults to None.
            runs (Union[str, Sequence[str]], optional): Run identifier(s). Corresponding
                files will be located in the location provided by ``folders``. Takes
                precendence over ``files`` and ``folders``. Defaults to None.
            ftype (str, optional): File type to read ('parquet', 'json', 'csv', etc).
                If a folder path is given, all files with the specified extension are
                read into the dataframe in the reading order. Defaults to "parquet".
            metadata (dict, optional): Manual meta data dictionary. Auto-generated
                meta data are added to it. Defaults to None.
            collect_metadata (bool): Option to collect metadata from files. Requires
                a valid config dict. Defaults to False.
            **kwds: keyword arguments. See the keyword arguments for the specific file
                parser in``dask.dataframe`` module.

        Raises:
            ValueError: Raised if neither files nor folder provided.
            FileNotFoundError: Raised if the fileds or folder cannot be found.
            ValueError: Raised if the file type is not supported.

        Returns:
            Tuple[ddf.DataFrame, dict]: Dask dataframe, timed dataframe and metadata
            read from specified files.
        """
        # pylint: disable=duplicate-code
        super().read_dataframe(
            files=files,
            folders=folders,
            runs=runs,
            ftype=ftype,
            metadata=metadata,
        )

        if not self.files:
            raise FileNotFoundError("No valid files found!")

        if collect_metadata:
            # TODO implementation
            self.metadata = self.metadata

        if ftype == "parquet":
            return (ddf.read_parquet(self.files, **kwds), None, self.metadata)

        if ftype == "json":
            return (ddf.read_json(self.files, **kwds), None, self.metadata)

        if ftype == "csv":
            return (ddf.read_csv(self.files, **kwds), None, self.metadata)

        try:
            return (ddf.read_table(self.files, **kwds), None, self.metadata)
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise ValueError(
                "The file format cannot be understood!",
            ) from exc

    def get_files_from_run_id(
        self,
        run_id: str,  # noqa: ARG002
        folders: Union[str, Sequence[str]] = None,  # noqa: ARG002
        extension: str = None,  # noqa: ARG002
        **kwds,  # noqa: ARG002
    ) -> List[str]:
        """Locate the files for a given run identifier.

        Args:
            run_id (str): The run identifier to locate.
            folders (Union[str, Sequence[str]], optional): The directory(ies) where the raw
                data is located. Defaults to None.
            extension (str, optional): The file extension. Defaults to "h5".
            kwds: Keyword arguments

        Return:
            str: Path to the location of run data.
        """
        raise NotImplementedError

    def get_count_rate(
        self,
        fids: Sequence[int] = None,  # noqa: ARG002
        **kwds,  # noqa: ARG002
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create count rate data for the files specified in ``fids``.

        Args:
            fids (Sequence[int], optional): fids (Sequence[int]): the file ids to
                include. Defaults to list of all file ids.
            kwds: Keyword arguments

        Return:
            Tuple[np.ndarray, np.ndarray]: Arrays containing countrate and seconds
            into the scan.
        """
        # TODO
        return None, None

    def get_elapsed_time(
        self,
        fids: Sequence[int] = None,  # noqa: ARG002
        **kwds,  # noqa: ARG002
    ) -> float:
        """Return the elapsed time in the files specified in ``fids``.

        Args:
            fids (Sequence[int], optional): fids (Sequence[int]): the file ids to
                include. Defaults to list of all file ids.
            kwds: Keyword arguments

        Returns:
            float: The elapsed time in the files in seconds.
        """
        return None


LOADER = GenericLoader
