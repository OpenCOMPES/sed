"""
The module provides a MetadataRetriever class for retrieving metadata
from a Scicat Instance based on beamtime and run IDs.
"""
from __future__ import annotations

import requests
from sed.core.logging import setup_logging

logger = setup_logging("flash_metadata_retriever")


class MetadataRetriever:
    """
    A class for retrieving metadata from a Scicat instance based
    on beamtime and run IDs.
    """

    def __init__(self, metadata_config: dict, token: str = None) -> None:
        """
        Initializes the MetadataRetriever class.

        Args:
            metadata_config (dict): Takes a dict containing
            at least url, and optionally token for the scicat instance.
            token (str, optional): The token to use for fetching metadata.
        """
        self.token = metadata_config.get("token", None)
        if token:
            self.token = token
        self.url = metadata_config.get("archiver_url", None)

        if not self.token or not self.url:
            raise ValueError("No URL or token provided for fetching metadata from scicat.")

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.token = metadata_config["token"]

    def get_metadata(
        self,
        beamtime_id: str,
        runs: list,
        metadata: dict = None,
    ) -> dict:
        """
        Retrieves metadata for a given beamtime ID and list of runs.

        Args:
            beamtime_id (str): The ID of the beamtime.
            runs (list): A list of run IDs.
            metadata (dict, optional): The existing metadata dictionary.
            Defaults to None.

        Returns:
            Dict: The updated metadata dictionary.

        Raises:
            Exception: If the request to retrieve metadata fails.
        """
        logger.debug(f"Fetching metadata for beamtime {beamtime_id}, runs: {runs}")

        if metadata is None:
            metadata = {}

        for run in runs:
            pid = f"{beamtime_id}/{run}"
            logger.debug(f"Retrieving metadata for PID: {pid}")
            metadata_run = self._get_metadata_per_run(pid)
            metadata.update(metadata_run)

        logger.debug(f"Retrieved metadata with {len(metadata)} entries")
        return metadata

    def _get_metadata_per_run(self, pid: str) -> dict:
        """
        Retrieves metadata for a specific run based on the PID.

        Args:
            pid (str): The PID of the run.

        Returns:
            dict: The retrieved metadata.

        Raises:
            Exception: If the request to retrieve metadata fails.
        """
        headers2 = dict(self.headers)
        headers2["Authorization"] = f"Bearer {self.token}"

        try:
            logger.debug(f"Attempting to fetch metadata with new URL format for PID: {pid}")
            dataset_response = requests.get(
                self._create_new_dataset_url(pid),
                headers=headers2,
                timeout=10,
            )
            dataset_response.raise_for_status()

            if not dataset_response.content:
                logger.debug("Empty response, trying old URL format")
                dataset_response = requests.get(
                    self._create_old_dataset_url(pid),
                    headers=headers2,
                    timeout=10,
                )
            return dataset_response.json()

        except requests.exceptions.RequestException as exception:
            logger.warning(f"Failed to retrieve metadata for PID {pid}: {str(exception)}")
            return {}

    def _create_old_dataset_url(self, pid: str) -> str:
        return "{burl}/{url}/%2F{npid}".format(
            burl=self.url,
            url="Datasets",
            npid=self._reformat_pid(pid),
        )

    def _create_new_dataset_url(self, pid: str) -> str:
        return "{burl}/{url}/{npid}".format(
            burl=self.url,
            url="Datasets",
            npid=self._reformat_pid(pid),
        )

    def _reformat_pid(self, pid: str) -> str:
        """SciCat adds a pid-prefix + "/"  but at DESY prefix = "" """
        return (pid).replace("/", "%2F")
