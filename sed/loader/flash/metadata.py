"""
The module provides a MetadataRetriever class for retrieving metadata
from a Scicat Instance based on beamtime and run IDs.
"""

import warnings
from typing import Dict
from typing import Optional

import requests


class MetadataRetriever:
    """
    A class for retrieving metadata from a Scicat instance based
    on beamtime and run IDs.
    """

    def __init__(self, metadata_config: Dict, scicat_token: str = None) -> None:
        """
        Initializes the MetadataRetriever class.

        Args:
            metadata_config (dict): Takes a dict containing
            at least url, and optionally token for the scicat instance.
            scicat_token (str, optional): The token to use for fetching metadata.
        """
        self.token = metadata_config.get("scicat_token", None)
        if scicat_token:
            self.token = scicat_token
        self.url = metadata_config.get("scicat_url", None)

        if not self.token or not self.url:
            raise ValueError("No URL or token provided for fetching metadata from scicat.")

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.token = metadata_config["scicat_token"]

    def get_metadata(
        self,
        beamtime_id: str,
        runs: list,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Retrieves metadata for a given beamtime ID and list of runs.

        Args:
            beamtime_id (str): The ID of the beamtime.
            runs (list): A list of run IDs.
            metadata (Dict, optional): The existing metadata dictionary.
            Defaults to None.

        Returns:
            Dict: The updated metadata dictionary.

        Raises:
            Exception: If the request to retrieve metadata fails.
        """
        # If metadata is not provided, initialize it as an empty dictionary
        if metadata is None:
            metadata = {}

        # Iterate over the list of runs
        for run in runs:
            pid = f"{beamtime_id}/{run}"
            # Retrieve metadata for each run and update the overall metadata dictionary
            metadata_run = self._get_metadata_per_run(pid)
            metadata.update(
                metadata_run,
            )  # TODO: Not correct for multiple runs

        return metadata

    def _get_metadata_per_run(self, pid: str) -> Dict:
        """
        Retrieves metadata for a specific run based on the PID.

        Args:
            pid (str): The PID of the run.

        Returns:
            Dict: The retrieved metadata.

        Raises:
            Exception: If the request to retrieve metadata fails.
        """
        headers2 = dict(self.headers)
        headers2["Authorization"] = "Bearer {}".format(self.token)

        try:
            dataset_response = requests.get(
                self._create_new_dataset_url(pid),
                headers=headers2,
                timeout=10,
            )
            dataset_response.raise_for_status()
            # Check if response is an empty object because wrong url for older implementation
            if not dataset_response.content:
                dataset_response = requests.get(
                    self._create_old_dataset_url(pid), headers=headers2, timeout=10
                )
            # If the dataset request is successful, return the retrieved metadata
            # as a JSON object
            return dataset_response.json()
        except requests.exceptions.RequestException as exception:
            # If the request fails, raise warning
            print(warnings.warn(f"Failed to retrieve metadata for PID {pid}: {str(exception)}"))
            return {}  # Return an empty dictionary for this run

    def _create_old_dataset_url(self, pid: str) -> str:
        return "{burl}/{url}/%2F{npid}".format(
            burl=self.url, url="Datasets", npid=self._reformat_pid(pid)
        )

    def _create_new_dataset_url(self, pid: str) -> str:
        return "{burl}/{url}/{npid}".format(
            burl=self.url, url="Datasets", npid=self._reformat_pid(pid)
        )

    def _reformat_pid(self, pid: str) -> str:
        """SciCat adds a pid-prefix + "/"  but at DESY prefix = "" """
        return (pid).replace("/", "%2F")
