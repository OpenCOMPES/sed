"""
The module provides a MetadataRetriever class for retrieving metadata
from a Scicatinstance based on beamtime and run IDs.
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
        headers2["Authorization"] = f"Bearer {self.token}"
        try:
            # Create the dataset URL using the PID
            dataset_response = requests.get(
                self._create_dataset_url_by_PID(pid),
                params={"access_token": self.token},
                headers=headers2,
                timeout=10,
            )
            dataset_response.raise_for_status()  # Raise HTTPError if request fails
            # If the dataset request is successful, return the retrieved metadata
            # as a JSON object
            return dataset_response.json()
        except requests.exceptions.RequestException as exception:
            # If the request fails, raise warning
            warnings.warn(f"Failed to retrieve metadata for PID {pid}: {str(exception)}")
            return {}  # Return an empty dictionary for this run

    def _create_dataset_url_by_PID(self, pid: str) -> str:  # pylint: disable=invalid-name
        """
        Creates the dataset URL based on the PID.

        Args:
            pid (str): The PID of the run.

        Returns:
            str: The dataset URL.

        Raises:
            Exception: If the token request fails.
        """
        npid = pid.replace(
            "/",
            "%2F",
        )  # Replace slashes in the PID with URL-encoded slashes
        url = f"{self.url}/Datasets/{npid}"
        return url
