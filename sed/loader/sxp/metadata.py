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

    def __init__(self, metadata_config: Dict) -> None:
        """
        Initializes the MetadataRetriever class.

        Args:
            metadata_config (dict): Takes a dict containing
            at least url, username and password
        """
        self.url = metadata_config["scicat_url"]
        self.username = metadata_config["scicat_username"]
        self.password = metadata_config["scicat_password"]
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
        try:
            # Create the dataset URL using the PID
            dataset_response = requests.get(self._create_dataset_url_by_PID(pid), timeout=10)
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
        npid = ("/" + pid).replace(
            "/",
            "%2F",
        )  # Replace slashes in the PID with URL-encoded slashes
        url = f"{self.url}/RawDatasets/{npid}?access_token={self._get_token()}"
        return url

    def _get_token(self) -> str:
        """
        Retrieves the access token for authentication.

        Returns:
            str: The access token.

        Raises:
            Exception: If the token request fails.
        """
        try:
            token_url = f"{self.url}/Users/login"
            # Send a POST request to the token URL with the username and password
            token_response = requests.post(
                token_url,
                headers=self.headers,
                json={"username": self.username, "password": self.password},
                timeout=10,
            )
            token_response.raise_for_status()
            # If the token request is successful, return the access token from the response
            return token_response.json()["id"]

            # Otherwise issue warning
        except requests.exceptions.RequestException as exception:
            warnings.warn(f"Failed to retrieve authentication token: {str(exception)}")
            return ""  # Return an empty string if token retrieval fails
