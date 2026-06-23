"""
The module provides a MetadataRetriever class for retrieving metadata
from a Scicat Instance based on beamtime and run IDs.
"""
from __future__ import annotations

import requests
import json, yaml
from pathlib import Path

from sed.core.config import read_env_var
from sed.core.config import save_env_var
from sed.core.logging import setup_logging

logger = setup_logging("flash_metadata_retriever")


class MetadataRetriever:
    """
    Retrieves metadata from SciCat or local YAML files for a given beamtime and runs.
    """

    def __init__(self, metadata_config: dict, token: str = None) -> None:
        """
        Initializes the MetadataRetriever class.

        Args:
            metadata_config (dict): Takes a dict containing at least url for the scicat instance.
            token (str, optional): The token to use for fetching metadata. If provided,
                will be saved to .env file for future use.
        """
        # Token handling
        if token:
            self.token = token
            save_env_var("SCICAT_TOKEN", self.token)
        else:
            # Try to load token from config or .env file
            self.token = read_env_var("SCICAT_TOKEN")

        if not self.token:
            raise ValueError(
                "Token is required for metadata collection. Either provide a token "
                "parameter or set the SCICAT_TOKEN environment variable.",
            )

        self.url = metadata_config.get("archiver_url")
        if not self.url:
            raise ValueError("No URL provided for fetching metadata from scicat.")

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def get_metadata(
        self,
        beamtime_id: str,
        runs: list,
        metadata: dict | None = None,
    ) -> dict:
        """
        Retrieves metadata for a beamtime and runs from SciCat.
        Returns a dict with 'scientificMetadata' keyed by run ID.

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

        all_runs_metadata: dict[str, dict] = {}

        for run in runs:
            pid = f"{beamtime_id}/{run}"
            # logger.debug(f"Retrieving metadata for PID: {pid}")
            metadata_run = self._get_metadata_per_run(pid)
            # metadata.update(metadata_run)  # TODO: Not correct for multiple runs
            # Use 'scientificMetadata' if available, otherwise entire dict
            all_runs_metadata[run] = metadata_run.get("scientificMetadata", metadata_run)

        metadata["scientificMetadata"] = all_runs_metadata
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
            # logger.debug(f"Attempting to fetch metadata with new URL format for PID: {pid}")
            # dataset_response = requests.get(
            #     self._create_new_dataset_url(pid),
            #     headers=headers2,
            #     timeout=10,
            # )
            # dataset_response.raise_for_status()
            logger.debug(f"Fetching metadata (new URL) for PID: {pid}")
            response = requests.get(self._create_new_dataset_url(pid), headers=headers2, timeout=10)
            response.raise_for_status()

            # Check if response is an empty object because wrong url for older implementation
            # if not dataset_response.content:
            #     logger.debug("Empty response, trying old URL format")
            #     dataset_response = requests.get(
            #         self._create_old_dataset_url(pid),
            #         headers=headers2,
            #         timeout=10,
            #     )
            if not response.content:
                logger.debug("Empty response, trying old URL format")
                response = requests.get(self._create_old_dataset_url(pid), headers=headers2, timeout=10)
            # If the dataset request is successful, return the retrieved metadata
            # as a JSON object
            # return dataset_response.json()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to retrieve metadata for PID {pid}: {e}")
            return {}

        # except requests.exceptions.RequestException as exception:
        #     logger.warning(f"Failed to retrieve metadata for PID {pid}: {str(exception)}")
        #     return {}  # Return an empty dictionary for this run

    def _create_old_dataset_url(self, pid: str) -> str:
        return f"{self.url}datasets/%2F{self._reformat_pid(pid)}"

    def _create_new_dataset_url(self, pid: str) -> str:
        return f"{self.url}datasets/{self._reformat_pid(pid)}"

    def _reformat_pid(self, pid: str) -> str:
        """SciCat adds a pid-prefix + "/"  but at DESY prefix = "" """
        """Replace '/' with '%2F' for SciCat PID."""
        return pid.replace("/", "%2F")

    # ----------------------------
    # Local metadata
    # ----------------------------
    def get_local_metadata(
        self,
        beamtime_id: str,
        beamtime_dir: str | Path,
        meta_dir: str | Path,
        runs: list,
        metadata: dict | None = None,
    ) -> dict:
        """
        Retrieves metadata for a beamtime and runs from local YAML files.
        Returns a dict with 'scientificMetadata' keyed by run ID.

        Args:
            beamtime_dir (str)|Path: Beamtime directory.
            meta_dir (str)|Path: Local metadata directory..
            runs (list): A list of run IDs.
            metadata (dict, optional): The existing metadata dictionary.
            Defaults to None.

        Returns:
            Dict: The updated metadata dictionary.

        Raises:
            Exception: If the request to retrieve metadata fails.
        """
        if metadata is None:
            metadata = {}

        beamtime_metadata = self._get_beamtime_metadata(beamtime_dir, beamtime_id)
        metadata.update(beamtime_metadata)
        
        all_runs_metadata: dict[str, dict] = {}

        for run in runs:
            logger.debug(f"Retrieving local metadata for run: {run}")
            run_metadata = self._get_local_metadata_per_run(meta_dir, run)
            all_runs_metadata[run] = run_metadata.get("_data", {})

        metadata["scientificMetadata"] = all_runs_metadata

        logger.debug(f"Retrieved metadata with {len(metadata)} entries")
        return metadata

    def _get_beamtime_metadata(
        self,
        beamtime_dir: str | Path,
        beamtime_id: str,
    ) -> dict:
        """
        Retrieves general metadata for a given beamtime ID from beamtime-metadata-{beamtime_id}.json file

        Args:
            beamtime_dir (str)|Path: Beamtime directory.
            beamtime_id (str): The ID of the beamtime.

        Returns:
            Dict: The retrieved metadata dictionary.

        Raises:
            Exception: If the request to retrieve metadata fails.
        """     
        try:
            beamtime_dir = Path(beamtime_dir)
            filepath = beamtime_dir / f"beamtime-metadata-{beamtime_id}.json"
            with filepath.open("r") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to retrieve metadata for beamtime {beamtime_id}: {exc}")
            return {}


    def _get_local_metadata_per_run(self, meta_dir: str | Path, run: str) -> dict:
        """
        Retrieves metadata for a specific run from the latest YAML file:
        {run}_N.yaml (highest N) or fallback to {run}.yaml

            Args:
                meta_dir(str): The existing local metadata folder.

            Returns:
                dict: The retrieved metadata.

            Raises:
                Exception: If the request to retrieve metadata fails.
        """
        try:
            # run = str(run)
            meta_dir = Path(meta_dir)
    
            # candidates = []
            run = str(run)
            candidates: list[tuple[int, Path]] = []

            # Look for versioned YAML files
            for path in meta_dir.glob(f"{run}_*.yaml"):
                try:
                    version = int(path.stem.split("_")[-1])
                    candidates.append((version, path))
                except ValueError:
                    continue

            # Fallback: unversioned single file
            if not candidates:
                single_file = meta_dir / f"{run}.yaml"
                if single_file.exists():
                    candidates.append((0, single_file))
                    
            if not candidates:
                raise FileNotFoundError(f"No metadata files found for run {run} in {meta_dir}")

            # Pick the latest version
            _, latest_path = max(candidates, key=lambda x: x[0])
    
            logger.info(f"Loading local metadata from {latest_path.name}")
    
            run_metadata = yaml.safe_load(latest_path.read_text())
            return run_metadata or {"_data": {}}
    
        except Exception as exc:
            logger.warning(f"Failed to retrieve local metadata for run {run}: {exc}")
            return {"_data": {}}
