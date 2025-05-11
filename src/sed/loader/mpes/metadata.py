"""
The module provides a MetadataRetriever class for retrieving metadata
from an EPICS archiver and an elabFTW instance.
"""
from __future__ import annotations

import datetime
import json
from copy import deepcopy
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import urlopen

import elabapi_python
import numpy as np

from sed.core.config import read_env_var
from sed.core.config import save_env_var
from sed.core.logging import setup_logging

logger = setup_logging("mpes_metadata_retriever")


class MetadataRetriever:
    """
    A class for retrieving metadata from an EPICS archiver and an elabFTW instance.
    """

    def __init__(self, metadata_config: dict, token: str = None) -> None:
        """
        Initializes the MetadataRetriever class.

        Args:
            metadata_config (dict): Takes a dict containing at least url for the EPICS archiver and
                elabFTW instance.
            token (str, optional): The token to use for fetching metadata. If provided,
                will be saved to .env file for future use.
        """
        self._config = deepcopy(metadata_config)
        # Token handling
        if token:
            self.token = token
            save_env_var("ELAB_TOKEN", self.token)
        else:
            # Try to load token from config or .env file
            self.token = read_env_var("ELAB_TOKEN")

        if not self.token:
            logger.warning(
                "No valid token provided for elabFTW. Fetching elabFTW metadata will be skipped.",
            )
            return

        self.url = self._config.get("elab_url")
        if not self.url:
            logger.warning(
                "No URL provided for elabFTW. Fetching elabFTW metadata will be skipped.",
            )
            return

        # Config
        self.configuration = elabapi_python.Configuration()
        self.configuration.api_key["api_key"] = self.token
        self.configuration.api_key_prefix["api_key"] = "Authorization"
        self.configuration.host = str(self.url)
        self.configuration.debug = False
        self.configuration.verify_ssl = False

        # create an instance of the API class
        self.api_client = elabapi_python.ApiClient(self.configuration)
        # fix issue with Authorization header not being properly set by the generated lib
        self.api_client.set_default_header(header_name="Authorization", header_value=self.token)

        # create an instance of Items
        self.itemsApi = elabapi_python.ItemsApi(self.api_client)
        self.experimentsApi = elabapi_python.ExperimentsApi(self.api_client)
        self.linksApi = elabapi_python.LinksToItemsApi(self.api_client)
        self.experimentsLinksApi = elabapi_python.LinksToExperimentsApi(self.api_client)
        self.usersApi = elabapi_python.UsersApi(self.api_client)

    def fetch_epics_metadata(self, ts_from: float, ts_to: float, metadata: dict) -> dict:
        """Fetch metadata from an EPICS archiver instance for times between ts_from and ts_to.
        Channels are defined in the config.

        Args:
            ts_from (float): Start timestamp of the range to collect data from.
            ts_to (float): End timestamp of the range to collect data from.
            metadata (dict): Input metadata dictionary. Will be updated

        Returns:
            dict: Updated metadata dictionary.
        """
        if not self._config.get("archiver_url"):
            logger.warning(
                "No URL provided for fetching metadata from the EPICS archiver. "
                "Fetching EPICS metadata will be skipped.",
            )
            return metadata

        logger.info("Collecting data from the EPICS archive...")

        start = datetime.datetime.utcfromtimestamp(ts_from)

        # Get metadata from Epics archive if not present already
        epics_channels = self._config["epics_pvs"]

        channels_missing = set(epics_channels) - set(
            metadata["file"].keys(),
        )
        for channel in channels_missing:
            try:
                _, vals = get_archiver_data(
                    archiver_url=str(self._config.get("archiver_url")),
                    archiver_channel=channel,
                    ts_from=ts_from,
                    ts_to=ts_to,
                )
                metadata["file"][f"{channel}"] = np.mean(vals)

            except IndexError:
                logger.info(
                    f"Data for channel {channel} doesn't exist for time {start}",
                )
            except HTTPError as exc:
                logger.warning(
                    f"Incorrect URL for the archive channel {channel}. "
                    "Make sure that the channel name and file start and end times are "
                    "correct.",
                )
                logger.warning(f"Error code: {exc}")
            except URLError as exc:
                logger.warning(
                    f"Cannot access the archive URL for channel {channel}. "
                    f"Make sure that you are within the FHI network."
                    f"Skipping over channels {channels_missing}.",
                )
                logger.warning(f"Error code: {exc}")
                break

        # Determine the correct aperture_config
        stamps = sorted(
            list(self._config["aperture_config"].keys()) + [start],
        )
        current_index = stamps.index(start)
        timestamp = stamps[current_index - 1]  # pick last configuration before file date

        # Aperture metadata
        if "instrument" not in metadata.keys():
            metadata["instrument"] = {"analyzer": {}}
        metadata["instrument"]["analyzer"]["fa_shape"] = "circle"
        metadata["instrument"]["analyzer"]["ca_shape"] = "circle"
        metadata["instrument"]["analyzer"]["fa_size"] = np.nan
        metadata["instrument"]["analyzer"]["ca_size"] = np.nan
        # get field aperture shape and size
        if {
            self._config["fa_in_channel"],
            self._config["fa_hor_channel"],
        }.issubset(set(metadata["file"].keys())):
            fa_in = metadata["file"][self._config["fa_in_channel"]]
            fa_hor = metadata["file"][self._config["fa_hor_channel"]]
            for key, value in self._config["aperture_config"][timestamp]["fa_size"].items():
                if value[0][0] < fa_in < value[0][1] and value[1][0] < fa_hor < value[1][1]:
                    try:
                        metadata["instrument"]["analyzer"]["fa_size"] = float(key)
                    except ValueError:  # store string if numeric interpretation fails
                        metadata["instrument"]["analyzer"]["fa_shape"] = key
                    break
            else:
                logger.warning("Field aperture size not found.")

        # get contrast aperture shape and size
        if self._config["ca_in_channel"] in metadata["file"]:
            ca_in = metadata["file"][self._config["ca_in_channel"]]
            for key, value in self._config["aperture_config"][timestamp]["ca_size"].items():
                if value[0] < ca_in < value[1]:
                    try:
                        metadata["instrument"]["analyzer"]["ca_size"] = float(key)
                    except ValueError:  # store string if numeric interpretation fails
                        metadata["instrument"]["analyzer"]["ca_shape"] = key
                    break
            else:
                logger.warning("Contrast aperture size not found.")

        # Storing the lens modes corresponding to lens voltages.
        # Use lens voltages present in first lens_mode entry.
        lens_list = self._config["lens_mode_config"][
            next(iter(self._config["lens_mode_config"]))
        ].keys()

        lens_volts = np.array(
            [metadata["file"].get(f"KTOF:Lens:{lens}:V", np.nan) for lens in lens_list],
        )
        for mode, value in self._config["lens_mode_config"].items():
            lens_volts_config = np.array([value[k] for k in lens_list])
            if np.allclose(
                lens_volts,
                lens_volts_config,
                rtol=0.005,
            ):  # Equal upto 0.5% tolerance
                metadata["instrument"]["analyzer"]["lens_mode"] = mode
                break
        else:
            logger.warning(
                "Lens mode for given lens voltages not found. "
                "Storing lens mode from the user, if provided.",
            )

        # Determining projection from the lens mode
        try:
            lens_mode = metadata["instrument"]["analyzer"]["lens_mode"]
            if "spatial" in lens_mode.split("_")[1]:
                metadata["instrument"]["analyzer"]["projection"] = "real"
                metadata["instrument"]["analyzer"]["scheme"] = "spatial dispersive"
            else:
                metadata["instrument"]["analyzer"]["projection"] = "reciprocal"
                metadata["instrument"]["analyzer"]["scheme"] = "momentum dispersive"
        except IndexError:
            logger.warning(
                "Lens mode must have the form, '6kV_kmodem4.0_20VTOF_v3.sav'. "
                "Can't determine projection. "
                "Storing projection from the user, if provided.",
            )
        except KeyError:
            logger.warning(
                "Lens mode not found. Can't determine projection. "
                "Storing projection from the user, if provided.",
            )

        return metadata

    def fetch_elab_metadata(self, runs: list[str], metadata: dict) -> dict:
        """Fetch metadata from an elabFTW instance

        Args:
            runs (list[str]): List of runs for which to fetch metadata
            metadata (dict): Input metadata dictionary. Will be updated

        Returns:
            dict: Updated metadata dictionary
        """
        if not self.token:
            logger.warning(
                "No valid token found. Token is required for metadata collection. Either provide "
                "a token parameter or set the ELAB_TOKEN environment variable.",
            )
            return metadata

        if not self.url:
            logger.warning(
                "No URL provided for fetching metadata from elabFTW. "
                "Fetching elabFTW metadata will be skipped.",
            )
            return metadata

        logger.info("Collecting data from the elabFTW instance...")
        # Get the experiment
        try:
            experiment = self.experimentsApi.read_experiments(q=f"'Metis scan {runs[0]}'")[0]
        except IndexError:
            logger.warning(f"No elabFTW entry found for run {runs[0]}")
            return metadata

        if "elabFTW" not in metadata:
            metadata["elabFTW"] = {}

        exp_id = experiment.id
        # Get user information
        user = self.usersApi.read_user(experiment.userid)
        metadata["elabFTW"]["user"] = {}
        metadata["elabFTW"]["user"]["name"] = user.fullname
        metadata["elabFTW"]["user"]["email"] = user.email
        metadata["elabFTW"]["user"]["id"] = user.userid
        if user.orcid:
            metadata["elabFTW"]["user"]["orcid"] = user.orcid
        # Get the links to items
        links = self.linksApi.read_entity_items_links(entity_type="experiments", id=exp_id)
        # Get the items
        items = [self.itemsApi.get_item(link.entityid) for link in links]
        items_dict = {item.category_title: item for item in items}
        items_dict["scan"] = experiment

        # Sort the metadata
        for category, item in items_dict.items():
            category = category.replace(":", "").replace(" ", "_").lower()
            if category not in metadata["elabFTW"]:
                metadata["elabFTW"][category] = {}
            metadata["elabFTW"][category]["title"] = item.title
            metadata["elabFTW"][category]["summary"] = item.body
            metadata["elabFTW"][category]["id"] = item.id
            metadata["elabFTW"][category]["elabid"] = item.elabid
            if item.sharelink:
                metadata["elabFTW"][category]["link"] = item.sharelink
            if item.metadata is not None:
                metadata_json = json.loads(item.metadata)
                for key, val in metadata_json["extra_fields"].items():
                    if val["value"] is not None and val["value"] != "" and val["value"] != ["None"]:
                        try:
                            metadata["elabFTW"][category][key] = float(val["value"])
                        except ValueError:
                            metadata["elabFTW"][category][key] = val["value"]

        # group beam profiles:
        if (
            "laser_status" in metadata["elabFTW"]
            and "pump_profile_x" in metadata["elabFTW"]["laser_status"]
            and "pump_profile_y" in metadata["elabFTW"]["laser_status"]
        ):
            metadata["elabFTW"]["laser_status"]["pump_profile"] = [
                float(metadata["elabFTW"]["laser_status"]["pump_profile_x"]),
                float(metadata["elabFTW"]["laser_status"]["pump_profile_y"]),
            ]
        if (
            "laser_status" in metadata["elabFTW"]
            and "probe_profile_x" in metadata["elabFTW"]["laser_status"]
            and "probe_profile_y" in metadata["elabFTW"]["laser_status"]
        ):
            metadata["elabFTW"]["laser_status"]["probe_profile"] = [
                float(metadata["elabFTW"]["laser_status"]["probe_profile_x"]),
                float(metadata["elabFTW"]["laser_status"]["probe_profile_y"]),
            ]

        # calculate temporal resolution
        if (
            "laser_status" in metadata["elabFTW"]
            and "pump_pulse_duration" in metadata["elabFTW"]["laser_status"]
            and "probe_pulse_duration" in metadata["elabFTW"]["laser_status"]
        ):
            metadata["elabFTW"]["laser_status"]["temporal_resolution"] = np.sqrt(
                metadata["elabFTW"]["laser_status"]["pump_pulse_duration"] ** 2
                + metadata["elabFTW"]["laser_status"]["probe_pulse_duration"] ** 2,
            )

        # fix preparation date
        if "sample" in metadata["elabFTW"] and "preparation_date" in metadata["elabFTW"]["sample"]:
            metadata["elabFTW"]["sample"]["preparation_date"] = (
                datetime.datetime.strptime(
                    metadata["elabFTW"]["sample"]["preparation_date"],
                    "%Y-%m-%d",
                )
                .replace(tzinfo=datetime.timezone.utc)
                .isoformat()
            )

        # fix polarizations
        if (
            "scan" in metadata["elabFTW"]
            and "pump_polarization" in metadata["elabFTW"]["scan"]
            and isinstance(metadata["elabFTW"]["scan"]["pump_polarization"], str)
        ):
            if metadata["elabFTW"]["scan"]["pump_polarization"] == "s":
                metadata["elabFTW"]["scan"]["pump_polarization"] = 90
            elif metadata["elabFTW"]["scan"]["pump_polarization"] == "p":
                metadata["elabFTW"]["scan"]["pump_polarization"] = 0
            else:
                try:
                    metadata["elabFTW"]["scan"]["pump_polarization"] = float(
                        metadata["elabFTW"]["scan"]["pump_polarization"],
                    )
                except ValueError:
                    pass

        if (
            "scan" in metadata["elabFTW"]
            and "probe_polarization" in metadata["elabFTW"]["scan"]
            and isinstance(metadata["elabFTW"]["scan"]["probe_polarization"], str)
        ):
            if metadata["elabFTW"]["scan"]["probe_polarization"] == "s":
                metadata["elabFTW"]["scan"]["probe_polarization"] = 90
            elif metadata["elabFTW"]["scan"]["probe_polarization"] == "p":
                metadata["elabFTW"]["scan"]["probe_polarization"] = 0
            else:
                try:
                    metadata["elabFTW"]["scan"]["probe_polarization"] = float(
                        metadata["elabFTW"]["scan"]["probe_polarization"],
                    )
                except ValueError:
                    pass

        if (
            "scan" in metadata["elabFTW"]
            and "pump2_polarization" in metadata["elabFTW"]["scan"]
            and isinstance(metadata["elabFTW"]["scan"]["pump2_polarization"], str)
        ):
            if metadata["elabFTW"]["scan"]["pump2_polarization"] == "s":
                metadata["elabFTW"]["scan"]["pump2_polarization"] = 90
            elif metadata["elabFTW"]["scan"]["pump2_polarization"] == "p":
                metadata["elabFTW"]["scan"]["pump2_polarization"] = 0
            else:
                try:
                    metadata["elabFTW"]["scan"]["pump2_polarization"] = float(
                        metadata["elabFTW"]["scan"]["pump2_polarization"],
                    )
                except ValueError:
                    pass

        # fix pump status
        if "scan" in metadata["elabFTW"] and "pump_status" in metadata["elabFTW"]["scan"]:
            try:
                metadata["elabFTW"]["scan"]["pump_status"] = (
                    "open" if int(metadata["elabFTW"]["scan"]["pump_status"]) else "closed"
                )
            except ValueError:
                pass
        if "scan" in metadata["elabFTW"] and "pump2_status" in metadata["elabFTW"]["scan"]:
            try:
                metadata["elabFTW"]["scan"]["pump2_status"] = (
                    "open" if int(metadata["elabFTW"]["scan"]["pump2_status"]) else "closed"
                )
            except ValueError:
                pass

        # remove pump information if pump not applied:
        if metadata["elabFTW"]["scan"].get("pump_status", "closed") == "closed":
            if "pump_photon_energy" in metadata["elabFTW"].get("laser_status", {}):
                del metadata["elabFTW"]["laser_status"]["pump_photon_energy"]
            if "pump_repetition_rate" in metadata["elabFTW"].get("laser_status", {}):
                del metadata["elabFTW"]["laser_status"]["pump_repetition_rate"]
        else:
            # add pulse energy if applicable
            try:
                metadata["elabFTW"]["scan"]["pump_pulse_energy"] = (
                    metadata["file"]["trARPES:Pump:Power.RBV"]
                    / metadata["elabFTW"]["laser_status"]["pump_repetition_rate"]
                )
            except KeyError:
                pass

        if metadata["elabFTW"]["scan"].get("pump2_status", "closed") == "closed":
            if "pump2_photon_energy" in metadata["elabFTW"].get("laser_status", {}):
                del metadata["elabFTW"]["laser_status"]["pump2_photon_energy"]

        return metadata


def get_archiver_data(
    archiver_url: str,
    archiver_channel: str,
    ts_from: float,
    ts_to: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract time stamps and corresponding data from and EPICS archiver instance

    Args:
        archiver_url (str): URL of the archiver data extraction interface
        archiver_channel (str): EPICS channel to extract data for
        ts_from (float): starting time stamp of the range of interest
        ts_to (float): ending time stamp of the range of interest

    Returns:
        tuple[np.ndarray, np.ndarray]: The extracted time stamps and corresponding data
    """
    iso_from = datetime.datetime.utcfromtimestamp(ts_from).isoformat()
    iso_to = datetime.datetime.utcfromtimestamp(ts_to).isoformat()
    req_str = archiver_url + archiver_channel + "&from=" + iso_from + "Z&to=" + iso_to + "Z"
    with urlopen(req_str) as req:
        data = json.load(req)
        secs = [x["secs"] + x["nanos"] * 1e-9 for x in data[0]["data"]]
        vals = [x["val"] for x in data[0]["data"]]

    return (np.asarray(secs), np.asarray(vals))
