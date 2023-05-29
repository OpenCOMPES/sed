import json
import os
from pathlib import Path

import requests

# Set the headers for the HTTP requests
headers = {"Content-Type": "application/json", "Accept": "application/json"}

# Load user details from a JSON file
file_path = Path("/home/sohailmu/sed/sed/loader/flash/user.json")
if os.path.isfile(file_path):
    print("Scicat metadata can be read.")
    user_details = json.load(open(file_path))
    # Extract necessary information from user details
    scicat_url = user_details["scicat_url"]
    username = user_details["username"]
    pw = user_details["pw"]
else:
    print("Can not fetch metadata since user.json is undefined")
    scicat_url = ""
    username = ""
    pw = ""


# Function to retrieve metadata for a given beamtime ID and list of runs
def get_metadata(beamtime_id, runs, metadata):
    # If metadata is not provided, initialize it as an empty dictionary
    if metadata is None:
        metadata = {}

    # Iterate over the list of runs
    for run in runs:
        pid = f"{beamtime_id}/{run}"
        # Retrieve metadata for each run and update the overall metadata dictionary
        metadata_run = get_metadata_per_run(pid)
        metadata.update(metadata_run)  # TODO: Not correct for multiple runs

    return metadata


# Function to retrieve metadata for a specific run based on the PID
def get_metadata_per_run(pid: str):
    # Create the dataset URL using the PID
    dataset_response = requests.get(_create_dataset_url_by_PID(pid))

    # If the dataset request is successful, return the retrieved metadata
    # as a JSON object
    if dataset_response.ok:
        return dataset_response.json()
    else:
        # If the request fails, raise an exception with the error message
        raise Exception(f"{dataset_response.text}")


# Helper function to create the dataset URL based on the PID
def _create_dataset_url_by_PID(pid: str) -> str:
    npid = ("/" + pid).replace(
        "/",
        "%2F",
    )  # Replace slashes in the PID with URL-encoded slashes
    dataset_url_by_pid = (
        f"{scicat_url}/RawDatasets/{npid}?access_token={_get_token()}"
    )
    return dataset_url_by_pid


# Helper function to retrieve the access token for authentication
def _get_token() -> str:
    token_url = f"{scicat_url}/Users/login"
    # Send a POST request to the token URL with the username and password
    token_response = requests.post(
        token_url,
        headers=headers,
        json={"username": username, "password": pw},
    )

    # If the token request is successful, return the access token from the response
    if token_response.ok:
        return token_response.json()["id"]
    else:
        # If the request fails, raise an exception with the error message
        raise Exception(f"{token_response.text}")
