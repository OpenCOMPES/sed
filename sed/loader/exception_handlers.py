"Module for purpose of exception handling"

class NoFilesFoundError(Exception):
    """Custom exception for when no files are found in the specified directory."""
    def __init__(self, message):
        self.message = message
