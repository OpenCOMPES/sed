"Module for purpose of exception handling"


class NoFilesFoundError(Exception):
    """Custom exception for when no files are found in the specified directory."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class H5ParsingError(Exception):
    """Exception raised for errors related to parsing H5 files."""

    def __init__(
        self,
        message="Error occurred while parsing H5 file.",
        cause=None,
    ):
        self.message = message
        self.cause = cause
        super().__init__(self.message)

    def __str__(self):
        if self.cause:
            return f"{self.message} Caused by: {self.cause}"
        return self.message
