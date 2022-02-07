"""
Wrapper module that preprocesses data from any defined source.
Currently, this works for Flash and Lab data sources.
"""

import yaml
from sed.data_sources.flash import FlashLoader
from sed.data_sources.lab import LabLoader


class dataframeReader(FlashLoader, LabLoader):
    """
    The class inherits attributes from the FlashLoader and LabLoader classes
    and calls them depending on predefined source value in config file.
    The dataframe is stored in self.dd.
    """

    def __init__(self, config, runNumber=None, fileNames=None):
        if (runNumber or fileNames) is None:
            raise ValueError("Must provide a run, list of runs, or fileNames!")
        if runNumber and fileNames:
            raise ValueError("Only provide either run number(s) or fileNames")
        # Parse the source value to choose the necessary class
        with open(config) as file:
            config_ = yaml.load_all(file, Loader=yaml.FullLoader)
            for doc in config_:
                if "general" in doc.keys():
                    self.source = doc["general"]["source"]

        if not self.source:
            raise ValueError("Please define data source in config file.")

        if self.source == "flash":
            FlashLoader.__init__(self, runNumber, config)

        if self.source == "lab":
            LabLoader.__init__(self, fileNames, config)

        # data = [data for data in [runNumber, fileNames] if data]

        self.readData()
