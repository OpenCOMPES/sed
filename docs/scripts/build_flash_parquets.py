import os
from importlib.util import find_spec

from sed import SedProcessor
from sed.dataset import dataset

package_dir = os.path.dirname(find_spec("sed").origin)

config_file = package_dir + "/config/flash_example_config.yaml"

dataset.get("Gd_W110", root_dir="./tutorial")
data_path = dataset.dir


config_override = {
    "core": {
        "paths": {
            "raw": data_path,
            "processed": data_path + "/processed/",
        },
    },
}

runs = ["44762", "44797", "44798", "44799", "44824", "44825", "44826", "44827"]
for run in runs:
    sp = SedProcessor(
        runs=run,
        config=config_override,
        system_config=config_file,
        collect_metadata=False,
    )

dataset.get("W110", root_dir="./tutorial")
data_path = dataset.dir


config_override = {
    "core": {
        "paths": {
            "raw": data_path,
            "processed": data_path + "/processed/",
        },
    },
}

runs = ["44498", "44455"]
for run in runs:
    sp = SedProcessor(
        runs=run,
        config=config_override,
        system_config=config_file,
        collect_metadata=False,
    )
