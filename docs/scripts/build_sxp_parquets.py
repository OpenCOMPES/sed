import os
from importlib.util import find_spec

from sed import SedProcessor
from sed.dataset import dataset

package_dir = os.path.dirname(find_spec("sed").origin)

config_file = package_dir + "/config/sxp_example_config.yaml"

dataset.get("Au_Mica", root_dir="./tutorial")
data_path = dataset.dir


config_override = {
    "core": {
        "paths": {
            "raw": data_path,
            "processed": data_path + "/processed/",
        },
    },
}

runs = [
    "0058",
    "0059",
    "0060",
    "0061",
    "0064",
    "0065",
    "0066",
    "0067",
    "0068",
    "0069",
    "0070",
    "0071",
    "0072",
    "0073",
    "0074",
]
for run in runs:
    sp = SedProcessor(
        runs=run,
        config=config_override,
        system_config=config_file,
        collect_metadata=False,
    )
