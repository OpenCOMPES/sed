from pathlib import Path

import sed
from sed import SedProcessor
from sed.dataset import dataset

config_file = Path(sed.__file__).parent / "config/flash_example_config.yaml"

dataset.get("Gd_W110", root_dir="./tutorial")
data_path = dataset.dir


config_override = {
    "core": {
        "paths": {
            "data_raw_dir": data_path,
            "data_parquet_dir": data_path + "/processed/",
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
