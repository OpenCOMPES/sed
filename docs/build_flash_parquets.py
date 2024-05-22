from pathlib import Path

import sed
from sed import SedProcessor
from sed.dataset import load_dataset

config_file = Path(sed.__file__).parent / "config/flash_example_config.yaml"
data_path = load_dataset(
    "Gd_W(110)",
    "./tutorial/",
)  # Put in Path to a storage of at least 20 Gbyte free space.

config_override = {
    "core": {
        "paths": {
            "data_raw_dir": data_path,
            "data_parquet_dir": str(data_path) + "/processed/",
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
