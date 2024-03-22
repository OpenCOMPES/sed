import os
from pathlib import Path

import sed
from sed import SedProcessor

config_file = Path(sed.__file__).parent / "config/flash_example_config.yaml"
data_path = "./tutorial/"  # Put in Path to a storage of at least 20 Gbyte free space.
if not os.path.exists(data_path + "/Gd_W110_flash.zip"):
    os.system(
        f"curl -L --output {data_path}/Gd_W110_flash.zip https://zenodo.org/records/10658470/files/single_event_data.zip",
    )
if not os.path.isdir(data_path + "/flash_data"):
    if not os.path.isdir(data_path + "/analysis_data") or not os.path.isdir(
        data_path + "/calibration_data",
    ):
        os.system(f"unzip -d {data_path} -o {data_path}/Gd_W110_flash.zip")

    os.system(f"mkdir {data_path}/flash_data")
    os.system(f"mv {data_path}/analysis_data/*/*.h5 {data_path}/flash_data")
    os.system(f"mv {data_path}/calibration_data/*/*.h5 {data_path}/flash_data")

config_override = {
    "core": {
        "paths": {
            "data_raw_dir": data_path + "/flash_data/",
            "data_parquet_dir": data_path + "/parquet/",
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
