import os
from importlib.util import find_spec
from typing import Any

from sed import SedProcessor
from sed.dataset import dataset

package_dir = os.path.dirname(find_spec("sed").origin)

config_file = package_dir + "/config/flash_example_config.yaml"

dataset.get("Gd_W110", root_dir="./tutorial")
data_path = dataset.dir


config_override: dict[str, Any] = {
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

dataset.get("Photon_peak", root_dir="./tutorial")
data_path = dataset.dir

config_override = {
    "core": {
        "paths": {
            "raw": data_path,
            "processed": data_path + "/processed/",
        },
    },
    "dataframe": {
        "ubid_offset": 0,
        "channels": {
            "timeStamp": {
                "index_key": "/zraw/TIMINGINFO/TIME1.BUNCH_FIRST_INDEX.1/dGroup/index",
                "dataset_key": "/zraw/TIMINGINFO/TIME1.BUNCH_FIRST_INDEX.1/dGroup/time",
            },
            "pulseId": {
                "index_key": "/zraw/FLASH.FEL/HEXTOF.DAQ/DLD1/dGroup/index",
                "dataset_key": "/zraw/FLASH.FEL/HEXTOF.DAQ/DLD1/dGroup/value",
            },
            "dldPosX": {
                "index_key": "/zraw/FLASH.FEL/HEXTOF.DAQ/DLD1/dGroup/index",
                "dataset_key": "/zraw/FLASH.FEL/HEXTOF.DAQ/DLD1/dGroup/value",
            },
            "dldPosY": {
                "index_key": "/zraw/FLASH.FEL/HEXTOF.DAQ/DLD1/dGroup/index",
                "dataset_key": "/zraw/FLASH.FEL/HEXTOF.DAQ/DLD1/dGroup/value",
            },
            "dldTimeSteps": {
                "index_key": "/zraw/FLASH.FEL/HEXTOF.DAQ/DLD1/dGroup/index",
                "dataset_key": "/zraw/FLASH.FEL/HEXTOF.DAQ/DLD1/dGroup/value",
            },
            "dldAux": {
                "index_key": "/zraw/FLASH.FEL/HEXTOF.DAQ/DLD1/dGroup/index",
                "dataset_key": "/zraw/FLASH.FEL/HEXTOF.DAQ/DLD1/dGroup/value",
            },
            "bam": {
                "index_key": "/zraw/FLASH.SDIAG/BAM.DAQ/4DBC3.HIGH_CHARGE_ARRIVAL_TIME/dGroup/index",
                "dataset_key": "/zraw/FLASH.SDIAG/BAM.DAQ/4DBC3.HIGH_CHARGE_ARRIVAL_TIME/dGroup/value",
            },
            "delayStage": {
                "index_key": "/zraw/FLASH.SYNC/LASER.LOCK.EXP/FLASH1.MOD1.PG.OSC/FMC0.MD22.1.ENCODER_POSITION.RD/dGroup/index",
                "dataset_key": "/zraw/FLASH.SYNC/LASER.LOCK.EXP/FLASH1.MOD1.PG.OSC/FMC0.MD22.1.ENCODER_POSITION.RD/dGroup/value",
            },
            "opticalDiode": {
                "format": "per_train",
                "index_key": "/uncategorised/FLASH.LASER/FLACPUPGLASER1.PULSEENERGY/PG2_incoupl/PULSEENERGY.MEAN/index",
                "dataset_key": "/uncategorised/FLASH.LASER/FLACPUPGLASER1.PULSEENERGY/PG2_incoupl/PULSEENERGY.MEAN/value",
            },
            "gmdBda": {},
            "pulserSignAdc": {},
        },
    },
}

runs = ["40887"]
for run in runs:
    sp = SedProcessor(
        runs=run,
        config=config_override,
        system_config=config_file,
        collect_metadata=False,
    )
