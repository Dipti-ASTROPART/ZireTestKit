import os
import json
from modules.ZLogger import get_logger

log = get_logger()

def perform_task(params):
    print("[Placeholder] : For generating the json file for the Zire configuration")

    log.info(params)

    num_asics = params['num_asic']
    nchannels = params['channels_per_asic']
    input_DACref_daq1 = 0
    input_DACref_daq2 = 1
    default_DAC = 127

    output_file = params['filename']
    
    config = {
        "identifier": "master",
        "ipaddress": "192.168.102.178",
        "config": {
            "trigger": {
                "concentrator_mode": 3,
                "concentrator_periodic": 1000000,
                "concentrator_logic": 0,
                "ctrig_edit_counts": 1000,
                "daq1": "ftk",
                "daq1_trg_en": 0,
                "daq1_trg_mode": "000",
                "daq1_trg_src": 1,
                "daq1_trg_period": 500000,
                "daq2": "pst",
                "daq2_trg_en": 0,
                "daq2_trg_mode": "000",
                "daq2_trg_src": 1,
                "daq2_trg_period": 500000,
                "validation_src": 0,
                "validation": False,
                "timeout_valid": 250000,
                "discard": False,
                "fake_gen": False,
                "mode": "master",
                "trigpsc": False,
                "holdwin": 10,
                "HGpdorth": 0,
                "LGpdorth": 0,
                "PSCbypass": False,
                "rstb_en": False
            },
            "hv_settings": {
                "ftk_slave": 0,
                "ftk_master": 0,
                "pst": 0,
                "calo": 0
            },
            "daq_1": {},
            "daq_2": {}
        }
    }

    for daq_key, input_DACref in [("daq_1", input_DACref_daq1), ("daq_2", input_DACref_daq2)]:
        for asic_id in range(1, num_asics + 1):
            asic_name = f"asic_{asic_id}"
            asic_block = {
                "general": {
                    "fastsh_source": "hg",
                    "HGshtime": 0,
                    "LGshtime": 0,
                    "timethrs": 300,
                    "chargeth": 300,
                    "input_DACref": input_DACref
                }
            }
            # Add channels
            for ch in range(nchannels):
                asic_block[f"ch_{ch}"] = {
                    "inputDAC": default_DAC,
                    "DACtime": 7,
                    "mask": False,
                    "test_hg": False,
                    "test_lg": False,
                    "gain_hg": 1,
                    "gain_lg": 1
                }
            config["config"][daq_key][asic_name] = asic_block

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)

    print(f"✅ Configuration saved to: {output_file}")

    return None

