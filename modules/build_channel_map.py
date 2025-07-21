from modules.ZLogger import get_logger
import os, csv
import pandas as pd

log = get_logger()

def build_channel_map(pst_map_file, calog_map_file, acs_map_file):
    mapping = {}

    # Check if the channel map files exist and read them
    if pst_map_file and os.path.exists(pst_map_file):
        log.info(f"Reading PST channel map from {pst_map_file}")
        pst_csv = pd.read_csv(pst_map_file)
    else:
        log.warning(f"PST channel map file '{pst_map_file}' not found or not specified.")
        exit(1)

    if calog_map_file and os.path.exists(calog_map_file):
        log.info(f"Reading Calog channel map from {calog_map_file}")
        calo_csv = pd.read_csv(calog_map_file)
    else:
        log.warning(f"Calog channel map file '{calog_map_file}' not found or not specified.")
        exit(1)

    if acs_map_file and os.path.exists(acs_map_file):
        log.info(f"Reading ACS channel map from {acs_map_file}")
        acs_csv = pd.read_csv(acs_map_file)
    else:
        log.warning(f"ACS channel map file '{acs_map_file}' not found or not specified.")
        exit(1)

    # print(pst_csv["Layer"])
    
    # Map PST channels
    gain = {"HG", "LG"}
    for _, row in pst_csv.iterrows():
        layer   = int(row['Layer'])
        bar     = int(row['Bar'])
        asic    = int(row['Asic'])
        daq     = int(row['DAQ'])
        channel = int(row['Channel'])

        suffix  = chr(65 + bar) # Convert bar number to letter (0 -> A, 1 -> B, etc.)

        for g in gain:
            key     = f"DAQ{daq}_ASIC{asic}_CH{channel}_{g}"
            mapping[key] = f"PST_Layer{layer}_{suffix}_{g}"

        
    # Map Calog channels
    gain = {"HG", "LG"}
    for _, row in calo_csv.iterrows():
        if row["Asic"] == -999 or row["Channel"] == -999 or row["DAQ"] == -999:
            continue
        for _, row in calo_csv.iterrows():
            try:
                daq     = int(row["DAQ"])
                crystal = int(row["Crystal"])
                size    = int(row["Sensor_Size"])
                asic    = int(row["Asic"])
                chan    = int(row["Channel"])
                layer   = int(row["Layer"])
            except ValueError:
                continue
            if size == 6:
                suffix = "6x6"
            elif size == 3:
                suffix = "3x3"
            elif size == 1:
                suffix = "1x1"
            else:
                continue
            if layer > 0:
                continue
            for g in gain:
                key = f"DAQ{daq}_ASIC{asic}_CH{chan}_{g}"
                mapping[key] = f"CALOg_Crystal{crystal}_{suffix}_{g}"
                # print(key, mapping[key])

    # Map ACS channels
    gain = {"HG", "LG"}
    for _, row in acs_csv.iterrows():
        if row["Asic"] == -999 or row["Channel"] == -999 or row["DAQ"] == -999:
            continue
        try:
            daq = int(row["DAQ"])
            asic = int(row["Asic"])
            chan = int(row["Channel"])
            pos = int(row["Tile"])
            sensor = int(row["Sensor"])
            key = f"DAQ{daq}_ASIC{asic}_CH{chan}"
                        
            for g in gain: 
                key_gain = f"{key}_{g}"
                mapping[key_gain] = f"ACS{pos}_3x3_{sensor}_{g}"
        except ValueError:
            continue
    return mapping



