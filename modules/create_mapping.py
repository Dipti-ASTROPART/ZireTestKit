import pandas as pd
import json

def parse_csv_to_mapping(calo_csv, pst_csv, acs_csv):
    mapping = {}

    # CALOg
    for _, row in calo_csv.iterrows():
        try:
            daq = int(row["DAQ"])
            crystal = int(row["Crystal"])
            size = int(row["Sensor_Size"])
            asic = int(row["Asic"])+1
            chan = int(row["Channel"])
            layer = int(row["Layer"])
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
        key = f"daq_{daq}_asic_{asic}_ch_{chan}"
        mapping[key] = f"calo_{crystal}_{suffix}"

    # PST
    for _, row in pst_csv.iterrows():
        try:
            daq = int(row["DAQ"])
            layer = int(row["Layer"])
            bar = int(row["Bar"])
            asic = int(row["Asic"])+1
            chan = int(row["Channel"])
        except ValueError:
            continue
        suffix = chr(65 + bar)  # A, B, C
        key = f"daq_{daq}_asic_{asic}_ch_{chan}"
        mapping[key] = f"pst_{layer}_{suffix.lower()}"

    # ACS
    for _, row in acs_csv.iterrows():
        if row["Asic"] == -999 or row["Channel"] == -999 or row["DAQ"] == -999:
            continue
        try:
            daq = int(row["DAQ"])
            asic = int(row["Asic"])+1
            chan = int(row["Channel"])
            pos = int(row["Tile"])
            sensor = int(row["Sensor"])
            key = f"daq_{daq}_asic_{asic}_ch_{chan}"
            mapping[key] = f"acs{pos}_3x3_{sensor}"
        except ValueError:
            continue

    return mapping

# Load CSVs
calo_csv = pd.read_csv("calo.csv")
pst_csv = pd.read_csv("pst.csv")
acs_csv = pd.read_csv("acs.csv")

# Create mapping
mapping = parse_csv_to_mapping(calo_csv, pst_csv, acs_csv)

# Save to file
with open("mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)

print("mapping.json created with", len(mapping), "entries.")
