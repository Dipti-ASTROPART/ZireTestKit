from modules.ZLogger import get_logger
import os, csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import uproot
from modules.config_loader import get_channel_map_paths
from modules.build_channel_map import build_channel_map

log = get_logger()

NUM_DAQ = 2
NUM_ASIC_PER_DAQ = 4
NUM_CH = 32


def perform_task(par):
    input_file = par["data_dir"]+ "/data.csv"
    output_file = par["data_dir"] + "/data.root"

    if not os.path.exists(input_file):
        log.error(f"Error: The file '{input_file}' does not exist.")
        return None

    log.info(f'Reading CSV file: {input_file}')

    ch_map_files = get_channel_map_paths()
    zire_channel_map = build_channel_map(ch_map_files["pst_map_file"],
                                         ch_map_files["calog_map_file"],
                                         ch_map_files["acs_map_file"])

    df = pd.read_csv(input_file)
    log.info(f"CSV file '{input_file}' read successfully with and {df.shape[1]} columns.")
    df.columns = map_columns(df.shape[1])

    # for col in df.columns:
    #     if col not in zire_channel_map: continue
    #     print(col, zire_channel_map.get(col, "Not Mapped"))    

    # filter and rename only the mapped columns
    mapped_columns = {
        col: zire_channel_map[col] 
        for col in df.columns 
        if col in zire_channel_map 
    }

    # Change the column names to the physical names
    df_mapped = df[list(mapped_columns.keys())].rename(columns=mapped_columns)
    
    # Convert the DataFrame to a dictionary with float32 values for ROOT compatibility
    root_format_data = {
        col: df_mapped[col].values.astype(np.float32) 
        for col in df_mapped.columns
    }

    # Write the DataFrame to a ROOT file
    with uproot.recreate(output_file) as root_file:
        root_file["output_tree"] = root_format_data
    
    log.info(f"Data written to ROOT file: {output_file}")

    # # Plot all 8 ASICs (change gain="LG" to plot low-gain)
    # for daq in range(1, NUM_DAQ + 1):
    #     if daq != 1: continue
    #     for asic in range(1, NUM_ASIC_PER_DAQ + 1):
    #         if asic != 1: continue
    #         plot_asic(df, zire_channel_map, daq_num=daq, asic_num=asic, gain="HG", bin_start=3000, bin_count=2000)
    
    return None

# Map the columns with correct DAQ–ASIC–CH–Gain
def map_columns(num_columns=512):
    column_map = []
    for col in range(num_columns):
        daq = col // 256
        gain = "LG" if (col % 256) < 128 else "HG"
        within_gain = col % 128
        asic = within_gain // 32
        channel = within_gain % 32
        col_name = f'DAQ{daq+1}_ASIC{asic+1}_CH{channel}_{gain}'
        column_map.append(col_name)
        # print(col, col_name)
    return column_map

# Function to plot 32 channels from one ASIC
def plot_asic(df, physical_name, daq_num, asic_num, gain="HG", bin_start=0, bin_count=4000):
    fig, axs = plt.subplots(4, 8, figsize=(20, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    fig.suptitle(f'DAQ{daq_num}_ASIC{asic_num} - {gain}', fontsize=16)

    for ch in range(0, 32):  # Channels 1 to 32
        col_name = f'DAQ{daq_num}_ASIC{asic_num}_CH{ch}_{gain}'
        if col_name not in df.columns:
            print(f"Missing column: {col_name}")
            continue

        y = df[col_name].values[bin_start:bin_start + bin_count]
        x = range(bin_start, bin_start + bin_count)
        axs[ch].plot(x, y, label=col_name)
        axs[ch].set_title(physical_name.get(col_name), fontsize=8)
        axs[ch].set_xlim(bin_start, bin_start + bin_count)
        axs[ch].set_ylim(0, 2500)
        axs[ch].tick_params(labelsize=6)

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()

# --- Main ---
