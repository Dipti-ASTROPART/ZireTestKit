 # **ZIRE TESTKIT SOFTWARE USER MANUAL**

The software is developed to do basic data handling and data analysis for zire testkit. 
## Features
- Channel mapping of PST, CALOg and ACS
- Read the CSV files from the DAQ and store them in ROOT format (using uproot)
- Read the data from the ROOT file based on the mapping on the DAQ
- Find the peaks in the finger plot of the SiPM data
- Fit around the peak area and find the peak positions then calibrate the gain
- Visualize data with built-in plotting t

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Dipti-ASTROPART/ZireTestKit.git
    ```
2. Install dependencies:
    ```bash
    pip install numpy pyyaml uproot scipy
    ```

## Usage

1. Launch the main application:
    ```bash
    python zire_testkit.py 
    or
    python zire_testkit.py testkit_config.yaml
    ```
By default, the main function reads _testkit_config.yaml_ as default. If you want to modify your yaml and save as a new, please provide the path to the yaml file.

## **CSV READER**
The output from the DAQ is stored as a CSV file where each column represents a channel in the DAQ. Based on the map, the columns get assigned to their corresponding detector tags. 

There are a total 512 columns sotred in 16384 rows, where each row corresponds to the bin number and the values in the columns represents the bin content. Hence CSV file in a way stores the histogram information in each channel. 

First 256 columns represents the data collected from the DAQ1 and the rest of the 256 columns from DAQ2. Each DAQ has four CITIROC chips that stores information of 32 channels. Each channel is then capable of storing both High Gain and Low Gain data. Hence one Single DAQ contains (32 channels x 4 ASIC x 2 gain) 256 columns. 

- Columns 001-128: **DAQ1**->  Low gain data from 4 ASICs (4 x 32 channels)
- Columns 129-256: **DAQ1**-> High gain data from 4 ASICs (4 x 32 channels)
- Columns 257-384: **DAQ2**->  Low gain data from 4 ASICs (4 x 32 channels)
- Columns 385-512: **DAQ2**-> High gain data from 4 ASICs (4 x 32 channels)

**The information are stored in a ROOT file where each entry corresponds to one row/bin**.