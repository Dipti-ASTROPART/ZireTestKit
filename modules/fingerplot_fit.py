import numpy as np
import math
import matplotlib.pyplot as plt
#from scipy.integrate import simps
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os
import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser(description="Process fingerplots and compute gain vs HV")
parser.add_argument("files", nargs='+', help="Input filenames (e.g. *_HV_*.csv)")
parser.add_argument("--column", help="Selected the data column from cvs")
parser.add_argument("--plot", action='store_true', help="Enable debug plots")
args = parser.parse_args()

def gauss(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset

def linear(x, m, q):
    return m*x+q

filelist = args.files
check_plot = args.plot
data_column = args.column
HV_list, gains, gains_err = [], [], []

with open('mapping.json') as json_data:
    map = json.load(json_data)
    json_data.close()

cnames = []

#daq_1_asic_0_ch_3
for daq_id in [1,2]:
    for ch_type in ["lg","hg"]:
        for asic_id in range(4):
            for ch_id in range(32):
                _key = f'daq_{daq_id}_asic_{asic_id}_ch_{ch_id}'
                if _key in map:
                    cnames.append( map[_key]+f'_{ch_type}' )
                else:
                    cnames.append( _key+f'_{ch_type}' )               


for filename in filelist:
    if check_plot: print(f"-->Processing {filename}...")

    # Extract HV from filename
    match = re.search(r"_HV_(\d+(?:\.\d+)?)", filename)
    if not match:
        print(f"Could not extract HV from filename {filename}")
        continue
    HV = float(match.group(1))
    HV_list.append(HV)

    # READ FILE....
    #data = calog[:, sipm_index, cryst_index] #MODIFICARE ?!?!?!
    df = pd.read_csv(filename, sep=',', names=cnames)
    #df.head()

    array = df[data_column]
    bins = np.linspace(0, len(array), len(array)+1)
    data = np.histogram(np.ones_like(array), weights=array, bins=bins)

    #hist, bins = np.histogram(data, bins=200)
    bin_cent = 0.5 * (bins[:-1] + bins[1:])
    if check_plot:
        plt.plot(bin_cent, data, label=filename) #data[0] ??
        plt.xlabel("ADC counts")
        plt.ylabel("Events")
        plt.title(f"Fingerplot - HV = {HV}")
        plt.legend()

    # Peak detection
    peaks, _ = find_peaks(data, height=15, distance=25, prominence=0.5)
    res_half = peak_widths(data, peaks, rel_height=0.8) # rel_height=0.5 = FWHM

    if check_plot: print(f"Number of peaks found: {len(peaks)}")
    mu_list, mu_err_list = [], []

    for i, peak in enumerate(peaks):
        half_width = int(np.ceil(res_half[0][i] * 0.75))
        left = max(0, peak - half_width)
        right = min(len(data), peak + half_width)

        x_data = bin_cent[left:right]
        y_data = data[left:right]

        sigma_est = res_half[0][i] * (bin_cent[1] - bin_cent[0]) / 2.355  # FWHM -> sigma
        p0 = [data[peak], bin_cent[peak], sigma_est, 0]
        try:
            params, covariance = curve_fit(gauss, x_data, y_data, p0=p0)
            A, mu, sigma, offset = params
            mu_err = np.sqrt(np.abs(covariance[1][1]))
            mu_list.append(mu)
            mu_err_list.append(mu_err)
            if check_plot: plt.plot(x_data, gauss(x_data, *params), 'r--')
        except:
            print(f"Failed Gaussian fit for peak {i}")

    if check_plot: plt.show()    

    mu_sort = np.array(sorted(mu_list))
    mu_err_sorted = np.array([x for _, x in sorted(zip(mu_list, mu_err_list))])
    gain = np.diff(mu_sort)
    gain_err = np.sqrt(mu_err_sorted[1:]**2 + mu_err_sorted[:-1]**2)

    gain_mean = np.mean(gain)
    gain_std = np.std(gain) / np.sqrt(len(gain)) #mean error

    gains.append(gain_mean)
    gains_err.append(gain_std)
    print(f"HV = {HV:.2f} --> Gain = {gain_mean:.2e} +/- {gain_std:.2e}")

# Linear fit: gain vs HV
HV_array = np.array(HV_list)
gains_array = np.array(gains)
gains_err_array = np.array(gains_err)
plt.errorbar(HV_array, gains_array, yerr=gains_err_array, fmt='o', label='Data')
plt.xlabel("HV")
plt.ylabel("Gain")

params, covariance = curve_fit(linear, HV_array, gains_array, sigma=gains_err_array)
m, q = params
m_err, q_err = np.sqrt(np.diag(covariance))

x_fit = np.linspace(min(HV_array), max(HV_array), 100)
y_fit = linear(x_fit, m, q)
plt.plot(x_fit, y_fit, 'r-')
plt.grid(True)
plt.show()

# Compute working point HV for a target gain
gain_WP = 0.005 # MODIFY!!
HV_WP = (gain_WP - q) / m
print(f"Working point for gain = {gain_WP} is HV = {HV_WP}")
print(f"(computed from fit parameters)")
