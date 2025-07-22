#----------------------------------------------------------
# Zire Test Kit - SiPM Data Fitting Module
# Author: Elisabetta Casilli and Diptiranjan Pattanaik
# Date: 2025-07-21
#----------------------------------------------------------

from modules.ZLogger import get_logger
from modules.build_channel_map import build_channel_map
from modules.config_loader import get_channel_map_paths
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, peak_widths                # Necessary for peak analysis
from scipy.optimize import curve_fit
import os, csv
import pandas as pd
import numpy  as np
import uproot


# Initialize the logger
log = get_logger()

#--------------------------------------------------------------------------------------
def perform_task(par):
    log.info("Performing fitting to the SiPM data (Fingerplots)...")

    # Get the root file name based on the pst channel and operating voltage
    rootfile_name = get_rootfile_name(par)
    if rootfile_name is None:
        exit(1)
    else:
        log.info(f"[Input] : {rootfile_name}")

    # Get the branch name from the channel map
    branch_name = get_channel_map_key(par['map'])
    if branch_name is None:
        log.error("Failed to get the branch name from the channel map.")
        exit(1)
    
    plot_title = f"[SiPM CH{par['pst_ch']} | OP volt.:{par['op_voltage']}V] [DAQ{par['map'][0]}_ASIC{par['map'][1]}_CH{par['map'][2]}] [{branch_name}]"


    # Get the SiPM data from the ROOT file
    fit_data = get_sipm_data(rootfile_name, branch_name)
    if fit_data is None:
        log.error(f"Failed to read data from the ROOT file '{rootfile_name}'")
        exit(1)    

    # Rebin the data for better fitting
    rebin_factor   = par["fit_utils"]["rebin_factor"]
    peak_threshold = par["fit_utils"]["peak_threshold"]
    peak_distance  = par["fit_utils"]["peak_distance"]

    fit_data_rebinned, bin_edges, bin_centers = rebin_data(fit_data, rebin_factor=rebin_factor)
    
    # Check this parameter in case of rebinning
    binned_distance = int(peak_distance/rebin_factor)
    binned_height   = int(peak_threshold*rebin_factor)

    # Basic peak finding
    peaks, _ = find_peaks(fit_data_rebinned, 
                          height=binned_height,     # Minimum height of the peaks   
                          distance=binned_distance, # Minimum distance between peaks
                          prominence=10             # Helps in suppressing the noise (I don't understand this parameter)
                          )
    peak_widths_info = peak_widths(fit_data_rebinned, peaks, rel_height=0.9)

    fit_window = peak_widths_info[0]

    # Fit and extract the parameters of the peaks
    A, means, sigmas, fit_ranges = fit_peaks(bin_centers, fit_data_rebinned, peaks, window=fit_window)

    # Calculate the average offset between peaks
    mean_peak_distance = get_mean_peak_distance(means)

    # Accumulate the fitted parameters
    fitted_params = list(zip(A, means, sigmas, fit_ranges))
    
    # Print the peaks and fitted parameters for comparison
    log.info(f"Number of peaks found: {len(peaks)}")
    log.info(f"Mean peak distance: {mean_peak_distance:.2f} ADC counts")
    # for i, (peak, width, mu, sig) in enumerate(zip(peaks, peak_widths_info[0], means, sigmas)):
    #     if mu is None or sig is None: 
    #         log.warning(f"[{i}] Failed to fit peak at bin index {peak}, skipping...")
    #         continue
    #     log.info(f"Peak [{i:2d}] ScannedX = {bin_centers[peak]:.1f} | Width: {width:.2f} ADC counts")
    #     log.info(f"Peak [{i:2d}] Fitted μ = {mu:.2f}| Sigma = {sig:.2f} ADC counts\n")

    
    hist_mean = np.average(bin_centers, weights=fit_data_rebinned)
    variance = np.average((bin_centers - hist_mean) ** 2, weights=fit_data_rebinned)
    hist_rms = np.sqrt(variance)
    # Plot the SiPM data
    plot_sipm_data(fit_data_rebinned, bin_centers, plot_title, 
                   xlim=(hist_mean-3*hist_rms, hist_mean+3*hist_rms), 
                   fitted_params=fitted_params,
                   peak_distance=mean_peak_distance
                   )

    return None

def get_mean_peak_distance(means):
    """Calculate the average distance between peaks, ignoring None values."""
    # Remove None values
    filtered_means = [m for m in means if m is not None]

    if len(filtered_means) < 2:
        log.warning("Not enough valid peaks to calculate mean distance.")
        return filtered_means[0] if filtered_means else 0.0

    # Calculate the differences between consecutive means
    distances = np.diff(filtered_means)
    
    # Calculate the mean distance
    mean_distance = np.mean(distances)
    
    return mean_distance
#--------------------------------------------------------------------------------------
def fit_peaks(bin_centers, hist_data, peaks, window=30):
    means = []
    sigmas = []
    amps = []
    fit_ranges = []  # New list

    for i, peak_id in enumerate(peaks):
        mu_guess = bin_centers[peak_id]
        A_guess = hist_data[peak_id]
        fit_window = min(25, window[i])         # Check and balance the fit window
        if fit_window < 5:
            log.warning(f"Fit window for peak {i} is too small ({fit_window}), skipping...")
            continue
        sigma_guess = (bin_centers[1] - bin_centers[0]) * fit_window / 6

        fit_min = mu_guess - fit_window
        fit_max = mu_guess + fit_window

        # print("FIT RANGE:", fit_min, fit_max)
        fit_ranges.append((fit_min, fit_max))  # Store per-peak range

        mask = (bin_centers >= fit_min) & (bin_centers <= fit_max)
        x_data = bin_centers[mask]
        y_data = hist_data[mask]

        if len(x_data) < 5:
            means.append(None)
            sigmas.append(None)
            amps.append(None)
            continue

        try:
            popt, _ = curve_fit(gauss, x_data, y_data, p0=[A_guess, mu_guess, sigma_guess])
            a, mu, sigma = popt
            amps.append(a)
            means.append(mu)
            sigmas.append(sigma)

        except RuntimeError:
            amps.append(None)
            means.append(None)
            sigmas.append(None)

    return amps, means, sigmas, fit_ranges

#--------------------------------------------------------------------------------------
def rebin_data(temp_data, rebin_factor=10):
    """    Rebin the data by summing over the specified rebin size.
    Args:
        data (np.ndarray): The input data to be rebinned.
        rebin_size (int): The size of the bins to rebin the data.

    Returns:
    - rebinned_hist: np.ndarray
    - rebinned_edges: np.ndarray
    - rebinned_centers: np.ndarray
    """
    n_bins = len(temp_data)
    usable_bins = (n_bins // rebin_factor) * rebin_factor

    bin_edges = np.arange(len(temp_data) + 1)

    # Rebin contents
    rebinned_hist = temp_data[:usable_bins].reshape(-1, rebin_factor).sum(axis=1)

    # Rebin edges
    rebinned_edges = bin_edges[:usable_bins+1][::rebin_factor]
    
    # Recalculate centers for fitting
    rebinned_centers = 0.5 * (rebinned_edges[:-1] + rebinned_edges[1:])
    
    return rebinned_hist, rebinned_edges, rebinned_centers

#--------------------------------------------------------------------------------------
def plot_sipm_data(temp_data, bin_cnt, title_name, xlim=(3400, 7000), fitted_params=None, peak_distance=None):
    """    Plot the SiPM data.
    Args:
        fit_data (np.ndarray): The SiPM data as a NumPy array.
        branch_name (str): The branch name to be used in the plot title.
        xlim (tuple): The x-axis limits for the plot.
        fit
    """

    plt.figure(figsize=(10, 6))
    plt.step(bin_cnt, temp_data, where='post', color='black', label='Data')


    # Plot fitted Gaussian curves if provided
    if fitted_params is not None:
        for i, (amp, mu, sigma, (fit_min, fit_max)) in enumerate(fitted_params):
            if None in (amp, mu, sigma):
                continue
            x_fit = np.linspace(fit_min, fit_max, 300)
            y_fit = gauss(x_fit, amp, mu, sigma)
            plt.plot(x_fit, y_fit, '-', label=f"Peak{i:02d} @ {mu:.1f}", linewidth=2.0)

    if peak_distance is not None:
        plt.text(0.05, 0.95, f"Mean peak distance: {peak_distance:.2f}", 
                transform=plt.gca().transAxes,  # coordinates relative to axes (0 to 1)
                fontsize=13, verticalalignment='top', fontweight='bold', color='red')
    # Plot
    plt.xlabel("ADC counts", fontweight='bold')
    plt.ylabel("Counts", fontweight='bold')
    plt.xlim(xlim)
    plt.title(f"{title_name}", fontweight='bold')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    # output_dir = "plots"
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, f"{branch_name}_sipm_plot.png")
    # plt.savefig(output_file)
    
    # log.info(f"Plot saved to {output_file}")
    
    # Show the plot
    plt.show()

#--------------------------------------------------------------------------------------
def get_sipm_data(rootfile_name, branch_name):
    """    Get the SiPM data from the ROOT file.
    Args:
        rootfile_name (str): The name of the ROOT file.
        branch_name (str): The branch name to read from the ROOT file.
    Returns:
        np.ndarray: The SiPM data as a NumPy array.
    """
    
    try:
        with uproot.open(rootfile_name) as file:
            log.info(f"Loading data from {rootfile_name}...")
            tree = file["output_tree"]
            data = tree.arrays(library="pd")
            fit_data = data[branch_name].to_numpy()
            log.info(f"Data loaded successfully from branch '{branch_name}'")
            return fit_data
    except Exception as e:
        log.error(f"Error reading ROOT file '{rootfile_name}': {e}")
        return None

#--------------------------------------------------------------------------------------
def get_rootfile_name(par):
    """    Get the root file name from the configuration.
    Args:
        par (dict): A dictionary containing the parameters.
    Returns:
        str: The root file name.
    """
    
    data_dir = par['data_dir']
    pst_ch = par['pst_ch']
    op_voltage = par['op_voltage']
    if not os.path.exists(data_dir):
        log.error(f"Data directory '{data_dir}' does not exist.")
        return None
    
    # Get the data path from the configuration
    rootfile_name = f"{data_dir}/PST_CH{pst_ch:02d}/PST_Ch{pst_ch:02d}_{op_voltage}V/data.root"
    if not os.path.exists(rootfile_name):
        log.error(f"Data file '{rootfile_name}' does not exist.")
        return None
    
    return rootfile_name

#--------------------------------------------------------------------------------------
def get_channel_map_key(map_tuple):

    """    Get the channel map key from the configuration and return the branch name.
    Args:
        map_tuple (tuple): A tuple containing (daq, asic, channel).
    Returns:
        str: The branch name corresponding to the channel map key.
    """
    
    # Read the channel map files from the configuration
    ch_map_files = get_channel_map_paths()

    # Map the channels and assign physical names
    zire_channel_map = build_channel_map(ch_map_files["pst_map_file"],
                                         ch_map_files["calog_map_file"],
                                         ch_map_files["acs_map_file"])

    daq  = map_tuple[0]
    asic = map_tuple[1]
    ch   = map_tuple[2]

    # Set the channel map key
    ch_map = f"DAQ{daq}_ASIC{asic}_CH{ch}_HG"

    # Initialize the branch name of the tree
    branch_name = None

    # Get the physical name that represents the branch in the ROOT file
    if ch_map not in zire_channel_map:
        log.error(f"Channel map '{ch_map}' not found in the channel mapping.")
        return None
    # Get the branch name from the channel map
    else:
        # If the channel map exists, get the branch name
        branch_name = zire_channel_map.get(ch_map, None)

    # print(ch_map, branch_name)

    return branch_name

#--------------------------------------------------------------------------------------
# Define Gaussian function 
def gauss(x, A, mu, sigma, offset=0):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset   

# Define linear function
def linear(x, m, q):
    return m * x + q


