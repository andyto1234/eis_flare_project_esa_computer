import numpy as np
import matplotlib.pyplot as plt
from sunpy.timeseries import TimeSeries
import re
import pandas as pd
import glob
import pickle
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def process_goes_data(goes_ts):
    """Process GOES timeseries with spike removal and buffer zones around removed data."""
    xrsa_df = goes_ts.extract('xrsb').to_dataframe()
    flux = xrsa_df['xrsb']
    
    # Step 1: Initial smoothing with shorter window
    window = '10min'
    rolling_median = flux.rolling(window=window, center=True).median()
    rolling_std = flux.rolling(window=window, center=True).std()
    
    # Step 2: Calculate differences at multiple timescales
    diff_short = flux.diff()
    diff_forward = -flux.diff(-1)
    
    # Calculate rate of change
    rate_of_change = diff_short / flux
    forward_rate = diff_forward / flux

    # Vectorized spike detection criteria
    rapid_drop = (flux.shift(1) - flux) > (rolling_std * 1.5)
    rapid_recovery = (flux.shift(-1) - flux) > (rolling_std * 1.5)
    below_median = flux < (rolling_median * 0.8)
    local_anomaly = abs(flux - rolling_median) > (rolling_std * 2)
    sudden_change = abs(rate_of_change) > 0.5
    
    # Combined criteria
    is_spike = (
        (rapid_drop & rapid_recovery & below_median) |
        (flux < rolling_median * 0.4) |
        (local_anomaly & sudden_change) |
        ((abs(rate_of_change) > 1.0) & (abs(forward_rate) > 1.0))
    )
    
    # Step 4: Protect potential flare points
    flux_smooth = flux.rolling(window='5min', center=True).mean()
    trend = flux_smooth.diff().rolling(window='10min').mean()
    
    potential_flare = (
        (trend > 0) & 
        (flux > rolling_median * 1.2) &
        (flux.rolling('15min').max() > flux.rolling('15min').min() * 1.5)
    )
    
    # Don't clean points that might be part of flares
    points_to_clean = is_spike & ~potential_flare
    
    # Step 5: Add buffer around points to clean
    buffer_points = 50  # Number of points to add before and after
    buffered_points_to_clean = points_to_clean.copy()
    
    # Add buffer before and after each identified spike
    for i in range(buffer_points, len(points_to_clean) - buffer_points):
        if points_to_clean.iloc[i]:
            # Add buffer before the spike
            buffered_points_to_clean.iloc[i-buffer_points:i] = True
            # Add buffer after the spike
            buffered_points_to_clean.iloc[i+1:i+buffer_points+1] = True
    
    # Step 6: Find consecutive groups of points to clean
    cleaned_flux = flux.copy()
    cleaned_flux[buffered_points_to_clean] = np.nan
    
    # Remove any remaining negative or very small values
    cleaned_flux[cleaned_flux <= 1e-9] = np.nan
    
    # Calculate background (handling NaN values)
    bg_window = 100  # minutes
    rolling_min = cleaned_flux.rolling(window=f'{bg_window}min', center=True, min_periods=1).min()
    background = rolling_min.rolling(window='60min', center=True, min_periods=1).mean()
    
    # Subtract background
    background_subtracted = cleaned_flux - background
    background_subtracted[background_subtracted <= 0] = np.nan
    
    return cleaned_flux, background, background_subtracted
    
def extract_flare_time(filename):
    """Extract flare time from filename."""
    # Extract date and time from filename pattern
    match = re.search(r'GOES\d+_(\d{8}_\d{6})_goes', filename)
    if match:
        datetime_str = match.group(1)
        return pd.to_datetime(datetime_str, format='%Y%m%d_%H%M%S')
    return None

def extract_eis_time(filename):
    """Extract flare time from filename."""
    # Extract date and time from filename pattern
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        datetime_str = match.group(1)
        return pd.to_datetime(datetime_str, format='%Y%m%d_%H%M%S')
    return None

def calculate_goes_class_at_time(flux_series, time):
    """Calculate GOES class at specific time with ±2 minute window."""
    window_start = time - pd.Timedelta(minutes=3)
    window_end = time + pd.Timedelta(minutes=3)
    
    # Get flux within time window
    window_flux = flux_series[window_start:window_end]
    if window_flux.empty:
        return 'N/A'
        
    peak_flux = window_flux.max()
    
    if peak_flux <= 0:
        return 'A0.0'
    
    exp = np.floor(np.log10(peak_flux))
    mantissa = peak_flux / (10**exp)
    
    class_map = {-8: 'A', -7: 'B', -6: 'C', -5: 'M', -4: 'X'}
    goes_class = class_map.get(exp, 'X')
    
    return f"{goes_class}{mantissa:.1f}"

def plot_goes_data(goes_ts, filename, master_df, save_path=None):
    """Plot original and background-subtracted GOES data with flare time."""
    flux, background, background_subtracted = process_goes_data(goes_ts)
    flare_time = extract_flare_time(filename)
    
    # Get EIS observation time window
    row = master_df[(master_df['eis_file_time']==str(extract_eis_time(filename))) & 
                    (master_df['hek_flare_peak']==str(flare_time))]
    eis_start = pd.to_datetime(row['eis_file_time'].iloc[0]) if not row.empty else None
    eis_end = pd.to_datetime(row['eis_end_time'].iloc[0]) if not row.empty else None
    
    if flare_time:
        peak_class = calculate_goes_class_at_time(flux, flare_time)
        bg_subtracted_class = calculate_goes_class_at_time(background_subtracted, flare_time)
    else:
        peak_class = 'N/A'
        bg_subtracted_class = 'N/A'
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Add EIS observation window to all plots if times are available
    if eis_start is not None and eis_end is not None:
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(eis_start, eis_end, color='blue', alpha=0.2, label='EIS observation')
    
    # Original data plot
    ax1.plot(flux.index, flux, 'b-', label='Original flux')
    ax1.plot(background.index, background, 'r--', label='Background')
    if flare_time:
        ax1.axvline(x=flare_time, color='k', linestyle='--', label=f'Flare peak time (class {peak_class})')
    ax1.set_yscale('log')
    ax1.set_ylabel('Flux (W/m²)')
    ax1.set_title("GOES 1$-$8 $\mathrm{\AA}$ data with background - "+f"{flare_time}")
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Add GOES class markers for original flux plot first
    ax1_twin = ax1.twinx()
    goes_classes = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
    for label, level in goes_classes.items():
        ax1.axhline(y=level, color='gray', linestyle=':', alpha=0.5)
    ax1_twin.set_yscale('log')
    ax1_twin.set_yticks(list(goes_classes.values()))
    ax1_twin.set_yticklabels(goes_classes.keys())
    ax1_twin.set_ylabel('GOES class')
    
    # Set ylim for both axes after setting up the twin axis
    ax1.set_ylim(bottom=np.min(flux)*0.7, top = np.max(flux)*2)
    ax1_twin.set_ylim(ax1.get_ylim())
    # Background-subtracted plot
    ax2.plot(background_subtracted.index, background_subtracted, 'g-', 
             label='Background subtracted')
    if flare_time:
        ax2.axvline(x=flare_time, color='k', linestyle='--', label=f'Flare peak time (class {bg_subtracted_class})')
    ax2.set_yscale('log')
    ax2.set_ylabel('Flux (W/m²)')
    ax2.set_title('Background-subtracted 1$-$8 $\mathrm{\AA}$ data')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend()
    
    # Add GOES class markers for background-subtracted plot first
    ax2_twin = ax2.twinx()
    for label, level in goes_classes.items():
        ax2.axhline(y=level, color='gray', linestyle=':', alpha=0.5)
    ax2_twin.set_yscale('log')
    ax2_twin.set_yticks(list(goes_classes.values()))
    ax2_twin.set_yticklabels(goes_classes.keys())
    ax2_twin.set_ylabel('GOES class')
    
    # Set ylim for both axes after setting up the twin axis
    ax2.set_ylim(bottom=5e-9, top = np.max(background_subtracted)*2)
    ax2_twin.set_ylim(ax2.get_ylim())
    # Calculate derivative
    derivative = background_subtracted.diff()
    
    smooth_window = '5min'  # Adjust window size as needed
    # smooth_derivative = derivative.rolling(window=smooth_window, center=True).mean()
    smooth_derivative = derivative
    
    ax3.plot(smooth_derivative.index, (smooth_derivative*1e6), 'r-', label='Derivative')
    if flare_time:
        ax3.axvline(x=flare_time, color='k', linestyle='--', label=f'Flare time')
    # ax3.set_yscale('log')
    ax3.set_ylabel('Flux derivative x$10^{6}$ (W/m²/s)')
    ax3.set_xlabel('Time')
    ax3.set_title('Derivative of background-subtracted data')
    ax3.grid(True, which='both', linestyle='--', alpha=0.5)
    # ax3.set_ylim(-1,3)
    ax3.legend()
    plt.tight_layout()
    
    # Save figure
    output_filename = filename.replace('.pkl', '.png').replace('pickle/','plots/').replace('goes_data',f"newgoes_{bg_subtracted_class}")

    plt.savefig(output_filename)
    plt.close(fig)  # Add this line
    
    save_processed_goes(filename,flux,background,background_subtracted,derivative, bg_subtracted_class)

    return flux, background, background_subtracted, peak_class, bg_subtracted_class

def extract_flare_class(filename):
   match = re.search(r'[A-Z]\d+\.\d+', filename)  # Matches any uppercase letter followed by numbers
   if match:
       return match[0]
   return None

def full_filename_extraction(filename):
    peak_time = str(extract_flare_time(filename))
    eis_file_time = str(extract_eis_time(filename))
    flare_class = extract_flare_class(filename)
    return peak_time,eis_file_time,flare_class

def save_processed_goes(filename,flux,background,background_subtracted,derivative,bg_subtracted_class):
    processed_data = {
       'flux': flux,
       'background': background, 
       'background_subtracted': background_subtracted,
       'derivative': derivative  # Add derivative to the dictionary
    }
    
    output_filename = filename.replace('goes_data.pkl', 'processed_goes_data.pkl').replace('pickle/','subtracted/').replace('goes_data',f"newgoes_{bg_subtracted_class}")
    with open(output_filename, 'wb') as f:
       pickle.dump(processed_data, f)

def process_single_file(filename, master_df):
    """Process a single GOES file."""
    try:
        with open(filename, 'rb') as f:
            goes_ts = pickle.load(f)
        
        flux, background, subtracted, peak_class, bg_subtracted_class = plot_goes_data(
            goes_ts, 
            filename, 
            master_df  # Pass master_df here
        )
        peak_time, eis_file_time, flare_class = full_filename_extraction(filename)
        
        return {
            'filename': filename,
            'eis_file_time': eis_file_time,
            'flare_class': flare_class,
            'peak_time': peak_time,
            'bg_subtracted_class': bg_subtracted_class
        }
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


if __name__ == "__main__":
    pickle_files = sorted(glob.glob('data_goes/pickle/*.pkl'))
    df_name = "flare_dfs/master_aligned_filtered_flares_v5.csv"
    master_df = pd.read_csv(df_name)
    master_df['GOES subtracted class'] = master_df['GOES subtracted class'].astype('object')
    
    # Parallel processing with progress bar
    num_cores = mp.cpu_count() - 1
    with mp.Pool(num_cores) as pool:
        results = list(tqdm(
            pool.imap(partial(process_single_file, master_df=master_df), pickle_files),
            total=len(pickle_files),
            desc="Processing GOES files",
            unit="file"
        ))
    
    # Update master_df with results (same as before)
    results = [r for r in results if r is not None]
    
    # Add progress bar for DataFrame updates
    for result in tqdm(results, desc="Updating master DataFrame"):
        mask = ((master_df['eis_file_time'] == result['eis_file_time']) & 
                (master_df['GOES class'] == result['flare_class']) & 
                (master_df['hek_flare_peak'] == result['peak_time']))
        master_df.loc[mask, 'GOES subtracted class'] = result['bg_subtracted_class']
    
    master_df.to_csv("flare_dfs/master_aligned_filtered_flares_v6.csv", index=False)