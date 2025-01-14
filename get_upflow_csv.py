import pandas as pd
import requests
import matplotlib.pyplot as plt
import sunpy.map
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import glob
import numpy as np
from scipy import ndimage
from skimage import filters, morphology
import csv
from goes_background_subtract import extract_eis_time
from tqdm import tqdm
import eispac
import multiprocessing as mp
from functools import partial

def find_largest_upflow_region(dopp_eis):
    # Extract data from the EIS map
    data = dopp_eis.data
    
    # Pre-processing: Remove positive velocities and extreme negative velocities
    # data = np.where(data > 0, 0, data)
    # data = np.where(data < -55, 0, data)
    data = np.clip(data, -55, 0)

    # Smoothing: Apply Gaussian filter to reduce noise
    smoothed_data = ndimage.gaussian_filter(data, sigma=1.5, mode='nearest')
    
    # Further noise reduction: Apply median filter
    median_filtered = filters.median(smoothed_data, morphology.disk(3))
    
    # Create binary mask for upflow regions
    upflow_mask = (median_filtered < -10)
    
    # Clean up the mask: Remove small objects and close gaps
    cleaned = morphology.remove_small_objects(upflow_mask, min_size=40)
    cleaned = morphology.closing(cleaned, morphology.disk(3))
    
    # Label connected regions in the cleaned mask
    labeled_mask, num_features = ndimage.label(cleaned)
    
    if num_features > 0:
        # Find the largest connected region
        sizes = ndimage.sum(cleaned, labeled_mask, range(1, num_features + 1))
        largest_feature_label = sizes.argmax() + 1
        largest_feature_mask = labeled_mask == largest_feature_label
        
        # Get the bounding box of the largest region
        y, x = np.where(largest_feature_mask)
        bottom_left_y, bottom_left_x = np.min(y), np.min(x)
        top_right_y, top_right_x = np.max(y), np.max(x)
        
        # Calculate spans
        x_span = top_right_x - bottom_left_x
        y_span = top_right_y - bottom_left_y
        
        # Define shrink factor function
        def get_shrink_factor(span):
            if span <= 8:
                return 1.0  # No shrinking for small spans
            elif span <= 15:
                return 0.9  # 10% shrink for medium spans
            else:
                return max(0.5, 0.8 - (span - 15) * 0.01)  # More aggressive shrinking for larger spans, minimum 50%
        
        # Calculate shrink factors
        shrink_factor_x = get_shrink_factor(x_span)
        shrink_factor_y = get_shrink_factor(y_span)
        
        # Calculate center points
        center_x = (bottom_left_x + top_right_x) / 2
        center_y = (bottom_left_y + top_right_y) / 2
        
        # Apply shrinking
        new_x_span = x_span * shrink_factor_x
        new_y_span = y_span * shrink_factor_y
        
        # Calculate new bounding box
        bottom_left_x = int(center_x - new_x_span / 2)
        bottom_left_y = int(center_y - new_y_span / 2)
        top_right_x = int(center_x + new_x_span / 2)
        top_right_y = int(center_y + new_y_span / 2)
        
        # Convert pixel coordinates to world coordinates
        bottom_left_world = dopp_eis.pixel_to_world((bottom_left_x) * u.pix, (bottom_left_y) * u.pix)
        top_right_world = dopp_eis.pixel_to_world((top_right_x) * u.pix, (top_right_y) * u.pix)
        
        return bottom_left_world, top_right_world
    else:
        print("No significant upflow region found.")
        return None, None

def process_single_file(vel_file, output_dir):
    """Process a single velocity file and return results"""
    try:
        dopp_eis = sunpy.map.Map(vel_file)
        eis_time = str(extract_eis_time(vel_file))
        bottom_left_world, top_right_world = find_largest_upflow_region(dopp_eis)
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection=dopp_eis)
        im = dopp_eis.plot(axes=ax, norm=plt.Normalize(vmin=-15, vmax=15), cmap='RdBu_r')
        
        if bottom_left_world and top_right_world:
            bottom_left_pixel = dopp_eis.world_to_pixel(bottom_left_world)
            top_right_pixel = dopp_eis.world_to_pixel(top_right_world)
            
            coords = SkyCoord(Tx=(bottom_left_world.Tx.value, top_right_world.Tx.value) * u.arcsec, 
                            Ty=(bottom_left_world.Ty.value, top_right_world.Ty.value) * u.arcsec, 
                            frame=dopp_eis.coordinate_frame)
            
            dopp_eis.draw_quadrangle(coords, axes=ax, edgecolor="red", lw=2)
            
            plot_prefix = ""
            bottom_left_x, bottom_left_y = bottom_left_pixel.x.value, bottom_left_pixel.y.value
            top_right_x, top_right_y = top_right_pixel.x.value, top_right_pixel.y.value
        else:
            plot_prefix = "no_upflow_"
            bottom_left_x, bottom_left_y, top_right_x, top_right_y = 0, 0, 0, 0

        plt.colorbar(im, ax=ax, label='Doppler Velocity [km/s]')
        plt.title(f"Doppler Velocity Map - {eis_time}")
        
        # Save plot
        plot_path = os.path.join(output_dir, f"{plot_prefix}doppler_velocity_map_{eis_time}.png")
        plt.savefig(plot_path, dpi=100)
        plt.close(fig)
        
        return {
            'eis_time': eis_time,
            'coords': {
                'bottom_left_x': bottom_left_x,
                'bottom_left_y': bottom_left_y,
                'top_right_x': top_right_x,
                'top_right_y': top_right_y
            }
        }
    
    except Exception as e:
        print(f"Error processing {vel_file}: {str(e)}")
        return None

if __name__ == "__main__":
    # Setup
    plt.switch_backend('agg')  # Non-interactive backend for multiprocessing
    output_dir = "data_eis/aligned/vel_png"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    csv_path = 'flare_dfs/master_aligned_filtered_flares_v6.csv'
    vel_files = sorted(glob.glob('data_eis/aligned/vel_maps/*.fits'))
    df = pd.read_csv(csv_path)
    
    # Initialize new columns
    new_entries = ['bottom_left_x', 'bottom_left_y', 'top_right_x', 'top_right_y']
    for new_entry in new_entries:
        df[new_entry] = np.nan
    
    # Setup multiprocessing
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    process_func = partial(process_single_file, output_dir=output_dir)
    
    # Process files in parallel
    print(f"Processing {len(vel_files)} files using {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, vel_files),
            total=len(vel_files)
        ))
    
    # Update DataFrame with results
    for result in results:
        if result is not None:
            eis_time = result['eis_time']
            coords = result['coords']
            df.loc[df['eis_file_time']==eis_time, new_entries] = [
                coords['bottom_left_x'],
                coords['bottom_left_y'],
                coords['top_right_x'],
                coords['top_right_y']
            ]
    
    # Save results
    results_csv_path = 'flare_dfs/upflow_results_v7.csv'
    df.to_csv(results_csv_path, index=False)
    print("Processing complete!")