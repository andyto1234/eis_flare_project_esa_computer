from astropy.time import Time, TimeDelta
from sunpy.net import Fido, attrs as a
import astropy.units as u
from parfive import Downloader
import tempfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from functools import partial
from datetime import datetime
import pickle
import pandas as pd
import numpy as np

def process_chunk(chunk_df):
    """Process a single chunk of data."""
    year_dates = chunk_df['eis_start_time'].unique()
    
    time_ranges = [
        (Time(time) - TimeDelta(5 * u.second),
         Time(time) + TimeDelta(30 * u.second))
        for time in year_dates
    ]
    
    chunk_results = []
    for start, end in tqdm(time_ranges, desc="Searching Fido", leave=False):
        try:
            result = Fido.search(
                a.Time(start, end),
                a.Instrument('AIA'),
                a.Wavelength(193*u.angstrom),
                a.Sample(1*u.minute)
            )
            chunk_results.append(result)
        except Exception as e:
            print(f"Error processing {start} to {end}: {str(e)}")
            continue
            
    return chunk_results


def main():
    filtered_df = pd.read_csv('dataframes/filtered_flares_in_fov_with_class.csv')
    filtered_df = filtered_df['eis_start_time'].unique()
    filtered_df = pd.DataFrame(filtered_df, columns=['eis_start_time'])
    years = range(2011, 2025)
    
    for year in years:
        # Filter for year
        year_mask = filtered_df['eis_start_time'].str.startswith(str(year))
        year_df = filtered_df[year_mask]
        
        # Split into chunks of 100
        chunks = np.array_split(year_df, len(year_df) // 100 + 1)
        
        print(f"Processing year {year} in {len(chunks)} chunks...")
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            # Process chunk
            chunk_results = process_chunk(chunk)
            
            # Flatten results
            flat_results = [item for sublist in chunk_results for item in sublist]
            
            # Download
            if flat_results:
                result = Fido.fetch(*flat_results, path='eis_flare_list_data/')
                
                for _ in range(6):
                    result = Fido.fetch(result, path='eis_flare_list_data/')
                
                print(f"Chunk {i+1} complete: {len(flat_results)} results")

if __name__ == '__main__':
    main()