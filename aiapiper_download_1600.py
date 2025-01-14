import pandas as pd
from useful_packages.useful_sdo import get_closest_aia
from aiapiper.tools import PipeFix
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import sys
import os

if __name__ == "__main__":
    df = pd.read_csv('dataframes/s6_filtered_flares_in_fov_AIAEIS_shifted_with_class.csv')
    df_filtered=df[pd.to_datetime(df['eis_end_time']) > pd.to_datetime(df['flare_start'])].reset_index(drop=True)
    print(f'{len(df_filtered["eis_end_time"])} EIS studies in EIS FOV')
    print(f'{len(df_filtered["eis_end_time"].unique())} Unique EIS studies in EIS FOV')
    print(f'{len(df_filtered[["goes_class", "flare_start"]].drop_duplicates())} Unique GOES flare')
    print(f'C class flares: {len(df_filtered[df_filtered["goes_class"].str.startswith("C")]["goes_class"])}')
    print(f'M class flares: {len(df_filtered[df_filtered["goes_class"].str.startswith("M")]["goes_class"])}') 
    print(f'X class flares: {len(df_filtered[df_filtered["goes_class"].str.startswith("X")]["goes_class"])}')

    unique_flares_df = df_filtered[["goes_class", "flare_start", "flare_peak"]].drop_duplicates()
    unique_flares_df = unique_flares_df.reset_index(drop=True)

    eis_df = df_filtered[['eis_start_time', 'flare_peak', 'eis_end_time']].drop_duplicates().reset_index(drop=True)


    # Redirect stdout to devnull to suppress prints from get_closest_aia
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    def process_row(row, downloader):
        # Get the flare peak time for this specific row
        flare_peak_dt = pd.to_datetime(row["flare_peak"])
        return get_closest_aia(flare_peak_dt, wavelength=1700, route='aiapiper', path=r'E:\eisflareproject\aia\1700')

    downloader = PipeFix()
    process_row_with_downloader = partial(process_row, downloader=downloader)

    # Process in batches of 5
    batch_size = 6
    results = []
    with tqdm(total=len(unique_flares_df), desc="Processing EIS entries") as pbar:
        with ThreadPoolExecutor(max_workers=6) as executor:
            # Submit all tasks at once
            future_to_row = {
                executor.submit(process_row_with_downloader, row): row 
                for _, row in unique_flares_df.iterrows()
            }
            
            # Process results as they complete
            for future in as_completed(future_to_row):
                result = future.result()
                results.append(result)
                pbar.update(1)