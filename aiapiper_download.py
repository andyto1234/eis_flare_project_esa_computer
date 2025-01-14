import pandas as pd
from useful_packages.useful_sdo import get_closest_aia
from aiapiper.tools import PipeFix
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import sys
import os
import time

if __name__ == "__main__":
    max_attempts = 3  # Number of full runs through the dataset
    attempt = 1

    while attempt <= max_attempts:
        try:
            print(f"\nAttempt {attempt} of {max_attempts}")
            df = pd.read_csv('dataframes/s6_filtered_flares_in_fov_AIAEIS_shifted_with_class.csv')
            df_filtered=df[pd.to_datetime(df['eis_end_time']) > pd.to_datetime(df['flare_start'])].reset_in dex(drop=True)
            print(f'{len(df_filtered["eis_end_time"])} EIS studies in EIS FOV')
            print(f'{len(df_filtered["eis_end_time"].unique())} Unique EIS studies in EIS FOV')
            print(f'{len(df_filtered[["goes_class", "flare_start"]].drop_duplicates())} Unique GOES flare')
            print(f'C class flares: {len(df_filtered[df_filtered["goes_class"].str.startswith("C")]["goes_class"])}')
            print(f'M class flares: {len(df_filtered[df_filtered["goes_class"].str.startswith("M")]["goes_class"])}') 
            print(f'X class flares: {len(df_filtered[df_filtered["goes_class"].str.startswith("X")]["goes_class"])}')

            eis_df = df_filtered[['eis_start_time', 'eis_end_time']].drop_duplicates().reset_index(drop=True)

            # Redirect stdout to devnull to suppress prints from get_closest_aia
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

            def process_row(row, downloader):
                try:
                    flare_peak_dt = pd.to_datetime(row["eis_start_time"])
                    return get_closest_aia(flare_peak_dt, wavelength=193, route='aiapiper', path=r'E:\eisflareproject\aia\193')
                except Exception as e:
                    print(f"Error processing row: {e}")
                    return None

            downloader = PipeFix()
            process_row_with_downloader = partial(process_row, downloader=downloader)

            # Process in batches of 5
            batch_size = 10
            results = []
            with tqdm(total=len(eis_df), desc=f"Processing EIS entries (Attempt {attempt})") as pbar:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    try:
                        # Submit all tasks at once
                        future_to_row = {
                            executor.submit(process_row_with_downloader, row): row 
                            for _, row in eis_df.iterrows()
                        }
                        
                        # Process results as they complete
                        for future in as_completed(future_to_row):
                            try:
                                result = future.result()
                                if result is not None:
                                    results.append(result)
                                pbar.update(1)
                            except Exception as e:
                                print(f"Error processing future: {e}")
                                continue
                    except KeyboardInterrupt:
                        print("\nKeyboard interrupt detected. Shutting down gracefully...")
                        executor.shutdown(wait=False)
                        sys.exit(0)
                    except Exception as e:
                        print(f"Error in execution: {e}")

            attempt += 1
            time.sleep(2)  # Brief pause between attempts

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Major error in attempt {attempt}: {e}")
            attempt += 1
            time.sleep(5)  # Longer pause after an error