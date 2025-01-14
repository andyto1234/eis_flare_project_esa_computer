from asheis.core import asheis
from asheis.util import download_hdf5
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
if __name__ == '__main__':


    df = pd.read_csv(r'dataframes\s6_filtered_flares_in_fov_AIAEIS_shifted_with_class.csv')
    df = df.drop_duplicates(subset=['python_filename'])

    try:
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing EIS files", unit="file"):
            int_file = f'data_eis_fitted/int/{row["python_filename"].replace(".data.h5", "_int.fits")}'
            if not os.path.exists(int_file):
                path = download_hdf5(row['python_filename'])
                try:
                    m_int = asheis(path).get_intensity('fe_12_195.12',plot=False)
                    m_vel = asheis(path).get_velocity('fe_12_195.12',plot=False)
                    m_width = asheis(path).get_width('fe_12_195.12',plot=False)
                    plt.close('all')
                    m_int.save(int_file)
                    m_vel.save(f'data_eis_fitted/doppler/{row["python_filename"].replace("data.h5", "_vel.fits")}')
                    m_width.save(f'data_eis_fitted/ntv/{row["python_filename"].replace("data.h5", "_ntv.fits")}')
                    data_eis_files = os.listdir('data_eis')
                    for file in data_eis_files:
                        os.remove(os.path.join('data_eis', file))
                except Exception as e:
                    print(f"Error processing {row['python_filename']}: {e}")
            else:
                print(f"Skipping {row['python_filename']} - intensity file already exists")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        raise
    except:
        pass
