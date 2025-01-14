import numpy as np
import astropy.units as u
from sunpy.net import Fido, attrs as a
import sunpy.map
from datetime import datetime, timedelta
from useful_packages.ashaia import ashaia

def get_closest_aia(date_time_obj, wavelength = 193):
    # Add this line near the top of the file, after the imports
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    """
    Get the closest AIA image to a given datetime for a specified wavelength.

    This function searches for and downloads the AIA image closest to the input datetime,
    within a 1-minute window (±30 seconds) for the specified wavelength.

    Parameters:
    -----------
    date_time_obj : datetime.datetime
        The target datetime for which to find the closest AIA image.
    wavelength : int, optional
        The desired wavelength of the AIA image in Angstroms. Default is 193.

    Returns:
    --------
    sunpy.map.sources.sdo.AIAMap
        A SunPy Map object containing the closest AIA image for the specified wavelength.

    Notes:
    ------
    - Uses Fido to search and fetch AIA data.
    - The search window is set to ±30 seconds around the input datetime.
    - Assumes DATE_FORMAT and necessary imports (e.g., Fido, attrs) are defined elsewhere in the code.
    """
    aia_start = date_time_obj - timedelta(minutes=0.5)
    aia_end = date_time_obj + timedelta(minutes=0.5)
    aia_start_str = aia_start.strftime(DATE_FORMAT)[:-3]
    aia_end_str = aia_end.strftime(DATE_FORMAT)[:-3]
    
    aia_res = Fido.search(a.Time(aia_start_str, aia_end_str), a.Instrument.aia, a.Wavelength(wavelength*u.angstrom))[0][0]
    aia_downloads = Fido.fetch(aia_res, path="./{instrument}/{file}")
    aia_map_plot = sunpy.map.Map(aia_downloads[0])
    return aia_map_plot

def get_closest_hmi_magnetogram(date_time_obj):
    """
    Get the closest HMI magnetogram to a given datetime.

    This function searches for and downloads the HMI magnetogram closest to the input datetime,
    within a 1-minute window (±30 seconds).

    Parameters:
    -----------
    date_time_obj : datetime.datetime
        The target datetime for which to find the closest HMI magnetogram.

    Returns:
    --------
    sunpy.map.sources.sdo.HMIMap
        A SunPy Map object containing the closest HMI magnetogram.

    Notes:
    ------
    - Uses Fido to search and fetch HMI data.
    - The search window is set to ±30 seconds around the input datetime.
    - Assumes DATE_FORMAT and necessary imports (e.g., Fido, attrs) are defined elsewhere in the code.
    """

    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    hmi_start = date_time_obj - timedelta(minutes=0.5)
    hmi_end = date_time_obj + timedelta(minutes=0.5)
    hmi_start_str = hmi_start.strftime(DATE_FORMAT)[:-3]
    hmi_end_str = hmi_end.strftime(DATE_FORMAT)[:-3]
    
    hmi_res = Fido.search(a.Time(hmi_start_str, hmi_end_str), 
                          a.Instrument.hmi, 
                          a.Physobs.los_magnetic_field)[0][0]
    hmi_downloads = Fido.fetch(hmi_res, path="./{instrument}/{file}")
    hmi_map = sunpy.map.Map(hmi_downloads[0])
    return hmi_map

def get_prepped_aia_maps(date_time_obj, wavelengths = [94, 131, 171, 193, 211, 335, 1700]):
    """
    Download AIA images for multiple wavelengths and return them as maps.

    This function downloads AIA images for the wavelengths [94Å, 131Å, 171Å, 193Å, 211Å, 335Å]
    closest to the given datetime and returns them as a dictionary of SunPy Map objects.

    Parameters:
    -----------
    date_time_obj : datetime.datetime
        The target datetime for which to find the closest AIA images.

    Returns:
    --------
    dict
        A dictionary where keys are wavelengths (as integers) and values are corresponding
        sunpy.map.sources.sdo.AIAMap objects.

    Notes:
    ------
    - Uses the get_closest_aia function to fetch individual wavelength images.
    - The wavelengths downloaded are [94, 131, 171, 193, 211, 335] Angstroms.
    """
    import os
    import pickle

    if not os.path.exists('./data_aia'):
        os.makedirs('./data_aia')
    
    aia_maps = {}
    prepped_file_path = f'./data_aia/aia_prepped_{date_time_obj.strftime("%Y%m%d_%H%M%S")}.pickle'
    
    # check if prepped file exists
    if not os.path.exists(prepped_file_path):
        file_num = 0

        # check if all wavelengths exist
        for i, wavelength in enumerate(wavelengths):
            file_path = f'./data_aia/aia_{wavelength}_{date_time_obj.strftime("%Y%m%d_%H%M%S")}.fits'
            if os.path.exists(file_path):
                print(f'{wavelength} exists')
                # aia_maps[wavelength] = sunpy.map.Map(file_path)
                file_num += 1

        # if all wavelengths exist, load them
        if file_num == len(wavelengths):
            for i, wavelength in enumerate(wavelengths):
                aia_maps[wavelength] = sunpy.map.Map(file_path)
        # if not all wavelengths exist, download them
        else:
            aia_search = Fido.search(a.Time(date_time_obj - timedelta(minutes=0.2), date_time_obj + timedelta(minutes=0.2)),
                a.Instrument.aia,
                a.AttrOr([a.Wavelength(wav) for wav in wavelengths*u.angstrom]),
                a.Sample(1*u.minute))  
            aia_downloads = Fido.fetch(aia_search, path="./{instrument}/{file}")
            aia_maps_list = sunpy.map.Map(aia_downloads, sequence=True)

            # save the downloaded maps
            for num, downloaded_map in enumerate(aia_maps_list):
                prepped_map = ashaia(downloaded_map).aiaprep(normalise=True)
                prepped_map.save(file_path, overwrite=True)
                aia_maps[int(prepped_map.wavelength.value)] = prepped_map

        # get hmi magnetogram
        hmi_map = ashaia(get_closest_hmi_magnetogram(date_time_obj)).aiaprep(normalise=True)
        aia_maps['hmi'] = hmi_map
        # Save aia_maps into a pickle file
        with open(prepped_file_path, 'wb') as f:
            pickle.dump(aia_maps, f)

        print(f"AIA maps have been saved to {prepped_file_path}")

    else:
        with open(prepped_file_path, 'rb') as f:
            aia_maps = pickle.load(f)
        print(f"AIA maps have been loaded from {prepped_file_path}")

    return aia_maps
