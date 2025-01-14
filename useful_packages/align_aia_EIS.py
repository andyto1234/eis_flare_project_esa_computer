import sunpy.map 
from sunpy.net import Fido, attrs as a
from astropy import units as u
from tqdm import tqdm
from sunkit_image.coalignment import _calculate_shift as calculate_shift


def alignment(eis_fit, return_shift=False, wavelength = 193*u.angstrom, return_aia=False):
    """
    Aligns the EIS map with the AIA map by calculating the shift in coordinates and applying the shift to the EIS map. Cross correlation

    Parameters:
    eis_fit (str): The file path of the EIS map FITS file.

    Returns:
    sunpy.map.Map: The aligned EIS map.
    """
    
    # Load the EIS map
    if type(eis_fit) == str:
        eis_map_int = sunpy.map.Map(eis_fit)
    else:
        eis_map_int = eis_fit  
    
    # Search for AIA map within a specific time range and wavelength
    aia_result = Fido.search(a.Time(eis_map_int.date-5*u.second, eis_map_int.date+10*u.second),
                             a.Instrument('AIA'), a.Wavelength(wavelength), a.Sample(1*u.minute))
    
    # Fetch the AIA map and save it to a temporary directory
    aia_map = sunpy.map.Map(Fido.fetch(aia_result, path='./tmp/', overwrite=False)[0])
    # Calculate the resampling factors for aligning the maps
    n_x = (aia_map.scale.axis1 * aia_map.dimensions.x) / eis_map_int.scale.axis1
    n_y = (aia_map.scale.axis2 * aia_map.dimensions.y) / eis_map_int.scale.axis2
    
    # Resample the AIA map
    aia_map_r = aia_map.resample(u.Quantity([n_x, n_y]))
    
    # Calculate the shift in coordinates between the AIA and EIS maps
    yshift, xshift = calculate_shift(aia_map_r.data, eis_map_int.data)

    # Convert the shift in coordinates to world coordinates
    reference_coord = aia_map_r.pixel_to_world(xshift, yshift)
    Txshift = reference_coord.Tx - eis_map_int.bottom_left_coord.Tx
    Tyshift = reference_coord.Ty - eis_map_int.bottom_left_coord.Ty
    
    # Print the date and shift values for debugging
    print(eis_map_int.date)

    # Check if the shift is within a certain range
    if (abs(Tyshift/u.arcsec) < 150) and (abs(Txshift/u.arcsec) < 150):
        # Shift the EIS map
        m_eis_fixed = eis_map_int.shift_reference_coord(Txshift, Tyshift)
        print(f'shifted - Tx:{Txshift}, Ty:{Tyshift}')
    else:
        # Keep the EIS map unchanged
        m_eis_fixed = eis_map_int
        print(f'not shifted - Tx:{Txshift}, Ty:{Tyshift}')

    if return_shift:
        return m_eis_fixed, Txshift, Tyshift
    elif return_aia and return_shift:
        return m_eis_fixed, aia_map, Txshift, Tyshift
    else:
        return m_eis_fixed

# Iterate over the EIS FITS files and align each one with the AIA map
# for num, fit in tqdm(enumerate(eis_dir_int)):
#     alignment(fit)