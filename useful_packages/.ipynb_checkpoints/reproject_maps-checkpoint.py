import sunpy.map  # Importing the map module from sunpy
from astropy.coordinates import SkyCoord  # Importing SkyCoord from astropy.coordinates
import astropy.units as u  # Importing units from astropy
from sunpy.net import Fido, attrs as a  # Importing Fido and attrs from sunpy.net
from reproject import reproject_interp
from sunpy.coordinates import propagate_with_solar_surface
import numpy as np


def reproject_map(map1, map2, solar_rotation=False):
    # map 2 = reference map
    # print("not propagating with solar surface")
    # with propagate_with_solar_surface():
    if True:
        # Extract the low-level WCS object from map1
        map1_wcs = map1.wcs.low_level_wcs
        map2_wcs = map2.wcs.low_level_wcs

        # Reproject map1 to the FOV of map2
        if not solar_rotation:
            reprojected_data, _ = reproject_interp((map1.data, map1_wcs), map2_wcs)
        else:
            with propagate_with_solar_surface(): 
                reprojected_data, _ =   reproject_interp((map1.data, map1_wcs), map2_wcs)

    # Create a new map object using the reprojected data and metadata from map2
    reprojected_map = sunpy.map.Map(reprojected_data, map2.meta)
    return reprojected_map
