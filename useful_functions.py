import os
import json
import pandas as pd
import eispac
import sqlite3
import astropy
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ImageNormalize, SqrtStretch
from astropy.io import fits

import sunpy.coordinates  # NOQA
import sunpy.map
import sunpy.data.sample  
from sunpy.net import Fido
from sunpy.net import attrs as a
from eispac.net.attrs import FileType
from asheis import asheis
from sunkit_image.coalignment import _calculate_shift as calculate_shift
from datetime import datetime, timedelta
def alignment(eis_fit):
    """
    Aligns the EIS map with the AIA map by calculating the shift in coordinates and applying the shift to the EIS map. Cross correlation

    Parameters:
    eis_fit (str): The file path of the EIS map FITS file.

    Returns:
    sunpy.map.Map: The aligned EIS map.
    """
    
    # Load the EIS map
    eis_map_int = sunpy.map.Map(eis_fit)

    # Search for AIA map within a specific time range and wavelength
    aia_result = Fido.search(a.Time(eis_map_int.date-5*u.second, eis_map_int.date+10*u.second),
                             a.Instrument('AIA'), a.Wavelength(193*u.angstrom), a.Sample(1*u.minute))
    
    # # Fetch the AIA map and save it to a temporary directory
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
    print(Txshift)
    print(Tyshift)
    
    # Check if the shift is within a certain range
    if (abs(Tyshift/u.arcsec) < 150) and (abs(Txshift/u.arcsec) < 150):
        # Shift the EIS map
        m_eis_fixed = eis_map_int.shift_reference_coord(Txshift, Tyshift)
        print('shifted')
    else:
        # Keep the EIS map unchanged
        m_eis_fixed = eis_map_int
        print('not shifted')

    return m_eis_fixed