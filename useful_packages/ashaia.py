import sunkit_image.enhance as enhance
import scipy.ndimage as ndimage
from aiapy.calibrate import register, update_pointing
import aiapy.psf
from aiapy.calibrate import degradation
import sunpy
from astropy.visualization import ImageNormalize, quantity_support
import numpy as np
from pathlib import Path
# from asheis import load_plotting_routine, load_axes_labels
from matplotlib import pyplot as plt
from datetime import datetime
from astropy.coordinates import SkyCoord
import warnings
warnings.filterwarnings('ignore')
import pickle


def directory_setup_aia(amap):
    date = datetime.strptime(amap.meta['date-obs'],"%Y-%m-%dT%H:%M:%S.%f")
    wvln = amap.meta['wavelnth']
    fullpath = f'images/mgn/{date.year}_{date.month:02d}_{date.day:02d}/{wvln}/fits/'
    filename = f'mgn_{wvln}A_{date.year}_{date.month:02d}_{date.day:02d}__{date.hour:02d}_{date.minute:02d}_{date.second:02d}'
    Path(fullpath).mkdir(parents=True, exist_ok=True)
    return fullpath, filename

class ashaia:
    def __init__(self, filename):
        self.filename = filename
        self.map = sunpy.map.Map(filename)

    def plot_map(self, date, amap, colorbar=False, savefig=True, **kwargs):
        # load_plotting_routine()
        amap.plot()
        if colorbar==True: plt.colorbar() 
        load_axes_labels()
        # plt.savefig(f'{date}/eis_{m.measurement.lower().replace(" ","_").replace(".","_")}.png')
        if savefig==True: plt.savefig(f'images/{amap.measurement.lower().split()[-1]}/eis_{date}_{amap.measurement.lower().replace(" ","_").replace(".","_")}.png')
        # plt.savefig(f'images/{amap.measurement.lower().split()[-1]}/eis_{date}_{amap.measurement.lower().replace(" ","_").replace(".","_")}.png')

    def aiaprep(self, normalise=None):
        print('Prepping...')


        if self.map.detector == 'AIA':
            if self.map.wavelength.value in [94.0, 131.0, 171.0, 193.0, 211.0, 304.0, 335.0]:
                try:
                    with open('demregpy/psf_dict.pkl', 'rb') as f:
                        psf_dict = pickle.load(f)
                    psf_wavelength = psf_dict[self.map.wavelength.value]
                    print(f"PSF dictionary found for wavelength {self.map.wavelength.value}")
                except:
                    print(f"PSF dictionary not found for wavelength {self.map.wavelength.value}, Calculating PSF")
                    psf_wavelength = aiapy.psf.psf(self.map.wavelength)                    
                aia_map_deconvolved = aiapy.psf.deconvolve(self.map, psf=psf_wavelength)
            else:
                aia_map_deconvolved = self.map
            prep_map = update_pointing(aia_map_deconvolved)
            prep_map = register(prep_map)
            # if normalise is not None:
            correction_factor = degradation(self.map.wavelength, self.map.date)
            prep_map = prep_map/prep_map.exposure_time/correction_factor
        else:
            prep_map = register(self.map)            

        print('Prepped!')
        return prep_map
    
    def euv_super_res(self, bottom_left=None, top_right=None, savefile = False):
        passband = str(self.map.wavelength).split(" ")[0].split(".")[0]
        h_param = {
            '171': 0.915,
            '193': 0.94,
            '211': 0.94,
            '131': 0.915,
            '94': 0.94,
            '304': 0.95,
            '335': 0.8,
            '1600': 0.9,
            '1700': 0.9,
            '4500': 0.9,
            '6173': 0.8
        }
        smth = 2 if h_param[passband] != '94' else 4
        # m_pre_mgn = self.aiaprep(normalise=True)

        if bottom_left != None:
            m_pre_mgn = self.map.submap(SkyCoord(*bottom_left, frame=self.map.coordinate_frame),
                      top_right=SkyCoord(*top_right, frame=self.map.coordinate_frame))
        else:
            m_pre_mgn = self.map

        # m_pre_mgn = sunpy.map.Map(m_pre_mgn.data / m_pre_mgn.exposure_time, m_pre_mgn.meta)
        # m_pre_mgn = normalize_exposure(m_pre_mgn)
        m_pre_mgn = m_pre_mgn/m_pre_mgn.exposure_time

        # print('normalize_exposure done')
        m_mgn_data = enhance.mgn(m_pre_mgn.data.astype(float), h=h_param[passband], sigma=[2.5,5,10,20,40])
        m_mgn_data = ndimage.gaussian_filter(m_mgn_data, sigma=(smth), order=0)
        m_mgn = sunpy.map.Map(m_mgn_data, m_pre_mgn.meta)
        # m_mgn.plot_settings['norm'] = ImageNormalize(vmin=0.07,vmax=1.07)
        if savefile != False:
            fullpath, filename = directory_setup_aia(m_mgn)
            m_mgn.save(f'{fullpath}{filename}.fits', overwrite=True)
        return m_mgn

