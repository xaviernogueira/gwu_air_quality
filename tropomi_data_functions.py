"""
Python 3 functions for reading and plotting TROPOMI Air Quality data

@author: xrnogueira
"""
# import dependencies
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams['hatch.linewidth'] = 0.3

# inputs for running locally
DIR = r'C:\Users\xrnogueira\Documents\Data'
LATLONG = DIR + '\\LatLonGrid.ncf'
TROP_ALL = DIR + '\\Tropomi_NO2_griddedon0p01grid_allyears_QA75.ncf'
TROP_2019 = DIR + '\\Tropomi_NO2_griddedon0p01grid_2019_QA75.ncf'

#
