"""
Python 3 functions for reading and plotting TROPOMI Air Quality data

@author: xrnogueira
"""
# import dependencies
import os
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import netCDF4 as nc
from sklearn.preprocessing import StandardScaler
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import requests
import iso3166

# inputs for running locally
DIR = r'C:\Users\xrnogueira\Documents\Data\NO2_tropomi'
LATLONG = DIR + '\\LatLonGrid.ncf'
TROP_2019 = DIR + '\\Tropomi_NO2_griddedon0p01grid_2019_QA75.ncf'
IN_LIST = [LATLONG, TROP_2019]


def ncf_metadata(ncf_files):
    """
     Parameters: a path (str) or a list of paths (list) to .ncf files
     ----------
     Returns: a text file in the directory of the first .ncf file w/ all input file info
     """
    import os
    import netCDF4 as nc
    print('Input files: %s' % ncf_files)

    # define output dir for text file
    if isinstance(ncf_files, list):
        main_dir = os.path.dirname(ncf_files[0])
        in_list = ncf_files
    elif isinstance(ncf_files, str):
        main_dir = os.path.dirname(ncf_files)
        in_list = [ncf_files]
    elif isinstance():
        return print('ncf_file parameter is not a valid .ncf path of a list of paths')

    # create a new text file
    txt_dir = main_dir + '\\ncf_files_info.txt'
    out_txt = open(txt_dir, 'w+')
    out_txt.write('INPUT FILES METADATA\n----------------\n')

    # format text file with ncf file metadata
    for ncf in in_list:
        out_txt.write('FILE: ' + ncf + '\n')
        ds = nc.Dataset(ncf)
        ds_dict = ds.__dict__
        dims = ds.dimensions
        for key in ds_dict.keys():
            val = ds_dict[key]
            out_txt.write('%s: %s\n' % (key, val))

        # write the number of dimensions, their names, and sizes
        out_txt.write('\n# of dimensions: %s\n' % len(dims))
        for dim in ds.dimensions.values():
            dim_txt = str(dim)
            if 'name' in dim_txt:
                split = dim_txt.split(':')[1]
                out = split.replace('name', 'dimension')[1:]
                out_txt.write(out + '\n')

        # write all variable descriptions
        variables = ds.variables.values()
        out_txt.write('\n# of variables: %s' % len(variables))
        for var in variables:
            var_txt = str(var)
            out = var_txt.split('>')[1]
            out_txt.write('%s\n' % out)
        out_txt.write('\n-------------\n')
    out_txt.close()

    return print('METADATA text file @ %s' % txt_dir)


def get_boundingbox(place, output_as='boundingbox', state_override=False):
    """
    get the bounding box of a country or US state in EPSG4326 given it's name
    based on work by @mattijin (https://github.com/mattijn)

    Parameters:
    place - a name (str) of a country, city, or state in english and lowercase (i.e., beunos aires)
    output_as - either boundingbox' or 'center' (str)
         * 'boundingbox' for [latmin, latmax, lonmin, lonmax]
         * 'center' for [latcenter, loncenter]
    integer - default is False (bool), if True the output list is converted to integers
    state_override - default is False (bool), only make True if mapping a state
    ------------------
    Returns:
    output - a list with coordinates as floats i.e., [[11.777, 53.7253321, -70.2695876, 7.2274985]]
    """
    # create url to pull openstreetmap data
    url_prefix = 'http://nominatim.openstreetmap.org/search?country='

    country_list = [j.lower() for j in iso3166.countries_by_name.keys()]

    if place not in country_list:
        if state_override:
            url_prefix = url_prefix.replace('country=', 'state=')
        else:
            url_prefix = url_prefix.replace('country=', 'city=')

    url = '{0}{1}{2}'.format(url_prefix, place, '&format=json&polygon=0')
    response = requests.get(url).json()[0]

    # parse response to list, convert to integer if desired
    if output_as == 'boundingbox':
        lst = response[output_as]
        coors = [float(i) for i in lst]
        output = [coors[-2], coors[-1], coors[0], coors[1]]

    elif output_as == 'center':
        lst = [response.get(key) for key in ['lat', 'lon']]
        coors = [float(i) for i in lst]
        output = [coors[-1], coors[0]]

    else:
        print('ERROR: output_as parameter must set to either boundingbox or center (str)')
        return

    return output


def no2_plotting(no2_file, latlong_file, std=True, place='', state=False):
    """
     Parameters:
         no2_file - A path (str) of a .ncf file w/ no2 data
         latlong_file - A path (str, optional) to a matching extent .ncf file w/ lat long data
          place - optional, a country, city, or state* place name in lowercase (str) that sets the mapping extent
          state* - default=False, must be set to True if the place name is a state
     ----------
     Returns: an no2 concentration colormap plot w/ the selected extent
    """
    # bring in data
    no2_ncf = nc.Dataset(no2_file)
    no2_raw = no2_ncf.variables['NO2'][:]
    geo_ncf = nc.Dataset(latlong_file)

    lon, lat = geo_ncf.variables['LON'][:], geo_ncf.variables['LAT'][:]

    if std:
        no2 = StandardScaler().fit_transform(no2_raw)
        unit = 'standard deviations'
    else:
        no2 = no2_raw
        unit = 'molecules / cm2'

    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    im = ax.pcolormesh(lon, lat, no2, cmap=cm.coolwarm, shading='auto')

    # add geographic features (may not work unless on cartopy 0.20.0)
    ax.gridlines(proj, draw_labels=True, linestyle='--')
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linestyle=':')

    # set extent based on place parameter
    if place != '':
        bounds = get_boundingbox(place, output_as='boundingbox', state_override=state)
        ax.set_extent(bounds, crs=ccrs.PlateCarree())
        ax.set_title(place.title())
    print(bounds)
    # add color bar
    plt.colorbar(im, location='bottom', label='NO2 concentration (%s)' % unit)

    plt.show()


# run functions
#ncf_metadata(IN_LIST[:-1])
no2_plotting(TROP_2019, LATLONG, std=False, place='los angeles')







