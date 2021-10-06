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
DIR = r'C:\Users\xrnogueira\Documents\Data'
LATLONG = DIR + '\\LatLonGrid.ncf'
TROP_ALL = DIR + '\\Tropomi_NO2_griddedon0p01grid_allyears_QA75.ncf'
TROP_2019 = DIR + '\\Tropomi_NO2_griddedon0p01grid_2019_QA75.ncf'
IN_LIST = [LATLONG, TROP_ALL, TROP_2019]


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


def get_boundingbox_country(place, output_as='boundingbox', integer=False):
    """
    get the bounding box of a country or US state in EPSG4326 given it's name
    author @mattijin (https://github.com/mattijn)

    Parameters:
    place - a name (str) of a country or US state in english and lowercase
    output_as - either boundingbox' or 'center' (str)
         * 'boundingbox' for [latmin, latmax, lonmin, lonmax]
         * 'center' for [latcenter, loncenter]
    integer - default is False (bool), if True the output list is converted to integers
    ------------------
    Returns:
    output - a list with coordinates as floats i.e., [[11.777, 53.7253321, -70.2695876, 7.2274985]]
    """
    # create url to pull openstreetmap data
    url_prefix = 'http://nominatim.openstreetmap.org/search?country='

    country_list = [j.lower() for j in iso3166.countries_by_name.keys()]

    if place not in country_list:
        url_prefix = url_prefix.replace('country=', 'state=')

    url = '{0}{1}{2}'.format(url_prefix, place, '&format=json&polygon=0')
    response = requests.get(url).json()[0]

    # parse response to list, convert to integer if desired
    if output_as == 'boundingbox':
        lst = response[output_as]
        output = [float(i) for i in lst]

    elif output_as == 'center':
        lst = [response.get(key) for key in ['lat', 'lon']]
        output = [float(i) for i in lst]
    else:
        print('ERROR: output_as parameter must set to either boundingbox or center (str)')
        return

    if integer:
        output = [int(i) for i in output]

    return output


def no2_plotting(no2_file, latlong_file, std=True, cmap_max=0, extent=[]):
    """
     Parameters:
         no2_file - A path (str) of a .ncf file w/ no2 data
         latlong_file - A path (str, optional) to a matching extent .ncf file w/ lat long data
         cmap_max - optional, the maximum color map value to adjust color stretch
         extent - optional, either a len=4 list w/ [xlims, ylims] or a shapefile from which said values are extracted
         * Note that extent values must be in lat long if a latlong_file is specified *
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
    # ax.gridlines(proj, draw_labels=True, linestyle='--', color='grey')
    plt.colorbar(im, location='bottom', label='NO2 concentration (%s)' % unit)

    # add geographic features (may not work unless on cartopy 0.20.0)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linestyle=':')

    plt.show()


#no2_plotting(TROP_ALL, LATLONG, std=False)
print(get_boundingbox_country('washington', output_as='boundingbox', integer=False))






