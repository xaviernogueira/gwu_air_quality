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
NO2_AND_LATLONG = DIR + '\\Tropomi_NO2_latlon_griddedon0.01grid_2019_QA75.ncf'
ERA5 = r'C:\Users\xrnogueira\Documents\Data\ERA5\adaptor.mars.internal-1636309996.9508529-3992-17-5d68984c-35e3-4010-9da7-aaf52d0d05a6.nc'
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


def add_coors(no2_file, latlong_file):
    """
    Combines TROPOMI NetCDF with a lat/long NetCDF that can be converted to a GeoTIFF
    :param no2_file: netCDF with arbitrary variables
    :param latlong_file: netCDF with lat long variables
    :return: new netCDF file with no2_file variables and lat long variables
    """
    # create a new netCDF
    out_cdf_name = str(no2_file).replace('.ncf', '_latlon.ncf')
    in_no2 = nc.Dataset(no2_file)
    in_coors = nc.Dataset(latlong_file)
    out_cdf = nc.Dataset(out_cdf_name, "w")

    # copy no2 attributes and dimensions via dictionary
    out_cdf.setncatts(in_no2.__dict__)
    for name, dimension in in_no2.dimensions.items():
        out_cdf.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

    # copy all no2 data and lat long
    for name, variable in in_no2.variables.items():
        dims = variable.dimensions
        out_cdf.createVariable(name, variable.datatype, dims)
        out_cdf[name][:] = in_no2[name][:]

        # copy variable attributes all at once via dictionary
        out_cdf[name].setncatts(in_no2[name].__dict__)

    # repeat for lat long netCDF
    for name, variable in in_coors.variables.items():
        out_cdf.createVariable(name, variable.datatype, variable.dimensions)
        out_cdf[name][:] = in_coors[name][:]

        # copy variable attributes all at once via dictionary
        out_cdf[name].setncatts(in_coors[name].__dict__)

    return out_cdf_name


def convert_raster(in_raster, out_folder='', out_form=['GTiff', '.tif']):
    """
    :param in_raster: An input raster that is not the same format as the out_form
    :param out_folder: optional, allows output folder to be specified and created. Default is same folder as input.
    :param out_form: A list storing the output format gdal name [0] and the extension [1]. default it GeoTIFF.
    :return: path of created GeoTIFF file (or other format)
    """
    import osgeo
    from osgeo import gdal

    # Open existing dataset
    in_file = gdal.Open(in_raster)

    if out_folder == '':
        out_folder = os.path.dirname(in_raster)

    no_ext = os.path.splitext(os.path.basename(in_raster))[0]
    out_dir = out_folder + '\\%s' % no_ext + out_form[1]

    # Ensure number of bands in GeoTiff will be same as in raster file.
    bands = []  # Set up array for gdal.Translate().
    if in_file is not None:
        band_num = in_file.RasterCount  # Get band count
    else:
        return print('No bands detected')
    for i in range(band_num + 1):  # Update array based on band count
        if i == 0:  # gdal starts band counts at 1, not 0 like the Python for loop does.
            pass
        else:
            bands.append(i)

    # Output to new format using gdal.Translate. See https://gdal.org/python/ for osgeo.gdal.Translate options.
    out_file = gdal.Translate(out_dir, in_file, format=out_form[0], bandList=bands)

    # Properly close the datasets to flush to disk
    in_file = None
    out_file = None

    return out_dir


def extract_vals_from_cdf(stations_csv, monthly_dir, latlong_file, year=2019):
    """
    This function extracts values using array indexing from a netCDF file at specified lat/long points.
    :param stations_csv: A list of station observations with a month column and lat/long values
    :param monthly_dir: Folder holding monthly t netCDF files as values encoded like ..._092019_...ncf (for september)
    :param lat_long_file: netCDF with a variable LAT and LONG
    :param year: year of netCDF files (2019 is default), used for path string slicing
    :return: a csv containing 2019 TROPOMI values assocaited with each observation
    """
    # import stations data
    import pandas as pd
    in_df = pd.read_csv(stations_csv)

    # list .ncf files in monthly_dir and build dictionary w/ month codes
    in_list = os.listdir(monthly_dir)
    in_list = [i for i in in_list if i[-3:] == 'ncf']

    month_dict = {}
    for m in list(range(1, 13)):
        code = "{0:0=2d}".format(m)
        temp_list = [i for i in in_list if '%s%s' % (code, year) in i]
        if len(temp_list) == 1:
            month_dict[m] = temp_list[0]
        else:
            print('WARNING: %s netCDF files with %s - %s code. See list below...' % (len(temp_list), code, year))
            print(temp_list)
            var = input('Select index (i.e., 0, 1,...) of the desired file for month %s' % code)
            month_dict[m] = temp_list[int(var)]

    # get lat, long and no2 values as arrays
    in_coors = nc.Dataset(latlong_file)
    longs = in_coors['LON'][:]
    lats = in_coors['LAT'][:]

    # define function to find nearest array values index
    def find_nearest(array, value):
        array = np.asarray(array)
        index = (np.abs(array - value)).argmin()
        value = array[index]
        return index

    # add the associated (month, location) TROPOMI values to a list, add the list as a column
    tropomi_data = []

    rows = in_df.shape[0]
    for index, row in in_df.iterrows():
        lat_ind = find_nearest(lats, float(row['lat']))
        lon_ind = find_nearest(longs, float(row['long']))

        # grab the ncf for the month and add NO2 values to list
        tropi_dir = monthly_dir + '\\%s' % month_dict[int(row['month'])]
        tropi_array = nc.Dataset(tropi_dir)['NO2'][:]
        tropomi_data.append(tropi_array[lat_ind][lon_ind])

        if index % 100 == 0:
            percent = round((index / rows) * 100, 2)
            print('%s percent finished...' % percent)

    in_df['tropomi'] = np.array(tropomi_data)
    in_df.to_csv(stations_csv.replace('.csv', '_wtropomi.csv'))
    return print(tropomi_data)




################################################################
# run functions

#no2_plotting(TROP_2019, LATLONG, std=False, place='los angeles')



# make a list of rasters

netcdf_months = r'C:\Users\xrnogueira\Documents\Data\NO2_tropomi\by_month'
no2_stations_daily = r'C:\Users\xrnogueira\Documents\Data\NO2_stations\clean_no2_daily_2019.csv'
cdf_files = os.listdir(netcdf_months)
inputs = [netcdf_months + '\\' + i for i in cdf_files]


#################################################
def main():
    ncf_metadata(ERA5)
    #convert_raster(NO2_AND_LATLONG, out_folder='', out_form=['GTiff', '.tif'])
    #extract_vals_from_cdf(no2_stations_daily, netcdf_months, LATLONG, year=2019)

if __name__ == "__main__":
    main()






