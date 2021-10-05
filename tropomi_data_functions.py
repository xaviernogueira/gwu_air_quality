"""
Python 3 functions for reading and plotting TROPOMI Air Quality data

@author: xrnogueira
"""
# import dependencies
import os
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader

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
    print('METADATA text file @ %s' % txt_dir)
    return txt_dir







