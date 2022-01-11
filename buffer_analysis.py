import logging
import os
import csv
import numpy as np
import arcpy
import matplotlib
import matplotlib.pyplot as plt
import arcpy.analysis
import arcpy.da
import pandas as pd
import logging


def init_logger(filename):
    """Initializes logger w/ same name as python file"""

    logging.basicConfig(filename=os.path.basename(filename).replace('.py', '.log'), filemode='w', level=logging.INFO)
    stderr_logger = logging.StreamHandler()
    stderr_logger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.getLogger().addHandler(stderr_logger)

    return


def tableToCSV(input_table, csv_filepath, fld_to_remove_override=None, keep_fields=None):
    """Returns the file path of a csv containing the attributes table of a shapefile or other table"""

    # control which fields are exported
    if keep_fields is None:
        keep_fields = []

    if fld_to_remove_override is None:
        fld_to_remove_override = [] # 'OBJECTID', 'SHAPE'

    fld_list = arcpy.ListFields(input_table)
    fld_names = [str(fld.name) for fld in fld_list]

    # Either delete specified fields, or only keep specified fields
    if len(fld_to_remove_override) > 0:
        for field in fld_to_remove_override:
            try:
                fld_names.remove(field)
            except:
                print("Can't delete field: %s" % field)

    elif len(keep_fields) > 0:
        fld_names = [i for i in fld_names if i in keep_fields]

    with open(csv_filepath, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(fld_names)
        with arcpy.da.SearchCursor(input_table, fld_names) as cursor:
            for row in cursor:
                writer.writerow(row)
    csv_file.close()

    return csv_filepath


def make_vars_dict(var_names, gis_files, methods):
    """
    Converts a list of variable names, raster addresses, and buffer methods into a dictionary"
    :param var_names: list of length N of names for variables
    :param gis_files: list storing N GIS files associated with each variable name
    :param methods: list storing N method codes for each variable (either: length_sum or zonal_sum)
    :return: a dictionary with each variable name as keys holding a list w/ gis file as [0], and method code as [2]
    """

    # test for equal list lengths
    if len(var_names) != len(gis_files) or len(gis_files) != len(methods):
        return print('Input list lengths are not the same')

    # make dictionary
    vars_dict = {}

    for i, var in enumerate(var_names):
        vars_dict[var] = []
        vars_dict[var].append(gis_files[i])
        vars_dict[var].append(methods[i])

    return vars_dict


def buffer_iters(points, max, step):
    """
    points :param is a point shapefile (.shp )w/ unique identifiers
    max :param is a number in meters (float or int) for maximum buffer length
    step :param is a number in meters (float or int) defining buffer length intervals
    """
    dir = os.path.dirname(points)
    temp_files = dir + '\\temp_files'

    if not os.path.exists(temp_files):
        os.makedirs(temp_files)

    buff_dists = list(range(step, max, step))
    buff_dict = {}

    for d in buff_dists:
        out_name = temp_files + '\\points_buffer_%sm.shp' % d
        try:
            arcpy.analysis.Buffer(points, out_name, buffer_distance_or_field='%s Meters' % d)
        except:
            print(out_name + ' has been already made')
        buff_dict[d] = out_name

    return buff_dict


def length_sum(data, buffer, dist, temp_files):
    """
    Takes a shapefile and calculates the associated buffer length sum values with each point's unique ID
    :param data: a shapefile path (.shp)
    :param buffer: a shapefile of buffers with unique point IDs
    :param dist: the buffer distances used (for labeling)
    :param temp_files: a folder to store temporary files
    :return: a buffer shapefile with length sum values [0] and it's attribute table as a .csv [1], and a list of file
    to delete [2]
    """
    inter_loc = temp_files + '\\inter_%s.shp' % dist
    dissolved = inter_loc.replace('inter', 'dissolve')
    pr = arcpy.SpatialReference('NAD 1983 Contiguous USA Albers')
    dissolved_p = dissolved.replace('.shp', '_p.shp')
    d_table = temp_files + '\\lengths_%s.csv' % dist

    arcpy.Intersect_analysis([buffer, data], inter_loc, output_type='LINE')
    arcpy.Dissolve_management(inter_loc, dissolved, dissolve_field=['station_id'])
    arcpy.Project_management(dissolved, dissolved_p, out_coor_system=pr,
                             transform_method='WGS_1984_(ITRF00)_To_NAD_1983')
    arcpy.AddGeometryAttributes_management(dissolved_p, 'LENGTH', Length_Unit='KILOMETERS')
    tableToCSV(dissolved_p, d_table)

    del_files = [inter_loc, d_table]

    return dissolved_p, d_table, del_files


def zonal_buffer(data, buffer, dist, temp_files, method, id='station_id'):
    """
    Takes a .tif raster and calculates the associated buffer zonal SUM or AVERAGE values with each point's unique ID
    :param data: a shapefile path (.shp)
    :param buffer: a shapefile of buffers with unique point IDs
    :param dist: the buffer distances used (for labeling)
    :param temp_files: a folder to store temporary files
    :param method: must be 'SUM' or 'MEAN'
    :return: a table with buffer data and unique point IDs [0], a list of files to delete [1]
    """
    temp_table = temp_files + '\\zonal_%s.dbf' % dist
    d_table = temp_files + '\\sums_%s.csv' % dist

    arcpy.sa.ZonalStatisticsAsTable(buffer, id, data, out_table=temp_table, statistics_type=method)
    tableToCSV(temp_table, d_table)

    del_files = [temp_table, d_table]

    return d_table, del_files


def buffer_regression(vars_dict, buff_dict, no2_csv):
    """
    This functions applies all buffer intervals to all variable in the var_dict according to specified method.
    This outputs a copy of var_dict but with a list w/ R^2 values appended to each variables sub_list.
    Linear regression is applied, the user can also specify to a priori apply a regression model, which calculates
    regression fit between the buffered variable and the residuals left after the previous linear model.
    :param vars_dict: Variable dict. Each variable name is associated with a list containing gis_file and method code.
    :param buff_dict: Buffer dict. Each buffer distance (int) is associated with it's shapefile path.
    :param no2_csv: A csv file containing annual averaged NO2 w/ station IDs matching the point file.
    :param fit_vars: (optional) A list containing variable names that will be used to fit a regression model. If used,
    the out
    :param chosen_buffs: a list w/ equal length as vars_dict.keys() that picks buffers, and saves the resulting csv
    :return: A list containing vars_dict[0] but with a dataframe data columns as the [2] item. List of buffer dists [1].
    """

    # initiate logger and arcpy environment settings
    arcpy.env.overwriteOutput = True
    init_logger(__file__)

    if arcpy.CheckExtension('Spatial') == 'Available':
            arcpy.CheckOutExtension('Spatial')
    else:
        return logging.error('ERROR: Cant check out spatial liscence')

    del_files = []
    dir = os.path.dirname(no2_csv)
    temp_files = dir + '\\temp_files'
    if not os.path.exists(temp_files):
        os.makedirs(temp_files)

    # pull in annual averaged no2 csv
    no2_df = pd.read_csv(no2_csv)
    no2_df.sort_values(by=['station_id'], inplace=True)

    # iterate over buffer variables stored in the dictionary or use chosen buffer distances
    vars_keys = list(vars_dict.keys())
    buffer_distances = list(buff_dict.keys())

    for key in vars_keys:
        var = vars_dict[key]
        data = var[0]
        method = var[1]

        # make a data frame for each variable storing values for each buffer
        var_df = no2_df.copy()

        for dist in buffer_distances:
            col_head = '%s_%s' % (key, dist)
            buffer = buff_dict[dist]
            logging.info('Pulling values for %s meter buffer' % dist)

            # if a line shapefile, calculate length
            try:
                if method == 'length_sum':
                    dissolved_p, d_table, deletes = length_sum(data, buffer, dist, temp_files)

                    t_df = pd.read_csv(d_table)
                    t_df.sort_values(by=['station_id'], inplace=True)
                    t_df.rename(columns={'LENGTH': col_head}, inplace=True)

                    # gather files for deletion
                    for file in deletes:
                        del_files.append(file)

                # if a raster file, calculate zonal sum
                elif method == 'zonal_sum':
                    d_table, deletes = zonal_buffer(data, buffer, dist, temp_files, method)

                    t_df = pd.read_csv(d_table)
                    t_df.rename(columns={'SUM': col_head}, inplace=True)

                    # gather files for deletion
                    for file in deletes:
                        del_files.append(file)

                else:
                    print('Method %s is not valid, choose from [length_sum, zonal_sum]' % method)
                    return

                t_df = t_df[['station_id', col_head]]
                var_df = var_df.merge(t_df, on='station_id', how='left')
                var_df[col_head] = var_df[col_head].fillna(0)

            except arcpy.ExecuteError:
                logging.info(str(arcpy.GetMessages()))

        # add the no2 table with the pulled values to the variable dictionary
        vars_dict[key].append(var_df)
        var_df.to_csv(dir + '\\no2_with_%s_buffs.csv' % key)

    # delete extra files
    for file in del_files:
        try:
            arcpy.Delete_management(file)
        except arcpy.ExecuteError:
                logging.info('Could not delete %s' % file)

    return vars_dict, buffer_distances


def plot_decay_curves(vars_dict, buffer_distances, input_vars=None, input_buffs=None):
    """"""
    if input_buffs is None:
        input_buffs = []
    if input_vars is None:
        input_vars = []
    if not len(input_vars) == len(input_buffs):
        print('ERROR: All input regression variables need an associated input_buffs distance (list lens must be =)')
        return

    # BUILD OPTION TO USE RESIDUAL AFTER A MULTIPLE REGRESSION

    coors_coefs = []
    legend_labels = []
    for key in list(vars_dict.keys()):
        var = vars_dict[key]
        sub_list = []
        methods = var[1]
        no2_df = var[2]
        legend_labels.append('%s_%s' % (key, methods))

        for dist in buffer_distances:
            col_head = '%s_%s' % (key, dist)
            no2 = no2_df['mean_no2']
            values = no2_df[col_head]
            coef = np.corrcoef(no2, values)
            sub_list.append(coef[0][1])

        sub_np = np.array(sub_list)
        coors_coefs.append(sub_np)

    # plot the decay curves
    buffs_np = np.array(buffer_distances)
    for i, curve in enumerate(coors_coefs):
        plt.plot(buffs_np, curve, label=legend_labels[i], marker='.')

    plt.xlabel('Buffer distance (m)')
    plt.ylabel('Correlation coeff.')
    plt.title('Buffer distance vs no2 correlation')

    # Define plotting extent
    plt.xlim(min(buffs_np), max(buffs_np))

    # Format plot
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.xticks(fontsize='x-small')
    plt.yticks(fontsize='x-small')
    plt.legend()

    plt.show()
    plt.cla()


def add_buffer_data_to_no2(no2_csv, buff_folder, var_dict, out_csv=''):
    """
    Joins selected buffer distance data to the full no2 observation data set by using station_id
    :param no2_csv: full no2 observations dataset w/ station_id column
    :param buff_folder: folder containing all buffer distance .csv files used for decay curve plotting
    :param var_dict: a dictionary with variable name keys (str) that are in the .csv file names used for decay curve
    plots, with corresponding list items containing chosen int buffer distances (i.e., var_dict[p_roads] = [1000, 1550])
    :param out_csv: specifies output .csv file name (optional, default is out_csv=no2_csv.replace(.csv, _buff.csv))
    :return: new no2 observations csv
    """
    # set up input data and directories
    no2_df = pd.read_csv(no2_csv)
    files = os.listdir(buff_folder)

    if out_csv == '':
        out_csv = no2_csv.replace('.csv', '_buff.csv')
    else:
        out_csv = out_csv

    # join buffer distance values by station_id to the input no2 observation data
    vars = list(var_dict.keys())
    for var in vars:
        print('Joining variable: %s...' % var)
        # find the appropriate buffer decay curve .csv files
        p_csvs = [i for i in files if var in i]
        if len(p_csvs) == 1:
            b_df = pd.read_csv(buff_folder + '\\%s' % p_csvs[0])
        else:
            return print('ERROR: Multiple .csv files %s have %s in their name. Please leave only one.' % (p_csvs, var))

        # create column headers matching the variable and desired buffer distance
        distances = var_dict[var]
        heads = []
        for d in distances:
            head = '%s_%s' % (var, d)

            sub_df = b_df[['station_id', head]]
            no2_df = no2_df.merge(sub_df, on='station_id', how='left')
            no2_df[head] = no2_df[head].fillna(0)
        print('Done')

    no2_df.to_csv(out_csv)
    print('Output .csv: %s' % out_csv)
    return out_csv


def make_buffer_table(vars_dict, copy_raster, out_folder, method_override=None):
    """
    This function creates a raster of artitrary shape/extent where each cell is assigned a value extract from a buffer
    :param vars_dict: dictionary with variable names (str) as keys storing lists containing the following inputs:
    [*variable layer path* (i.e., .shp or .tif file), *buffer distances to use in meters* ex: [1400, 8000]]
    :param copy_raster: a raster to match the extent and resolution of
    :param out_folder: directory path to store output GeoTIFFs
    :param method_override: (default is None) overrides methods via a list of the same length of vars_dict.keys()
    items must be string 'length_sum', 'point_sum', 'zonal_sum', or 'zonal_mean' (NOT COMPLETED FUNCTIONALITY)
    :return: a list storing the paths to csvs containing buffer values and a unique point id.
    """

    # initiate logger and arcpy environment settings
    arcpy.env.overwriteOutput = True
    init_logger(__file__)
    logging.info('Making buffer variable rasters....')

    if arcpy.CheckExtension('Spatial') == 'Available':
        arcpy.CheckOutExtension('Spatial')
    else:
        return logging.error('ERROR: Cant check out spatial licence')

    # set up folders
    del_files = []

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    temp_files = out_folder + '\\temp_files'
    if not os.path.exists(temp_files):
        os.makedirs(temp_files)

    # make a blank raster of the same extent as the input raster
    in_raster = arcpy.sa.Raster(copy_raster)
    blank_raster = in_raster * 0 + 1
    blank_tif = temp_files + '\\blank.tif'
    blank_raster.save(blank_tif)

    # convert the blank raster to points
    points = temp_files + '\\points.shp'
    arcpy.RasterToPoint_conversion(blank_tif, points)
    logging.info('Blank raster made and converted to points...')

    # make list to store output tables
    # iterate over variables
    out_list = []
    for i, name in enumerate(list(vars_dict.keys())):
        logging.info('Variable: %s' % name)
        data, dists = vars_dict[name]

        # establish method for buffer values
        if method_override is None:
            if data[-4:] == '.shp':
                method = 'length_sum'
            elif data[-4:] == '.tif':
                method = 'zonal_sum'
            else:
                return logging.error('ERROR: data is not a shapefile or a GeoTiff')
        elif isinstance(method_override, list):
            method = method_override[i]
        else:
            return logging.error('ERROR: method_orverride is specified but is not a list! Delete the parameter to return to default)')

        for dist in dists:
            logging.info('Using buffer distance %sm' % dist)
            buffer = temp_files + '\\%s_full_raster_buff.shp' % dist
            arcpy.analysis.Buffer(points, buffer, buffer_distance_or_field='%s Meters' % int(dist))
            error_msg = 'ERROR: method_override was used incorrectly, please use a valid method name'

            if 'zonal' in method:
                if 'sum' in method:
                    meth_str = 'SUM'
                    d_table, deletes = zonal_buffer(data, buffer, dist, out_folder, meth_str, id='pointid')
                elif 'mean' in method:
                    meth_str = 'MEAN'
                    d_table, deletes = zonal_buffer(data, buffer, dist, out_folder, meth_str)
                else:
                    return logging.error(error_msg)

            elif 'length' in method:
                dissolved_p, d_table, deletes = length_sum(data, buffer, dist, out_folder)

            elif 'point' in method:
                logging.info('ADD POINT FUNCTION AFTER TESTING')
                deletes = []

            else:
                return logging.error(error_msg)

            # set up files for deletion and save data tables to an out_list
            for file in deletes:
                del_files.append(file)

            out_list.append(d_table)

        # delete extra files
        for file in del_files:
            try:
                arcpy.Delete_management(file)
            except arcpy.ExecuteError:
                logging.info('Could not delete %s' % file)

    return out_list





#  ------------- INPUTS ------------------
POP_DEN = r'C:\Users\xrnogueira\Documents\Data\usa_rasters_0p001\popden.tif'
NO2_DIR = r'C:\Users\xrnogueira\Documents\Data\NO2_stations'
POINTS = NO2_DIR + '\\no2_annual_2019_points.shp'
NO2_CSV = NO2_DIR + '\\no2_annual_2019.csv'
NO2_DAILY = NO2_DIR + '\\clean_no2_daily_2019.csv'

# set up inputs for buffer analysis functions
road_dir = r'C:\Users\xrnogueira\Documents\Data\USGS_roads'
var_names = ['p_roads', 's_roads']
gis_files = [road_dir + '\\primary_prj.shp', road_dir + '\\secondary_prj.shp']
methods = ['length_sum', length_sum]
dist_dict = {'pod_den': [1100]}


f_vars_dict = {'p_roads': [road_dir + '\\primary_prj.shp', [1000]], 's_roads': [road_dir + '\\secondary_prj.shp', [1700, 3000]]}
copy_raster = r'C:\Users\xrnogueira\Documents\Data\Chicago_prediction\temp_files\blank_chi.tif'
out_folder = r'C:\Users\xrnogueira\Documents\Data\Chicago_prediction'


def main():
    #vars_dict = make_vars_dict(var_names, gis_files, methods)
    #buff_dict = buffer_iters(POINTS, 3100, 100)
    #out = buffer_regression(vars_dict, buff_dict, NO2_CSV)
    #print(out)
    #plot_decay_curves(out[0], out[1], input_vars=None, input_buffs=None)
    #add_buffer_data_to_no2(NO2_DAILY, NO2_DIR, vars_dict, out_csv=NO2_DAILY.replace('.csv', '_roads.csv'))
    #make_buffer_table(f_vars_dict, copy_raster, out_folder, method_override=None)

    buffers = [out_folder + '\\points_%s.shp' % i for i in [1000, 1700, 3000]]
    length_sum(gis_files[0], buffers[0], 1000, out_folder)
    length_sum(gis_files[1], buffers[1], 1700, out_folder)
    length_sum(gis_files[1], buffers[2], 3000, out_folder)


if __name__ == "__main__":
    main()
