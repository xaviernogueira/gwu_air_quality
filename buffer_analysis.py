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


def tableToCSV(input_table, csv_filepath, fld_to_remove_override=None, keep_fields=None):
    """Returns the file path of a csv containing the attributes table of a shapefile or other table"""

    # control which fields are exported
    if keep_fields is None:
        keep_fields = []

    if fld_to_remove_override is None:
        fld_to_remove_override = ['OBJECTID', 'SHAPE']

    fld_list = arcpy.ListFields(input_table)
    fld_names = [str(fld.name) for fld in fld_list]

    # Either delete specified fields, or only keep specified fields
    if len(fld_to_remove_override) > 0:
        for field in fld_to_remove_override:
            try:
                fld_names.remove(field)
            except:
                "Can't delete field: %s" % field

    elif len(keep_fields) > 0:
        fld_names = [i for i in fld_names if i in keep_fields]

    with open(csv_filepath, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(fld_names)
        with arcpy.da.SearchCursor(input_table, fld_names) as cursor:
            for row in cursor:
                writer.writerow(row)
        print(csv_filepath + " CREATED")
    csv_file.close()

    return csv_filepath


def make_vars_dict(var_names, gis_file, methods):
    """Converts a list of variable names, raster addresses, and buffer methods into a dictionary"""

    # test for equal list lengths
    if len(var_names) != len(gis_file) or len(gis_file) != len(methods):
        return print('Input list lengths are not the same')

    # make dictionary
    vars_dict = {}

    for i, var in var_names:
        vars_dict[var] = []
        vars_dict[var].append(gis_file[i])
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
        arcpy.analysis.Buffer(points, out_name, buffer_distance_or_field='%s Meters' % d)
        buff_dict[d] = out_name

    return buff_dict


def buffer_regression(vars_dict, buff_dict, no2_csv, fit_vars=[]):
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
    :return: A list containing vars_dict[0] but with a dataframe data columns as the [2] item. List of buffer dists [1].
    """
    # create folder for exta files and list to store their paths
    del_files = []
    temp_files = os.path.dirname(no2_csv) + '\\temp_files'
    if not os.path.exists(temp_files):
        os.makedirs(temp_files)

    # iterate over buffer variables stored in the dictionary
    buffer_distances = list(buff_dict.keys())
    for var in list(vars_dict.keys()):
        data = var[0]
        method = var[1]

        # pull in annual averaged no2 csv
        no2_df = pd.read_csv(no2_csv)
        no2_df.sort_values(by=['station_id'], inplace=True)

        for dist in buffer_distances:
            col_head = '%s_%s' % (var, dist)
            buffer = buff_dict[dist]

            # if a line shapefile, calculate length
            if method == 'length_sum':
                inter_loc = temp_files + '\\inter_%s.shp' % dist
                dissolved = inter_loc.replace('inter', 'dissolve')
                d_table = temp_files + '\\lengths_%s.csv' % dist

                arcpy.Intersect_analysis([buffer, data], inter_loc, output_type='LINE')
                arcpy.Dissolve_management(inter_loc, dissolved, dissolve_field=['station_id'])
                tableToCSV(dissolved, d_table)
                t_df = pd.read_csv(d_table)
                t_df.sort_values(by=['station_id'], inplace=True)
                no2_df[col_head] = t_df['Shape_Length'].to_numpy()

                # gather files for deletion
                del_files.append(inter_loc)
                del_files.append(d_table)

            # if a raster file, calculate zonal sum
            elif method == 'zonal_sum':
                temp_table = temp_files + '\\zonal_%s.dbf' % dist
                d_table = temp_files + '\\sums_%s.csv' % dist

                arcpy.sa.ZonalStatisticsAsTable(buffer, "station_id", data, out_table=temp_table, statistics_type="SUM")
                tableToCSV(temp_table, d_table)
                t_df = pd.read_csv(d_table)
                t_df.sort_values(by=['station_id'], inplace=True)
                no2_df[col_head] = t_df['SUM'].to_numpy()

                # gather files for deletion
                del_files.append(temp_table)
                del_files.append(d_table)

            else:
                print('Method %s is not valid, choose from [length_sum, zonal_sum]' % method)

        # add the no2 table with the pulled values to the variable dictionary
        var.append(no2_df)

    # delete extra files
    for file in del_files:
        try:
            arcpy.management.Delete(file)
        except:
            print('Could not delete %s' % file)

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
    for var in list(vars_dict.keys()):
        sub_list = []
        methods = var[1]
        no2_df = var[2]
        legend_labels.append('%s_%s' % (var, methods))

        for dist in buffer_distances:
            col_head = '%s_%s' % (var, dist)
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

    plt.show()



