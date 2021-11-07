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
                "Can't delete field: %s" % field

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


def buffer_regression(vars_dict, buff_dict, no2_csv, fit_vars=None):
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
    if fit_vars is None:
        fit_vars = []

    arcpy.env.overwriteOutput = True
    init_logger(__file__)
    del_files = []
    temp_files = os.path.dirname(no2_csv) + '\\temp_files'
    if not os.path.exists(temp_files):
        os.makedirs(temp_files)

    # pull in annual averaged no2 csv
    no2_df = pd.read_csv(no2_csv)
    no2_df.sort_values(by=['station_id'], inplace=True)

    # iterate over buffer variables stored in the dictionary
    buffer_distances = list(buff_dict.keys())
    for key in list(vars_dict.keys()):
        var = vars_dict[key]
        data = var[0]
        method = var[1]

        # make a data frame for each variable storing values for each buffer
        var_df = no2_df.copy()

        for dist in buffer_distances:
            col_head = '%s_%s' % (key, dist)
            buffer = buff_dict[dist]

            # if a line shapefile, calculate length
            try:
                if method == 'length_sum':
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
                    t_df = pd.read_csv(d_table)
                    t_df.sort_values(by=['station_id'], inplace=True)
                    t_df.rename(columns={'LENGTH': col_head}, inplace=True)

                    # gather files for deletion
                    del_files.append(inter_loc)
                    del_files.append(d_table)

                # if a raster file, calculate zonal sum
                elif method == 'zonal_sum':
                    # MAKE SURE THIS PART WORKS TOO ONCE WE HAVE FB DATA READY
                    temp_table = temp_files + '\\zonal_%s.dbf' % dist
                    d_table = temp_files + '\\sums_%s.csv' % dist

                    arcpy.sa.ZonalStatisticsAsTable(buffer, "station_id", data, out_table=temp_table, statistics_type="SUM")
                    tableToCSV(temp_table, d_table)
                    t_df = pd.read_csv(d_table)
                    t_df.rename(columns={'SUM': col_head}, inplace=True)

                    # gather files for deletion
                    del_files.append(temp_table)
                    del_files.append(d_table)

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

    # delete extra files
    for file in del_files:
        try:
            arcpy.Delete_management(file)
        except arcpy.ExecuteError:
                logging.info('Could not delete %s' % file)

    return vars_dict, buffer_distances


def plot_decay_curves(vars_dict, buffer_distances, out_dir, input_vars=None, input_buffs=None):
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
    plt.gcf()
    out_png = out_dir + '\\buffer_decay_curves.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.cla()


#  ------------- INPUTS ------------------
ROADS_DIR = r'C:\Users\xrnogueira\Documents\Data\Road data'
NO2_DIR = r'C:\Users\xrnogueira\Documents\Data\NO2_stations'
POINTS = NO2_DIR + '\\no2_annual_2019_points.shp'
NO2_CSV = NO2_DIR + '\\no2_annual_2019.csv'
var_names = ['p_roads', 's_roads']
gis_files = [ROADS_DIR + '\\primary_roads.shp', ROADS_DIR + '\\non_primary_roads.shp']
methods = ['length_sum', 'length_sum']


def main():
    vars_dict = make_vars_dict(var_names, gis_files, methods)
    buff_dict = buffer_iters(POINTS, 3100, 100)
    out = buffer_regression(vars_dict, buff_dict, NO2_CSV, fit_vars=None)
    plot_decay_curves(out[0], out[1], out_dir=NO2_DIR, input_vars=None, input_buffs=None)


if __name__ == "__main__":
    main()
