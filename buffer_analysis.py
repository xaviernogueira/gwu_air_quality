import logging
import os

import arcpy.analysis

def make_vars_dict(var_names, rasters, methods):
    """Converts a list of variable names, raster addresses, and buffer methods into a dictionary"""

    # test for equal list lengths
    if len(var_names) != len(rasters) or len(rasters) != len(methods):
        return print('Input list lengths are not the same')

    # make dictionary
    vars_dict = {}

    for i, var in var_names:
        vars_dict[var] = []
        vars_dict[var].append(rasters[i])
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

    if not os.path.exists(points):
        os.makedirs(temp_files)

    buff_dists = list(range(0, max, step))
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
    :param vars_dict: Variable dict. Each variable name is associated with a list containing raster dir and method code.
    :param buff_dict: Buffer dict. Each buffer distance (int) is associated with it's shapefile path.
    :param no2_csv: A csv file containing annual averaged NO2 w/ station IDs matching the point file.
    :param fit_vars: (optional) A list containing variable names that will be used to fit a regression model. If used,
    the out
    :return:
    """

def plot_decay_curves(no2_data, input_vars=[], input_buffs=[]):
    """"""
