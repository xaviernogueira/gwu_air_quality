import os

import arcpy.analysis


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

def make_vars_dict(var_names rasters, methods):
    vars_dict = {}

    for i, var in var_names:
        vars_dict[var] = []
        vars_dict[var].append()

def decay_curves(no2_data, input_vars=[], input_buffs=[]):
    """"""
