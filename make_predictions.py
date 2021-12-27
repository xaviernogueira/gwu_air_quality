import os
import joblib
import pandas as pd
from useful_functions import init_logger
from netcdf_functions import get_boundingbox
from arcpy_functions import align_rasters
import logging
import matplotlib.pyplot as plt

from pyspatialml import Raster
import rasterio


def bbox_poly(bbox, region, out_folder):
    import geopandas as gpd
    from shapely.geometry import Polygon

    # define output location
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    out_shp = out_folder + '\\%s_bbox.shp' % region

    # get bounding box coordinates and format
    long0, long1, lat0, lat1 = bbox
    logging.info('Prediction extent coordinates: %s' % bbox)

    poly = Polygon([[long0, lat0],
                    [long1, lat0],
                    [long1, lat1],
                    [long0, lat1]])

    # save as a shapefile and return it's path
    gpd.GeoDataFrame(pd.DataFrame(['p1'], columns=['geom']),
                     crs={'init': 'epsg:4326'},
                     geometry=[poly]).to_file(out_shp)
    return out_shp


def plot_month_rasters(cropped_raster_dict, month_index, out_folder):
    """
    This function stacks rasters for each variable for a specific month. The stacked rasters are plotted.
    :param raster_dict: a dictionary containing variable names (str) as keys and AOI cropped raster .tif files as items
    :param month_index: index (int) for the month (i.e., January = 1)
    :param out_folder: a folder to store the plotted rasters
    :return: a raster item for a given month containing each variable as a band
    """

    # set up static lists
    cmaps = ['Purples', 'Greens', 'Reds', 'YlGnBu', 'RdPu']
    months = ['January', 'Febuary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']

    # set up list for plotting
    stack_list = []
    names_list = []

    # set up iteration variables
    month = months[month_index - 1]
    cmap_i = 0

    # for each raster, check if it has 12 bands or 1, and if it has 12 add the correct month layer to a stack
    for i, name in enumerate(list(cropped_raster_dict.keys())):
        full_raster = Raster(cropped_raster_dict[name])
        names_list.append(name)

        bands = full_raster.count
        logging.info('Raster %s has %s bands' % (name, bands))

        if bands > 1 and bands == 12:
            raster = full_raster.iloc[month_index - 1]
        elif bands == 1:
            raster = full_raster.iloc[0]
        else:
            return logging.error('ERROR: Input raster %s is not single band and also does not contain 12 month bands' % name)

        # set up colormap iterator and assign each layer a different colormap for plotting 
        cmap_i += 1
        if cmap_i >= len(cmaps):
            cmap_i = 0
        raster.cmap = cmaps[cmap_i]

        stack_list.append(raster)

    # convert the stacked raster layers into a raster, and plot each layer
    stack = Raster(stack_list)
    stack.plot(
        title_fontsize=10,
        label_fontsize=8,
        legend_fontsize=6,
        names=names_list,
        fig_kwds={"figsize": (8, 4)},
        subplots_kwds={"wspace": 0.3}
    )
    plt.title(month)
    plt.show()

    # save figure
    fig_name = out_folder + '\\%s_input_rasters.png' % month
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    logging.info('Done. Input raster plot for month %s saved @ %s' % (month, fig_name))

    return stack

def make_prediction_raster(saved_model, region='FULL'):
    init_logger(__file__)

    # create output folder
    m_dir = os.path.dirname(saved_model)
    out_folder = m_dir + '\\%s_prediction' % region

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # define prediction extent
    if region != 'FULL':
        logging.info('Making prediction for region: %s' % region)
        bbox = get_boundingbox(region, output_as='boundingbox', state_override=False)
        bbox_shp = bbox_poly(bbox, region, out_folder)
    else:
        logging.info('Making prediction for the continental USA')
        # get raster intersections

    # load model
    model = joblib.load(saved_model)



    return


if __name__ == "__main__":
    dummy_model_file = r'C:\Users\xrnogueira\Documents\Data\NO2_stations\MODEL_RUNS\Run1\run1_log.txt'
    make_prediction_raster(dummy_model_file, region='Chicago')
