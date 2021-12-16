import os
import joblib
import pandas as pd
from useful_functions import init_logger
from netcdf_functions import get_boundingbox
from arcpy_functions import align_rasters
import logging
import matplotlib.pyplot as plt

# import pyspatialml package
from pyspatialml import Raster


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

    # load model
    model = joblib.load(saved_model)

    return


if __name__ == "__main__":
    dummy_model_file = r'C:\Users\xrnogueira\Documents\Data\NO2_stations\MODEL_RUNS\Run1\run1_log.txt'
    make_prediction_raster(dummy_model_file, region='Chicago')
