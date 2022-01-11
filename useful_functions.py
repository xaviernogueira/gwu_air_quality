"""
Quick and useful functions for data science: pre-processing, plotting, etc.

author @xaviernogueira
"""
import pandas as pd
import logging
import matplotlib.pyplot as plt
import os


def init_logger(filename, log_name=None):
    """Initializes logger w/ same name as python file or a specified name if log_name is given a valid path (.log)"""

    if log_name is not None and log_name[-4:] == '.log':
        if os.path.exists(os.path.dirname(log_name)):
            name = log_name
        else:
            return print('ERROR: Logger cannot be initiated @ %s' % log_name)
    else:
        name = os.path.basename(filename).replace('.py', '.log')

    logging.basicConfig(filename=name, filemode='w', level=logging.INFO)
    stderr_logger = logging.StreamHandler()
    stderr_logger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.getLogger().addHandler(stderr_logger)

    return


def to_df(data):
    """
    Converts .csv to data frames, and marks conversion with a boolean
    :param data: a pandas data frame or csv
    :return: list len(2) w/ pandas data frame [0], and True if input was a .csv file, False otherwise [1]
    """
    c = False
    if isinstance(data, pd.DataFrame):
        return data, c

    elif isinstance(data, str):
        if data[-3:] == 'csv':
            df = pd.read_csv(data).copy()
            c = True
        return df, c
    else:
        return print('ERROR: Input must be csv or pandas data frame')


def drop_redundant_cols(data, thresh=1):
    """ Drops columns w/ less than N unique values. If input is a csv, a csv is saved as the output.
    :param data: a pandas data frame or csv
    :param thresh:: set the amount of unique values where <= the column is dropped (default is 1)
    :return same format as input. i.e., pandas data frame or csv.
    """
    df, c = to_df(data)

    drop_list = []
    for col in list(df.columns):
        if len(df[str(col)].unique()) <= thresh:
            drop_list.append(col)

    if len(drop_list) > 0:
        df.drop(drop_list, axis=1, inplace=True)

    if c:
        out = df.to_csv(data)
    else:
        out = df

    return out


def spaces_format(data):
    """
    This function turns spaces in column headers as well as the data fields and replaces them w/ underscores
    :param data: a pandas data frame or a csv
    :return: same format as the input
    """
    df, c = to_df(data)

    for col in list(df.columns):
        if df[col].dtypes == object:
            df[col].replace(' ', '_', regex=True, inplace=True)
        if ' ' in str(col):
            new = str(col).replace(' ', '_')
            df.rename(str(col), new)

    if c:
        out = df.to_csv(data)
    else:
        out = df

    return out


def cartography(ax, projection):
    """"Add geographic features (may not work unless on cartopy 0.20.0)"""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.gridlines(projection, draw_labels=True, alpha=0, linestyle='--')


def make_test_csv(csv, rows=500):
    """
    Takes a csv and randomly samples N number of rows to make a ML test csv (faster computation)
    :param csv: a csv
    :param rows: number of rows for test csv (int, default is 500)
    :return: new test csv
    """

    in_df = pd.read_csv(csv)
    shuffled = in_df.sample(frac=1).reset_index()

    if isinstance(rows, int):
        out_df = shuffled.sample(n=rows)

    else:
        return print('ERROR: Rows parameter must be an integer')

    out_dir = os.path.dirname(csv)
    out_csv = out_dir + '\\%s' % os.path.basename(csv).replace('.csv', '_test_%s_rows.csv' % rows)

    out_df.to_csv(out_csv)
    return out_csv


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
    from pyspatialml import Raster
    import rasterio
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

    return stack, month

def main(csv, rows=500):
    make_test_csv(csv, rows)

# ########### DEFINE INPUTS #############
CSV_DIR = r'C:\Users\xrnogueira\Documents\Data\NO2_stations'
main_csv = CSV_DIR + '\\master_no2_daily.csv'

if __name__ == '__main__':
    main(main_csv)

