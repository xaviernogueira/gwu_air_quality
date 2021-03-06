import logging
import os
import arcpy
import numpy as np
import pandas as pd

from useful_functions import init_logger


def batch_resample_or_aggregate(in_folder, cell_size, out_folder='', str_in='.tif', agg=False):
    """
    This function resamples or aggregates every raster in a folder, and saves the new raster in a new folder
    :param in_folder: Folder containing raster datasets
    :param cell_size: The cell size (float or int) in the same units of the raster
    :param out_folder: Output folder, if not specified, a folder 'resampled_{DIST} will be made in in_folder'
    :param str_in: (.tif is default) A string within the raster file name to select for resampling
    :param agg: Bool. If true, a SUM aggregation is used (for data like population) instead of bilinear resampling
    :return: The new folder location containing resampled raster datasets
    """
    # initialize logger, environment, and delete files list
    init_logger(__file__)
    arcpy.env.overwriteOutput = True
    del_files = []

    # create list of valid input files
    all_files = os.listdir(in_folder)
    in_files = [in_folder + '\\%s' % i for i in all_files if str_in in i]
    in_files = [i for i in in_files if i[-4:] == '.tif']
    if len(in_files) == 0:
        return print('ERROR. No valid input .tif files w/ %s in their name. Please run again.' % str_in)

    # find raster units, give user a chance to change cell_size input before processing
    in_spatial_ref = arcpy.Describe(in_files[0]).spatialReference
    in_cell_size = float(arcpy.GetRasterProperties_management(in_files[0], 'CELLSIZEX').getOutput(0))
    units = [in_spatial_ref.linearUnitName, in_spatial_ref.angularUnitName]
    unit = [i for i in units if i != ''][0]

    print('Units are in %s! Input cell size is %s' % (unit, in_cell_size))
    print('Output cell size will be %s %s' % (cell_size, unit))
    var = input('Is this correct? Y or N:')

    if var is not 'Y':
        return print('Adjust cell_size parameter and run again.')

    # if out_folder does not exist or is not specified, make new folder
    if out_folder == '':
        out_folder = in_folder + '\\resampled_%s%s' % (cell_size, unit)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # select raster files to process and resample or aggregate then resample
    for i, file in enumerate(in_files):
        name = os.path.split(file)[1]
        try:
            if not agg:
                # create output path then resample
                out_file = out_folder + '\\%s' % name

                arcpy.Resample_management(file, out_file, cell_size, 'BILINEAR')
                logging.info('Resampled %s' % name)

            # if agg == True, either aggregate to the output cell size (if divisible) or aggregate and then resample
            elif agg:
                # create text file to record which of the newly named rasters correspond to what
                txt_dir = out_folder + '\\aggregate_key.txt'
                out_txt = open(txt_dir, 'w+')
                out_file = out_folder + '\\agg%s.tif' % i

                factor = int(cell_size // in_cell_size)
                if cell_size % in_cell_size == 0:
                    in_ras = arcpy.sa.Raster(file)
                    out_ras = arcpy.sa.Aggregate(in_ras, factor, 'Sum')
                else:
                    temp = out_folder + '\\temp%s.tif' % i
                    in_ras = arcpy.sa.Raster(file)
                    out_agg = arcpy.sa.Aggregate(in_ras, factor, 'Sum')
                    out_agg.save(temp)
                    del_files.append(temp)
                    out_ras = arcpy.sa.Resample(out_agg, 'Average', output_cellsize=cell_size)

                # Save the output
                out_txt.write('\n %s -> %s\n' % (name, out_file))
                out_ras.save(out_file)
                logging.info('Aggregated %s' % name)
                out_txt.close()

        except arcpy.ExecuteError:
            logging.info(str(arcpy.GetMessages()))
            logging.info('ERROR, skipped %s' % name)

    # delete extra files
    for file in del_files:
        try:
            arcpy.Delete_management(file)
        except arcpy.ExecuteError:
            logging.info('Could not delete %s' % file)

    return out_folder


def batch_raster_project(in_folder, spatial_ref, out_folder='', suffix='_p.tif'):
    """
    This function batch projects rasters and places them in a new flder
    :param in_folder: folder containing .tif rasters
    :param out_folder: folder to save output rasters
    :param spatial_ref: a spatial reference file or a raster/shapefile with the desired spatial reference
    :param suffix: suffix to add to output rasters (_p is default i.e., btw.tif -> btw_p.tif)
    :return: the out_folder
    """
    # initialize logger, environment, and delete files list
    init_logger(__file__)
    arcpy.env.overwriteOutput = True
    del_files = []

    # create list of valid input files
    all_files = os.listdir(in_folder)
    in_names = [i for i in all_files if i[-4:] == '.tif']
    in_files = [in_folder + '\\%s' % i for i in in_names]
    if len(in_files) == 0:
        return print('ERROR. No valid input .tif files in %s. Please run again.' % in_folder)

    # create output spatial reference object
    if isinstance(spatial_ref, str):
        ext = spatial_ref[-4:]
        if ext == '.tif' or ext == '.shp':
            out_sr = arcpy.Describe(spatial_ref).spatialReference
        else:
            out_sr = spatial_ref

    else:
        return print('spatial_ref must be a .tif, .shp, or a arcpy spatial reference object')

    # if out_folder does not exist or is not specified, make new folder
    if out_folder == '':
        out_folder = in_folder + '\\projected'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # project and save rasters
    for i, file in enumerate(in_files):
        name = in_names[i]
        out_ras = out_folder + '\\%s' % name.replace('.tif', suffix)

        try:
            arcpy.ProjectRaster_management(file, out_ras, out_coor_system=out_sr, resampling_type='BILINEAR')
            logging.info('Projected %s -> %s' % (name, out_ras))

        except arcpy.ExecuteError:
            logging.info(str(arcpy.GetMessages()))
            logging.info('ERROR, skipped %s' % file)

    return out_folder


def netcdf_to_tiff(ncf, tifname='', out_folder='', lon_lat=None):
    """
    Converts a .nc netCDF file into N rasters (one for each variable) with bands representing the time dimension.
    :param ncf: The location of a .nc netCDF file (no .ncf capabilities)
    :param tifname: Basename (str) for the output tif (i.e, era5) with length <= 5
    :param out_folder: An alternative output folder can be designated (default i same folder as ncf)
    :param lon_lat: A list the NetCDF X and Y dimension names (default is ['longitude', 'latitude'])
    :return: A dictionary with variable names as keys, and tiff locations are values
    """
    # initialize logger, format directories, and set defaults
    if lon_lat is None:
        lon_lat = ['longitude', 'latitude']

    init_logger(__file__)
    logging.info('Converting %s to TIFF...' % ncf)
    in_dir = os.path.dirname(ncf)
    out_dict = {}

    if out_folder == '':
        out_folder = in_dir
    if tifname == '':
        tifname = 'ncf'
    elif '.tif' == tifname[-4:]:
        tifname = tifname[:-4]

    # find all variables in the netcdf
    nc_fp = arcpy.NetCDFFileProperties(ncf)
    variables = [str(i) for i in nc_fp.getVariables()]

    drops = ['latitude', 'longitude', 'lat', 'long', 'time', 'lon']
    variables = [i for i in variables if i not in drops]

    if len(variables) == 0:
        return logging.info('ERROR: No NetCDF variables')

    else:
        logging.info('NetCDF variables: %s' % variables)

        # make a raster for each variable, with bands corresponding to the time dimension

        for var in variables:
            tras = out_folder + '\\%s_%sv1.tif' % (tifname, var)
            out_dir = tras.replace('v1.tif', '.tif')
            try:

                out = arcpy.MakeNetCDFRasterLayer_md(ncf, var, lon_lat[0], lon_lat[1],
                                                     out_raster_layer=tras, band_dimension='time')
                arcpy.CopyRaster_management(out, out_dir)
                out_dict[var] = out_dir
                logging.info('Made tiff @ %s' % out_dir)

            except arcpy.ExecuteError:
                logging.info(str(arcpy.GetMessages()))

    logging.info('Done')
    print(out_dict)
    return out_dict


def make_averaged_CRU_rasters(in_folder, variable='q_air'):
    """INCOMPLETE"""
    arcpy.CheckOutExtension("Spatial")

    # Define input folder and create list of TIF rasters in folder
    logging.info('Averaging %s rasters in %s' % (variable, in_folder))
    arcpy.env.workspace = in_folder
    all_rasters = arcpy.ListRasters(raster_type='TIF')
    rasters = [i for i in all_rasters if variable in i]

    # Run cell statistics
    calc = arcpy.sa.CellStatistics(rasters, statistics_type="MEAN")
    calc.save(r'E:\RASTERS\raster.img')

    # save as a new raster
    out_ras = in_folder + '%s_averaged.tif' % variable
    logging.info('Done \n Output: %s' % out_ras)

    return out_ras


def era5_sample_to_csv(in_table, tiff_dict, sample_points, out_table):
    """
    This function makes a multi-band sample of a batch of ERA5 data (i.e., using nearest raster cells), and
    creates a csv exporting only the aligned month's band data for all variables.
    :param in_table: A .csv file with all no2 observations and month codes
    :param tiff_dict: Dict (output by netcdf_to_tiff) where keys are variable names, and values are multi-band tiffs
    :param sample_points: points with unique station ID identifiers that have been aligned to ERA5 raster cells
    :param out_table: A NO2 station .csv file with monthly ERA5 variable values associated with each observation row
    """
    # initialize logger and format directories
    init_logger(__file__)
    logging.info('Sampling from ERA5 monthly level data...')
    out_dir = os.path.dirname(in_table)
    arcpy.env.overwriteOutput = True

    temp_files = out_dir + '\\temp_files'
    if not os.path.exists(temp_files):
        os.makedirs(temp_files)

    var_names = list(tiff_dict.keys())

    in_df = pd.read_csv(in_table)
    in_df.sort_values('station_id', inplace=True)

    for var in var_names:
        logging.info('Extracting values from variable: %s' % var)
        tiff = tiff_dict[var]
        t_dbf = temp_files + '\\%s_sample.dbf' % var
        t_csv = t_dbf.replace('.dbf', '.csv')

        # make a sample dataframe with a _Band_# header suffixes where # is the month index
        sample_table = arcpy.sa.Sample(tiff, sample_points, t_dbf, unique_id_field='station_id')

        if os.path.exists(t_csv):
            os.remove(t_csv)
        arcpy.TableToTable_conversion(sample_table, os.path.dirname(t_csv), os.path.basename(t_csv))

        samp_df = pd.read_csv(t_csv)

        # make an array of equal length as all no2 observations with corresponding monthly ERA5 values
        val_list = []
        prev_id = 9999999999  # facilities faster processing
        sub_df = ''  # placeholder

        samp_df.rename(columns={'era5_sp_Ba': 'era5_sp_12'}, inplace=True)

        # build month codes for column header identification
        months = []
        for m in list(range(1, 13)):
            if m <= 9:
                month_h = '_%s' % m

            else:
                month_h = m

            months.append(month_h)

        # iterate over all observation rows and pull values
        for j, row in enumerate(in_df.iterrows()):
            rowd = dict(row[1])
            id = int(rowd['station_id'])

            # make a station id specific sub dataframe if we haven't already
            if j == 0 or id != prev_id:
                samp_id = 'no2_annual'
                sub_df = samp_df.loc[lambda samp_df: samp_df[samp_id] == id]
                prev_id = id

            # identify the extra column header (see arcpy sample error description)
            col_heads = [i for i in sub_df.columns if 'era5' in i]

            extra = []
            normal = []
            if len(col_heads) == 12:
                present = False
                for m_label in months:
                    for head in col_heads:
                        if '_%s' % m_label in head:
                            normal.append(head)
                extra = [i for i in col_heads if i not in normal]

            # extract value for the appropriate month and add to list
            month = int(rowd['month'])
            month_h = months[month - 1]

            col_head = [i for i in col_heads if '_%s' % month_h in i]

            if len(col_head) == 1:
                val = sub_df[[col_head[0]]].to_numpy()[0]
                val_list.append(val)

            # to fix the strange arcpy sampling issue where band 12 (last band) is not get labeled correctly
            elif len(col_head) == 0:
                new_head = extra[0]
                val = sub_df[[new_head]].to_numpy()[0]
                val_list.append(val)

            else:
                return logging.info('ERROR: Multiple Band columns for each month')

        in_df[var] = np.array(val_list)

    in_df.to_csv(out_table)

    return out_table


def raster_sample(in_table, sample_points, var_dict):
    """
    Plain bagel raster sampling (w/o month or days)
    :param in_table: A table with daily NO2 observations
    :param sample_points: AQ station sample points with a station_id field
    :param var_dict: a dictionary with variable names as keys and associated rasters as items
    :return: a new csv
    """

    # initialize logger and format directories
    init_logger(__file__)
    logging.info('Running plain bagel (no months/days) raster sampling.')
    out_dir = os.path.dirname(sample_points)
    arcpy.env.overwriteOutput = True
    out_csv = in_table.replace('.csv', '_export.csv')
    temp_files = out_dir + '\\temp_files'
    if not os.path.exists(temp_files):
        os.makedirs(temp_files)

    # set variables names
    var_names = list(var_dict.keys())

    in_df = pd.read_csv(in_table)
    in_df.sort_values('station_id', inplace=True)
    out_df = in_df.copy()
    samp_dfs = []

    for var in var_names:
        ras = var_dict[var]
        ras_name = os.path.basename(ras)[:-4]
        logging.info('Pulling station point %s values...' % var)
        t_dbf = temp_files + '\\%s_sample.dbf' % var
        t_csv = t_dbf.replace('.dbf', '.csv')

        # make a sample dataframe with a _Band_# header suffixes where # is the month index
        sample_table = arcpy.sa.Sample(ras, sample_points, t_dbf, unique_id_field='station_id')

        if os.path.exists(t_csv):
            os.remove(t_csv)

        arcpy.TableToTable_conversion(sample_table, os.path.dirname(t_csv), os.path.basename(t_csv))

        samp_df = pd.read_csv(t_csv)
        samp_df.rename(columns={ras_name: var, 'no2_annual': 'station_id'}, inplace=True)
        samp_dfs.append(samp_df)

    # join to the daily observation csv
    for i, df in enumerate(samp_dfs):
        var = var_names[i]
        out_df = out_df.merge(df, on=['station_id'], how='left')
        out_df[var] = out_df[var].fillna(0)

    out_df.to_csv(out_csv)
    logging.info('Done\nOutput csv with variables %s @ %s' % (var_names, out_csv))

    return out_csv


def align_rasters(raster_dict, extent, region, out_folder):
    """
    A function that makes region clipped aligned rasters prepped for Pyspatialml
    :param raster_dict: a dictionary of variable names (len<9) as keys and rasters as objects
    :param extent: a rectangular shapefile to define processing extent
    :param region: the name of the region (used for folder naming only)
    :param out_folder: the top level folder in which to store city predictions
    :return: dictionary with raster names and cropped, aligned rasters
    """
    # prep output folder and and create sub folders
    region_dir = out_folder + '\\%s' % region
    logging.info('Making output folder %s' % region_dir)
    clipped_dir = region_dir + '\\clipped_no_align'
    aligned_dir = region_dir + '\\aligned_rasters'
    dirs = [out_folder, region_dir, clipped_dir, aligned_dir]

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for i, var in enumerate(raster_dict.keys()):
        ras = raster_dict[var]

        # clip by extent polygon
        out_ras = out_folder + '\\%s' % os.path.basename(ras)
        arcpy.Clip_managment(ras, extent, clipped_dir)

        # use snap raster to align cells in all clipping
        if i == 0:
            arcpy.env.snapRaster = out_ras

        # update dictionary with clipped raster
        raster_dict[var] = out_ras

    return raster_dict


#  ####### CHOOSE WHAT TO RUN ##########
raster_funcs = False  # batch raster resample/aggreagte and project
era5_extract = False  # run era5 extraction, must use aligned era5 points
cru_extract = False  # run CRU extraction, we can use real points
era_sl_extract = False  # run era5-sl extraction, we can use real points
elevation_extract = False  # run elevation (Z) and elevation difference (Z_d) extraction

#  ####### DEFINE CONSTANT INPUTS ##########

no2_stations_daily = r'C:\Users\xrnogueira\Documents\Data\NO2_stations\clean_no2_daily_2019.csv'
DIR = os.path.dirname(no2_stations_daily)
actual_sample_points = DIR + '\\no2_annual_2019_points.shp'
era5_aligned_points = DIR + '\\no2_annual_2019_points_era5_aligned.shp'

# ######### RUN CHOSEN DATA EXTRACTIONS ##########

if raster_funcs:
    dem_folder = r'C:\Users\xrnogueira\Documents\Data\3DEP'
    pop_den = r'C:\Users\xrnogueira\Documents\Data\Population_density'
    ras_for_reference = r'C:\Users\xrnogueira\Documents\Data\resampled_popden_old\agg5.tif'
    resampled_pop = r'C:\Users\xrnogueira\Documents\Data\resampled_popden'
    batch_resample_or_aggregate(in_folder=pop_den, cell_size=0.001, out_folder=resampled_pop, str_in='.tif', agg=True)
    batch_raster_project(resampled_pop, spatial_ref=ras_for_reference, out_folder='', suffix='_p.tif')

if era5_extract:
    era5_ncf = r'C:\Users\xrnogueira\Documents\Data\ERA5\adaptor.mars.internal-1636309996.9508529-3992-17-5d68984c-35e3-4010-9da7-aaf52d0d05a6.nc'
    era5_dict = {'sp': 'C:\\Users\\xrnogueira\\Documents\\Data\\ERA5\\era5_sp.tif',
                 'swvl1': 'C:\\Users\\xrnogueira\\Documents\\Data\\ERA5\\era5_swvl1.tif',
                 't2m': 'C:\\Users\\xrnogueira\\Documents\\Data\\ERA5\\era5_t2m.tif',
                 'tp': 'C:\\Users\\xrnogueira\\Documents\\Data\\ERA5\\era5_tp.tif',
                 'u10': 'C:\\Users\\xrnogueira\\Documents\\Data\\ERA5\\era5_u10.tif',
                 'v10': 'C:\\Users\\xrnogueira\\Documents\\Data\\ERA5\\era5_v10.tif'}
    era5_obs_table = DIR + '\\no2_obs_wERA5.csv'
    era5_dict = netcdf_to_tiff(era5_ncf, tifname='era5', out_folder='')
    era5_sample_to_csv(no2_stations_daily, tiff_dict=era5_dict, sample_points=era5_aligned_points,
                       out_table=era5_obs_table)

if cru_extract:
    cru_obs_table = DIR + '\\no2_obs_wCRU.csv'
    cru_dir = r'C:\Users\xrnogueira\Documents\Data\ERA5\CRU_data\Qair_specific_humidity'
    cru_files = os.listdir(cru_dir)
    for i, f in enumerate(cru_files):
        file = cru_dir + '\\%s' % f
        m_code = i + 1
        cru_dict = netcdf_to_tiff(file, tifname='cru_%s' % m_code, out_folder='', lon_lat=['lon', 'lat'])
    #  make_averaged_CRU_rasters(in_folder, variable='q_air')
    #  era5_sample_to_csv(no2_stations_daily, tiff_dict=cru_dict, sample_points=actual_sample_points, out_table=cru_obs_table)

if era_sl_extract:
    era_sl = r'C:\Users\xrnogueira\Documents\Data\ERA5\ERA5_SL\adaptor.mars.internal-1637015018.6141229-26853-13-ea530db3-0cdd-459e-b83c-f3d0540479c1.nc'
    era5_sl_obs_table = DIR + '\\no2_obs_wERA5_SL.csv'
    netcdf_to_tiff(era_sl, tifname='era5_SL', out_folder='')
    era5_sl_dict = {'blh': 'C:\\Users\\xrnogueira\\Documents\\Data\\ERA5\\ERA5_SL\\era5_SL_blh.tif',
                    'u100': 'C:\\Users\\xrnogueira\\Documents\\Data\\ERA5\\ERA5_SL\\era5_SL_u100.tif',
                    'v100': 'C:\\Users\\xrnogueira\\Documents\\Data\\ERA5\\ERA5_SL\\era5_SL_v100.tif'}
    era5_sample_to_csv(no2_stations_daily, era5_sl_dict, sample_points=actual_sample_points,
                       out_table=era5_sl_obs_table)

if elevation_extract:
    dem_dir = r'C:\Users\xrnogueira\Documents\Data\3PED'
    fine_dem = dem_dir + '\\ETOPO1_Ice_g_geotiff.tif'
    relative_dem = dem_dir + '\\z_rel.tif'
    var_dict = {'Z': fine_dem}
    raster_sample(no2_stations_daily, actual_sample_points, var_dict)

# inputs for running locally
DIR = r'C:\Users\xrnogueira\Documents\Data\ERA5\ERA5_rasters'
OUT = r'C:\Users\xrnogueira\Documents\Data\usa_rasters_0p001'
#batch_resample_or_aggregate(DIR, cell_size=0.001, out_folder=OUT, str_in='.tif', agg=False)