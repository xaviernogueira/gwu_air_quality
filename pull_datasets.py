"""
Python 3 functions for using APIs to pull and organize large temporal datasets

@author: xrnogueira
"""
import logging
import subprocess
import os

# static variables
USERNAME = ''  # data hub username
PASSWORD = ''  # data hub password


def init_logger(filename):
    """Initializes logger w/ same name as python file"""

    logging.basicConfig(filename=os.path.basename(filename).replace('.py', '.log'), filemode='w', level=logging.INFO)
    stderr_logger = logging.StreamHandler()
    stderr_logger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.getLogger().addHandler(stderr_logger)

    return


def cmd(command):
    """Executes command prompt command and logs messages"""

    logger = logging.getLogger(__name__)
    try:
        res = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        msg = 'Command failed: %s' % command
        logger.error(msg)
        raise Exception(msg)

    # log any cmd line messages
    msg = res.communicate()[1]
    logger.info(msg)

    return


def convert_raster(in_raster, out_folder='', out_form=['GTiff', '.tif']):
    """
    :param in_raster: An input raster that is not the same format as the out_form
    :param out_folder: optional, allows output folder to be specified and created. Default is same folder as input.
    :param out_form: A list storing the output format gdal name [0] and the extension [1]. default it GeoTIFF.
    :return: path of created GeoTIFF file (or other format)
    """
    import osgeo
    from osgeo import gdal

    # Open existing dataset
    in_file = gdal.Open(in_raster)

    if out_folder == '':
        out_folder = os.path.dirname(in_raster)

    no_ext = os.path.splitext(os.path.basename(in_raster))[0]
    out_dir = out_folder + '\\%s' % no_ext + out_form[1]

    # Ensure number of bands in GeoTiff will be same as in GRIB file.
    bands = []  # Set up array for gdal.Translate().
    if in_file is not None:
        band_num = in_file.RasterCount  # Get band count
    for i in range(band_num + 1):  # Update array based on band count
        if i == 0:  # gdal starts band counts at 1, not 0 like the Python for loop does.
            pass
        else:
            bands.append(i)

    # Output to new format using gdal.Translate. See https://gdal.org/python/ for osgeo.gdal.Translate options.
    out_file = gdal.Translate(out_dir, in_file, format=out_form[0], bandList=bands)

    # Properly close the datasets to flush to disk
    in_file = None
    out_file = None

    return out_dir


def move_or_delete_files(in_folder, out_folder, str_in):
    """
    Moves files from one folder to another by a string query
    :param in_folder: Folder containing files to be moved
    :param out_folder: Folder to move files to, if 'False', files will be deleted!!
    :param str_in: string that if is contained within a file path, the file is selected to move
    """
    import shutil

    # initialize logger and find files
    init_logger(__file__)
    all_files = os.listdir(in_folder)
    move_names = []

    for file in all_files:
        if str_in in file:
            move_names.append(file)

    move_dirs = [(in_folder + '\\%s' % i) for i in move_names]

    if isinstance(out_folder, str):
        # make out_folder if it doesn't exist already
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # create output locations and move files
        out_dirs = [(out_folder + '\\%s' % i) for i in move_names]

        for i, dir in enumerate(move_dirs):
            shutil.move(dir, out_dirs[i])
            logging.info('Moved %s from %s to %s' % (move_names[i], in_folder, out_folder))

        return out_folder

    elif isinstance(out_folder, bool) and not out_folder:
        # take user input to list of files to be deleted
        print('STOP SIGN: Check list of files that will be deletes, enter Y to proceed and N to stop!')
        print('Files to delete: %s ' % move_names)
        val = input('Input Y to proceed:')

        if val:
            for dir in move_dirs:
                try:
                    os.remove(dir)
                    logging.info('Deleted %s' % dir)
                except PermissionError:
                    print('Could not get permission to delete %s' % dir)

            return print('Deleted %s files in %s' % (len(move_dirs), in_folder))

    else:
        return print('out_folder parameter must be a directory name to move files, or =False to delete files')


#################################################
netcdf = r'C:\Users\xrnogueira\Documents\Data\NO2_tropomi\by_month'
cdf_files = os.listdir(netcdf)
dems = r'C:\Users\xrnogueira\Documents\Data\3DEP'

# make a list of rasters
inputs = [netcdf + '\\' + i for i in cdf_files]


#################################################
def main():
    for i in inputs:
        print(i)
        convert_raster(i, out_folder='', out_form=['GTiff', '.tif'])

if __name__ == "__main__":
    main()
