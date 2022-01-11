"""
Python 3 functions for using APIs to pull and organize large temporal datasets

@author: xrnogueira
"""
import logging
import subprocess
import os

# static variables
import pandas as pd


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


def build_master_csv(main_csv, in_csv, columns, out_csv=None, join_by=None):
    """
    This function allows columns to be joined to master daily no2 observations .csv file.
    :param main_csv: the main daily no2 observations csv (to be overwritten or copied)
    :param in_csv: a csv with columns to join to the main one
    :param columns: a list of strings or single string of column headers in in_csv to join to main_csv
    :param out_csv: if specified (optional) the joined csv is saved in a new location, rather than overwriting main_csv
    :param join_by: a list with column(s) that when combined form unique identifiers (default: [station_id, day])
    :return: the output csv location
    """
    # set up dataframes and defaults
    main_df = pd.read_csv(main_csv)
    in_df = pd.read_csv(in_csv)
    print('Joining columns %s from %s to %s' % (columns, os.path.basename(in_csv), os.path.basename(main_csv)))

    if out_csv is None:
        out_csv = main_csv

    if isinstance(columns, str):
        columns = [columns]

    if join_by is None:
        join_by = ['station_id', 'dayofyear']

    jcols = in_df[join_by + columns]
    out_df = main_df.merge(jcols, on=join_by, how='left')
    out_df[columns] = out_df[columns].fillna(0)

    out_df.to_csv(out_csv)
    print('Output: %s' % out_csv)

    return out_csv


def make_annual_csv(daily_csv):
    """
    Creates an annual averaged NO2 observation csv from daily observations.
    :param daily_csv: daily observation csv w/ station_id and mean_no2 columns
    :return: annual averaged mean_no2 values for each station
    """
    in_df = pd.read_csv(daily_csv)
    grouped = in_df.groupby('station_id').first().reset_index()
    avg = in_df.groupby('station_id').mean_no2.mean().reset_index()
    in_dfll = grouped[['lat', 'long', 'station_id']]
    merged = avg.merge(in_dfll, how='left', on='station_id')

    out_dir = os.path.dirname(daily_csv)
    out_name = out_dir + '\\no2_annual_2019.csv'
    merged.to_csv(out_name)

    return print('Annual averages .csv @ %s' % out_name)


def make_monthly_csv(daily_csv):
    """
        Creates an annual averaged NO2 observation csv from daily observations.
        :param daily_csv: daily observation csv w/ station_id and mean_no2 columns
        :return: monthly averaged mean_no2 values for each station
        """
    in_df = pd.read_csv(daily_csv)
    grouped = in_df.groupby(['station_id', 'month']).first().reset_index()
    avg = in_df.groupby(['station_id', 'month']).mean_no2.mean().reset_index()
    in_dfll = grouped[['lat', 'long', 'station_id']]
    merged = avg.merge(in_dfll, how='left', on='station_id')

    out_dir = os.path.dirname(daily_csv)
    out_name = out_dir + '\\no2_monthly_2019.csv'
    merged.to_csv(out_name)

    return print('Monthly averages .csv @ %s' % out_name)

#################################################

dems = r'C:\Users\xrnogueira\Documents\Data\3DEP'
CSV_DIR = r'C:\Users\xrnogueira\Documents\Data\NO2_stations'
main_csv = CSV_DIR + '\\master_no2_daily.csv'
in_csv = CSV_DIR + '\\clean_no2_daily_2019_ZandZr.csv'
clean_daily = CSV_DIR + '\\clean_no2_daily_2019.csv'
test_csv = CSV_DIR + '\\test.csv'
#build_master_csv(main_csv, in_csv, columns, out_csv=None, join_by=None)
#make_annual_csv(clean_daily)
#make_monthly_csv(main_csv)

# run with prediction tables
tables_dir = r'C:\Users\xrnogueira\Documents\Data\chicago_tables'
in_table = tables_dir + '\\sums_1100.csv'
era_samp = tables_dir + '\\era5_samples.csv'
combo = tables_dir + '\\chicago_prediction_table.csv'
columns = ['pod_den_1100']
build_master_csv(combo, in_table, columns, out_csv=None, join_by=['pointid'])  # run with POINT_X, and POINT_Y

