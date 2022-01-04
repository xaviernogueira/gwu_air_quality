# import dependencies
import logging
import numpy as np
import scipy
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pykrige
from pykrige.rk import RegressionKriging

import matplotlib.pyplot as plt
from useful_functions import init_logger


def prep_csv_by_month(in_csv, drop_cols=None, save_month_dfs=False):
    """
    Splits a no2 observation csv by unique month codes, deleted extra columns
    :param in_csv: the path of a daily scale csv with a dtypes=int 'month' column
    :param drop_cols: a string or list of column headers to drop for whatever reason
    :param save_month_dfs: boolean. False is default. If True each month DataFrame is saves in a new folder as csv
    :return: a list containing a list of month DataFrames [0] (i.e., monthly values for each station as unique DataFramed)
    and a list with indexes matching return[0] holding 2D lat/long values as a numpy array
    """
    in_df = pd.read_csv(in_csv)
    logging.info('Prepping data for leave-one-out Kriging')

    # drop necessary columns
    if not drop_cols is None and isinstance(drop_cols, list):
        in_df.drop(drop_cols, axis=1, inplace=True)
        logging.info('Dropped columns: %s' % drop_cols)
    elif isinstance(drop_cols, str):
        in_df.drop([drop_cols], axis=1, inplace=True)
        logging.info('Dropped column: %s' % drop_cols)
    else:
        logging.info('No columns dropped')

    # standardize column headers
    for col in list(in_df.columns):
        if in_df[col].dtypes == object:
            in_df[col].replace(' ', '_', regex=True, inplace=True)
        if ' ' in str(col)[:-1]:
            new = str(col).replace(' ', '_')
            if new[-1] == '_':
                new = new[:-1]
            in_df.rename(columns={str(col): new}, inplace=True)

    # split by month and create output lists storing monthly DataFrame and 2D numpy lat/long arrays
    data_list = []
    lat_lon_list = []

    for month in range(1, 13):
        m_df = in_df.loc[in_df['month'] == 1]
        lat_lon = m_df[['lat', 'long']].to_numpy()
        data_list.append(m_df)
        lat_lon_list.append(lat_lon)

    logging.info('Monthly DataFrames and 2D lat/long arrats generated and stored in lists')
    return [data_list, lat_lon_list]


def rebuild_full_csv(in_dfs, out_folder):
    """
    Takes a list of pandas dataframes with overlapping to identical column headers and stacks them, and saves as a csv
    :param in_dfs: a list of pandas dataframes with overlapping column headers
    :param out_folder: the folder in which the output csv is saves
    :return: output csv path
    """
    return


def master_kriging_workflow(in_csv, drop_cols):
    init_logger(__file__)
    prep_csv_by_month(in_csv, drop_cols=None)

    return


if __name__ == "__main__":
    master_kriging_workflow()
