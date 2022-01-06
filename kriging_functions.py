# import dependencies
import os
import logging
import numpy as np
import scipy
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pykrige

import matplotlib.pyplot as plt
from useful_functions import init_logger


def prep_csv_by_month(in_csv,  keep_cols=None, drop_cols=None, save_month_dfs=False):
    """
    Splits a no2 observation csv by unique month codes, deleted extra columns
    :param in_csv: the path of a daily scale csv with a dtypes=int 'month' column
    :param keep_cols: a list of column headers to keep
    :param drop_cols: (optional method) a string or list of column headers to drop instead
    :param save_month_dfs: boolean. False is default. If True each month DataFrame is saves in a new folder as csv
    :return: a dictionary with month indexes as keys containing a list of list for each month:
    [0] list of month DataFrames (i.e., monthly values for each station as unique DataFramed)
    [1] a list with indexes matching return[0] holding 2D lat/long values as a numpy array
    """
    in_df = pd.read_csv(in_csv)
    logging.info('Prepping data for leave-one-out Kriging')

    if keep_cols is not None:
        in_df = in_df[keep_cols]
    else:
        # drop necessary columns using drop_cols
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
    month_dict = {}

    for month in range(1, 13):
        copy = in_df.copy()
        m_df = copy.loc[copy['month'] == month]
        lat_lon = m_df[['lat', 'long']].to_numpy()

        month_dict[str(month)] = [m_df, lat_lon]
    logging.info('Monthly DataFrames and 2D lat/long arrays generated and stored in dict')

    return month_dict


def calculate_vif(month_dict, out_folder):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    logging.info('Checking Variance Inflation factor of each variable for each month')
    vif_dfs = []

    for month in range(1, 13):
        m_df = month_dict[str(month)][0]
        not_no2 = [i for i in list(m_df.columns) if i not in ['mean_no2', 'station_id', 'month', 'lat', 'long']]
        X = m_df[not_no2]
        X = X.reset_index()
        X = X.drop('index', axis=1)

        vif_data = pd.DataFrame()
        vif_data['variable'] = X.columns

        # calculating VIF for each feature
        vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                           for i in range(len(X.columns))]
        vif_data['month'] = int(month)

        vif_dfs.append(vif_data)

    # build master VIF DataFrame
    stk_df = pd.concat(vif_dfs, axis=0)
    master_vif = stk_df.groupby(['variable']).mean().reset_index()
    master_vif['consider_dropping'] = 'Yes'
    master_vif.loc[master_vif['VIF'] < 3, 'consider_dropping'] = 'No'
    master_vif.drop(['month'], axis=1, inplace=True)

    # save VIF csv
    out_csv = out_folder + '\\VIF_table.csv'
    master_vif.to_csv(out_csv)
    logging.info('VIF table saved @ %s' % out_csv)

    return out_csv


def leave_one_out_kriging(month_dict):
    from pykrige.rk import RegressionKriging

    rk_month_dfs = []
    for month in range(1, 13):
        logging.info('MONTH: %s' % month)
        m_df = month_dict[str(month)][0]
        lat_lon = month_dict[str(month)][1]

        # split data into dependent and independent data frames
        Y = m_df[['mean_no2']].to_numpy()
        not_no2 = [i for i in list(m_df.columns) if i not in ['mean_no2', 'station_id', 'month', 'lat', 'long']]
        X = m_df[not_no2]
        X = X.reset_index()
        X = X.drop('index', axis=1)

        # create a linear regression model (we could expirmenet with Lasso or Ridge regression)
        l_model = LinearRegression().fit(X, Y)
        logging.info('Linear regression score: %s' % l_model.score(X, Y))

        for i, var in enumerate(list(X.columns)):
            logging.info('Variable: %s' % var)
            coef = l_model.coef_[0][i]
            logging.info('Regression coef = %s\n' % coef)

        # build on LUR model with Regression Kringing with the "leave one out method"
        m_rk = RegressionKriging(regression_model=l_model, n_closest_points=20)
        logging.info("Pure LUR Score: ", m_rk.regression_model.score(X, Y))

        # set up lists to store each stations "leave out" prediction
        actuals = []  # dependent variable values
        predicts = []  # predictions at corresponding indices
        rk_scores = []  # kriging regression r2 scores (for averaging)

        # implement leave one out method
        for i in list(range(0, X.shape[0])):
            print('Index = %s' % i)
            out_X = X.iloc[lambda X: X.index == i]
            out_Y = Y[i]
            out_latlon = np.expand_dims(lat_lon[i], axis=1).T

            # OKAY FIX THIS ANNOYING ISSUE WHERE IT DROPS FOR INDEX 0 but for whatever reason can't for any other index
            in_X = X.iloc[lambda X: X.index != i]
            in_Y = np.delete(Y, i, axis=0)
            in_latlon = np.delete(lat_lon, i, axis=0)

            m_rk.fit(in_X, in_latlon, in_Y)

            kring_prediction = m_rk.predict(out_X, out_latlon)

            actuals.append(out_Y[0])
            predicts.append(kring_prediction[0][0])

        actuals = np.array(actuals)
        predicts = np.array(predicts)
        logging.info('Kriging prediction score: %s' % r2_score(actuals, predicts))
        m_df['no2_krig'] = predicts
        rk_month_dfs.append(m_df)

        logging.info('----------------------------------')

    return rk_month_dfs


def master_kriging_workflow(in_csv, keep_cols, check_vif=True):
    init_logger(__file__)

    out_folder = os.path.dirname(in_csv)
    month_dict = prep_csv_by_month(in_csv, keep_cols=keep_cols)  # can hard code to drop_cols if desired

    # calculate vif if desired and prompt user to check before continuing to linear regression
    if calculate_vif is True:
        vif_csv = calculate_vif(month_dict, out_folder)
        logging.info('Check the VIF table @ %s and re-run with a different drop_cols list if desired' % vif_csv)
        cont = input('Continue as is (YES or NO): ')

        if cont == 'YES':
            logging.info('Continuing Kriging workflow')
        elif cont == 'NO':
            return logging.info('Run again with updated drop_cols list')
        else:
            logging.error('ERROR: User input was: %s. It must be either YES or NO strings!' % cont)
    else:
        logging.info('Not calculating VIF. Set check_vif=True if this was not your intention.')

    # run kriging, make and stack predictions
    out_csv = out_folder + '\\master_monthly_no2_jan4_rk.csv'
    rk_month_dfs = leave_one_out_kriging(month_dict)
    stk_df = pd.concat(rk_month_dfs, axis=0)
    stk_df.to_csv(out_csv)

    return


########### INPUTS ##################
in_csv = r'C:\Users\xavie\Documents\DATASETS\master_monthly_no2_jan4.csv'
drop_cols = ['u10', 'v10', 'swvl1', 'u100', 'v100']
full_cols = ['lat', 'long', 'station_id', 'month', 'mean_no2', 'sp', 'swvl1', 't2m', 'tp', 'u10', 'v10', 'blh', 'u100', 'v100', 'p_roads_1000', 's_roads_1700', 's_roads_3000', 'tropomi', 'pod_den_1100', 'Z_r', 'Z']
keep_cols = [i for i in full_cols if i not in drop_cols]


if __name__ == "__main__":
    # keep in mind that drop_cols here can include more than the XGBoost. This is just for Regression Kriging
    master_kriging_workflow(in_csv, keep_cols, check_vif=False)
