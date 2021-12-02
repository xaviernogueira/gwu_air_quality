import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from useful_functions import init_logger
import logging


def prep_input(in_data, in_cols, test_prop=0.15):
    """
    This function takes a master NO2 observation .csv, keeps/cleans only specified columns, and outputs an X and Y DF.
    :param in_csv: a master NO2 observations csv containing independent and dependent variable columns
    :param in_cols: a list of strings containing valid column headers only
    :param test_prop: the proportion of the dataset (float, 0 to 1) that is reserved for testing
    :return: a list of len=2 containing a list w/ X and Y dataframes [0], and the train_test_split outputs [1]
    """
    # import dependencies and data
    from sklearn.model_selection import train_test_split
    logging.info('Prepping input data')

    # standardize column headers
    for col in list(in_data.columns):
        if in_data[col].dtypes == object:
            in_data[col].replace(' ', '_', regex=True, inplace=True)
        if ' ' in str(col)[:-1]:
            new = str(col).replace(' ', '_')
            if new[-1] == '_':
                new = new[:-1]
            in_data.rename(columns={str(col): new}, inplace=True)

    # keep only in_cols
    in_data = in_data[in_cols]

    # split to X and Y data
    ytr = in_data['mean_no2'].values  # define y variable
    xtr = in_data.drop('mean_no2', axis=1)  # define x variables

    # apply train/test split
    logging.info('Applying train/test %s/%s split...' % (round(1 - test_prop, 2), test_prop))
    X_train, X_test, y_train, y_test = train_test_split(xtr, ytr, test_size=test_prop, random_state=101)

    out = [[xtr, ytr], [X_train, X_test, y_train, y_test]]
    logging.info('Done')
    return out


def cross_cross(xtr, out_folder=None):
    """
    This function creates a cross-correlation plot for all independent variables.
    :param xtr: a pandas dataframe with only independent variables
    :param out_folder: a directory to save the plots (optional), if not specified, the plot saved in __file___
    :return:
    """
    # set up dependencies and folder
    import seaborn as sns
    logging.info('Creating independent variable cross-correlation plot...')
    sns.set_theme()
    sns.set_theme(style="whitegrid")

    if not out_folder is None:
        if isinstance(out_folder, str):
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

        else:
            return logging.error('ERROR: out_folder parameter must be None or a valid path string.')

    else:
        out_folder = os.path.dirname(__file__)

    out_file = out_folder + '\\cross_corrs.png'
    logging.info('Figure will be saved @ %s' % out_file)

    # Compute a correlation matrix and convert to long-form
    corr_mat = xtr.corr().stack().reset_index(name="correlation")

    # Draw each cell as a scatter point with varying size and color
    g = sns.relplot(
        data=corr_mat,
        x="level_0", y="level_1", hue="correlation", size="correlation",
        palette="rocket_r", hue_norm=(-0.5, 1), edgecolor=".2",
        height=12, sizes=(50, 1500), size_norm=(0, 0.5), legend='brief'
    )

    # Tweak the figure to finalize
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(.05)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")

    sns.set_theme()
    fig = g.figure.savefig(out_file)
    return fig


def train_xgb(X_train, y_train, param_grid, scoring='r2'):
    """
    Used GridCV to find optimal XGBoost parameters to fit the training dataset.
    :param X_train: dataframe or XDarray with independent variable training columns
    :param y_train: dataframe or XDarray with dependent variable training columns
    :param params_list: a list of lists of grid paramters to try. Must be of the form
    [gamma_range, eta_range, lambda_range, min_child_weight_range, max_depth_range]
    :param scoring: a scikit-learn scorer string (default is r2)
    :return: a list containing [model.cv_results_, model.best_estimator_, model.best_params_, model.best_score_]
    """
    # set up XGBoost regressor model
    xgb_model = xgb.XGBRegressor(booster='gbtree', eval_metric='reg:squarederror')
    xgb_model.fit(X_train, y_train)

    # iterate over all parameter combinations and use the best performer to fit
    xgb_iters = GridSearchCV(xgb_model, param_grid, cv=5, scoring=scoring, verbose=1, refit=True, return_train_score=True)
    xgb_iters.fit(X_train, y_train)

    cv_results_df = pd.from_dict(xgb_iters.cv_results_)
    print('Best params: %s, %s: %s' % (xgb_iters.best_params_, scoring, xgb_iters.best_score_))

    out_list = [cv_results_df, xgb_iters.best_estimator_, xgb_iters.best_params_, xgb_iters.best_score_]

    return out_list


def plot_model(X_test, y_test, best_estimator, best_params, out_folder):
    """
    Plots the GridSearch best_estimator against the test portion of the initial dataset
    :param X_test: the independent variable columns array or dataframe
    :param y_test: the test dependent variable array
    :param best_estimator: the best_estimator_ model selected during GridSearch (out_list[1])
    :param best_params: the best_params_ attribute of the selected model (out_list[2])
    :param out_folder: folder where the plot is saved as a figure
    :return: shows plot
    """
    from scipy.stats import gaussian_kde
    model = best_estimator
    grid_predictions = model.predict(X_test)
    plt.cla()

    # Calculate the point density
    xy = np.vstack([grid_predictions, y_test])
    z = gaussian_kde(xy)(xy)

    # make and format plot
    fig, ax = plt.subplots()
    ax.scatter(grid_predictions, y_test, c=z, s=20)

    plt.title('XGBoost - Predicting daily mean NO2 concentrations')
    plt.plot(np.arange(0, 60, 0.1), np.arange(0, 60, 0.1), c='red')
    plt.xlim(0, np.max(grid_predictions))
    plt.ylim(0, np.max(y_test))
    plt.xlabel('Predicted NO2 concentration')
    plt.ylabel('Actual daily NO2 concentration')
    plt.text(0, 22.5, best_params, fontsize='x-small')
    plt.text(1, 15, 'R2 = %s' % round(r2_score(y_test, grid_predictions), 2), fontsize='large')

    # save figure
    fig_name = out_folder + '\\model_test.png'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')

    return plt.show()


def plot_feature_importance(best_estimator, out_folder):
    """
    Plots feature importance for a model
    :param best_estimator: the best_estimator_ model selected during GridSearch (out_list[1]) or other model
    :param out_folder: folder where the plot is saved as a figure
    :return: shows plot
    """
    model = best_estimator
    plt.cla()

    # plot feature importance
    x = range(len(model.feature_importances_))
    plt.bar(x, model.feature_importances_)
    plt.xticks(x, model.get_booster().feature_names, rotation=-45)
    plt.subplots_adjust(bottom=0.40)

    # save figure
    fig_name = out_folder + '\\model_feature_importance.png'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')

    return plt.show()


def plot_hyperparams(cv_results_df, param_grid, out_folder):
    """
    This saves the model.cv_results_ item as a csv and saves plots of the distribution of scores for each parameter.
    :param cv_results_df: the model.cv_results_ item (out_list[0])
    :param param_grid: the param_grid dictionary with param name keys
    :param out_folder: a folder to save plots and .csv in (a sub-folder \\hyper_tuning is made_
    :return: none
    """
    # make folder to store hyper-parameter tuning
    hyp_dir = out_folder + '\\hyper_tuning'

    if not os.path.exists(hyp_dir):
        os.makedirs(hyp_dir)

    # get dictionary as pandas dataframe and save it as a csv
    scoring_df = pd.DataFrame.from_dict(cv_results_df)
    scoring_df.head(n=20)
    scoring_df.to_csv(hyp_dir + '\\hyper_params_scoring.csv')

    # make a list of hyper parameters to iterate over
    hypers = list(param_grid.keys())
    param_cols = []
    for param in hypers:
        if param != 'booster':
            # get column key for dataframe and add to list
            col_key = 'param_%s' % param
            param_cols.append(col_key)
            ax = sns.boxenplot(x=col_key, y='mean_train_score', data=scoring_df)
            ax.figure.savefig(hyp_dir + '\\%s.png' % param)
            ax.fig.clf()
    return


def main(in_csv, in_cols, params_list):
    init_logger(__file__)
    out_folder = os.path.dirname(in_csv)
    in_data = pd.read_csv(in_csv)
    in_data = in_data[in_cols]

    # set up parameter grid
    gammas, etas, lambdas, colsample_range, max_depths = params_list
    param_grid = {'gamma': gammas, 'eta': etas, 'reg_lambda': lambdas, 'colsample_bytree': colsample_range,
                  'max_depth': max_depths}

    # use GridSearch CV to tune model hyper-parameters
    out = prep_input(in_data, in_cols)
    X_df, Y_df = out[0]  # [0][0] is X dataframe, [0][1] is Y dataframe
    X_train, X_test, y_train, y_test = out[1]
    cross_cross(X_df, out_folder=out_folder)
    out_list = train_xgb(X_train, y_train, param_grid, scoring='r2')
    best_model = out_list[1]

    # plot model performance and feature importance
    plot_model(X_test, y_test, best_model, out_list[2], out_folder)
    plot_feature_importance(best_model, out_folder)

    return


#  ########## SET XGBOOST PARAMETER RANGES ###########
CSV_DIR = r'C:\Users\xrnogueira\Documents\Data\NO2_stations'
main_csv = CSV_DIR + '\\master_no2_daily.csv'
test_csv = CSV_DIR + '\\master_no2_daily_test_500_rows.csv'


keep_cols = ['mean_no2', 'weekend', 'sp', 'swvl1', 't2m', 'tp', 'u10', 'v10', 'blh', 'u100', 'v100', 'p_roads_1000',
                 's_roads_1700', 's_roads_3000', 'tropomi', 'pod_den_1100']
gamma_range = list(np.arange(0, 1, 0.5))
eta_range = list(np.arange(0.01, 0.3, 0.05))
lambda_range = list(np.arange(0.6, 1.4, 0.2))
colsample_range = list(np.arange(0.5, 1, 0.25))
max_depth_range = list(np.arange(5, 7, 1))


params_list = [gamma_range, eta_range, lambda_range, colsample_range, max_depth_range]

SCORERS = {'r2': make_scorer(r2_score), 'neg_mean_squared_error': make_scorer(mean_squared_error)}
SCORERS_str = list(SCORERS.keys())

if __name__ == "__main__":
    main(main_csv, keep_cols)


