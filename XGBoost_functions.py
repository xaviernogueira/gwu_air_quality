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
import joblib
import logging


def prep_input(in_data, in_cols, test_prop):
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


def prep_output(main_folder):
    """
    Folder organizing function. Creates sequential  main_folder/MODEL_RUNS/Run# folders to store results.
    :param main_folder: the folder containing the input csv
    :return: main_folder/MODEL_RUNS/Run#
    """
    runs_folder = main_folder + '\\MODEL_RUNS'
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    folders = os.listdir(runs_folder)
    subs = [name for name in folders if os.path.isdir(os.path.join(runs_folder, name))]
    run_dirs = [i for i in subs if 'Run' in i.split()[-1]]
    out_dir = ''
    num = 1
    stop = False
    while not stop:
        dir_name = 'Run%s' % num
        if dir_name in run_dirs:
            num += 1
        else:
            stop = True
            out_dir = runs_folder + '\\%s' % dir_name
            os.makedirs(out_dir)

    if out_dir != '':
        return out_dir
    else:
        return logging.info('ERROR: Output folder not defined.')


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

    out_file = out_folder + '\\x_variables_cross_corrs.png'

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
    logging.info('Cross-correlation figure saved @ %s' % out_file)

    return fig


def train_xgb(X_train, y_train, param_grid, k, scoring='r2'):
    """
    Used GridCV to find optimal XGBoost parameters to fit the training dataset.
    :param X_train: dataframe or XDarray with independent variable training columns
    :param y_train: dataframe or XDarray with dependent variable training columns
    :param params_list: a list of lists of grid paramters to try. Must be of the form
    [gamma_range, eta_range, lambda_range, min_child_weight_range, max_depth_range]
    :param k: Number of K-folds (integer, default passed in via train_and_test workflow is 5)
    :param scoring: a scikit-learn scorer string (default is r2)
    :return: a list containing [model.cv_results_, model.best_estimator_, model.best_params_, model.best_score_]
    """
    # set up XGBoost regression model
    xgb_model = xgb.XGBRegressor(eval_metric=r2_score, objective='reg:squarederror', booster='gbtree')
    xgb_model.fit(X_train, y_train)

    # iterate over all parameter combinations and use the best performer to fit
    logging.info('Commencing GridSearch...')
    logging.info('Using a %s-fold cross-validation' % k)
    xgb_iters = GridSearchCV(xgb_model, param_grid, cv=k, scoring=scoring, verbose=1, refit=True, return_train_score=True)
    xgb_iters.fit(X_train, y_train)

    cv_results_df = pd.DataFrame.from_dict(xgb_iters.cv_results_)
    logging.info('Done. Best params: %s' % xgb_iters.best_params_)
    logging.info('Best training score: %s = %s' % (scoring, xgb_iters.best_score_))

    out_list = [cv_results_df, xgb_iters.best_estimator_, xgb_iters.best_params_, xgb_iters.best_score_]

    return out_list


def test_metrics(y_test, prediction):
    """
    Calculate and print out model test metrics
    :param y_test: dependent variable test array
    :param prediction: model prediction of the y variable
    :return:
    """
    logging.info('--------- MODEL TEST PERFORMANCE METRICS ---------')
    r2 = r2_score(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    logging.info('R^2: %s' % r2)
    logging.info('Mean Squared Error: %s' % mse)

    return [r2, mse]


def model_test(X_test, y_test, best_estimator, best_params, out_folder):
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
    prediction = model.predict(X_test)
    plt.cla()
    logging.info('Applying model to test dataset...')

    # calculate test metrics
    r2 = test_metrics(y_test, prediction)[0]

    # Calculate the point density
    xy = np.vstack([prediction, y_test])
    z = gaussian_kde(xy)(xy)

    # make and format plot
    fig, ax = plt.subplots()
    ax.scatter(prediction, y_test, c=z, s=20)

    plt.title('XGBoost - Predicting daily mean NO2 concentrations')
    plt.plot(np.arange(0, 60, 0.1), np.arange(0, 60, 0.1), c='red')
    plt.xlim(0, np.max(prediction))
    plt.ylim(0, np.max(y_test))
    plt.xlabel('Predicted NO2 concentration')
    plt.ylabel('Actual daily NO2 concentration')
    plt.annotate(best_params, (0.2, 0.9), xycoords='subfigure fraction', fontsize='x-small')
    plt.annotate('R2 = %s' % round(r2, 2), (0.15, 0.8), xycoords='subfigure fraction', fontsize='large')

    # save figure
    fig_name = out_folder + '\\model_test.png'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    logging.info('Done. Prediction plot saved @ %s' % fig_name)

    return


def shap_analytics(model, X_train, out_folder):
    import shap
    shap_values = shap.TreeExplainer(model).shap_values(X_train)

    # plot both dot violin and bar plots to track feature importance
    #plt.tight_layout()
    #fig1 = shap.summary_plot(shap_values, X_train)
    #fig1.save(out_folder + '\\SHAP_dot_plot.png')
    #plt.cla()

    fig2 = shap.summary_plot(shap_values, X_train, plot_type="bar")
    fig2.save(out_folder + '\\SHAP_bar_plot.png')
    plt.cla()

    return logging.info('SHAP feature importance plots saved @ %s' % out_folder)


def plot_feature_importance(best_estimator, X_train, out_folder):
    """
    Plots feature importance for a model
    :param best_estimator: the best_estimator_ model selected during GridSearch (out_list[1]) or other model
    :param out_folder: folder where the plot is saved as a figure
    :return: shows plot
    """
    model = best_estimator
    plt.cla()
    logging.info('Plotting feature importance...')

    # plot feature importance
    x = range(len(model.feature_importances_))
    plt.bar(x, model.feature_importances_)
    plt.xticks(x, model.get_booster().feature_names, rotation=-45)
    plt.subplots_adjust(bottom=0.40)

    # save figure
    fig_name = out_folder + '\\model_feature_importance.png'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    logging.info('Done. Plot saved @ %s' % fig_name)
    plt.cla()

    # save SHAP feature importance plots
    #shap_analytics(model, X_train, out_folder)

    return logging.info('Done. All feature importance plots saved @ %s' % out_folder)


def plot_hyperparams(scoring_df, param_grid, out_folder):
    """
    This saves the model.cv_results_ item as a csv and saves plots of the distribution of scores for each parameter.
    :param cv_results_df: the model.cv_results_ item (out_list[0])
    :param param_grid: the param_grid dictionary with param name keys
    :param out_folder: a folder to save plots and .csv in (a sub-folder \\hyper_tuning is made_
    :return: none
    """
    # make folder to store hyper-parameter tuning
    hyp_dir = out_folder + '\\hyper_tuning'
    logging.info('Summarizing GridSearch hyper-parameters...')

    if not os.path.exists(hyp_dir):
        os.makedirs(hyp_dir)

    # get dictionary as pandas dataframe and save it as a csv
    scoring_df.head(n=20)
    score_csv = hyp_dir + '\\hyper_params_scoring.csv'
    scoring_df.to_csv(score_csv)
    logging.info('The model.cv_results_ converted to a .csv @ %s' % score_csv)

    # make a list of hyper parameters to iterate over
    hypers = list(param_grid.keys())
    param_cols = []
    for param in hypers:
        if param != 'booster':
            # get column key for dataframe and add to list
            col_key = 'param_%s' % param
            param_cols.append(col_key)
            ax = sns.boxenplot(x=col_key, y='mean_test_score', data=scoring_df)
            ax.figure.savefig(hyp_dir + '\\%s.png' % param)
            plt.cla()
    logging.info('Done. Plots made for each hyper-parameter @ %s' % hyp_dir)
    return


def train_and_run(in_csv, in_cols, params_list, test_prop, k=5):
    """
    Master function. Trains and tests an NO2 prediction XGBoost model using GridSearchCV
    :param in_csv: path of the csv containing independent and dependent variable columns (string)
    :param in_cols: independent variable column headers (list of strings)
    :param params_list: a list of ranges to test for XGBoost model parameters in the following order:
    [gamma_range, eta_range, lambda_range, colsample_range, max_depth_range]
    :param test_prop: the proportion of the dataset rows to exclude to final testing (float from 0 to 1)
    :param k: the number of K-folds used for cross-validation (integer, default is 5)
    :return: saves plots and logs @ csv_directory/MODEL_RUNS/Run#
    """

    # set up folders
    main_folder = os.path.dirname(in_csv)
    out_folder = prep_output(main_folder)

    # initiate logging in the model run folder
    init_logger(__file__, log_name=out_folder + '\\run_log.log')
    logging.info('Inputs variables: %s' % in_cols)

    # pull in data
    in_data = pd.read_csv(in_csv)
    in_data = in_data[in_cols]

    # set up parameter grid and print out grid nodes
    gammas, etas, lambdas, colsample_range, max_depths = params_list
    param_grid = {'gamma': gammas, 'eta': etas, 'reg_lambda': lambdas, 'colsample_bytree': colsample_range,
                  'max_depth': max_depths}
    for i in param_grid.keys():
        logging.info('Param: %s, testing: %s' % (i, param_grid[i]))

    # prepare model training inputs
    out = prep_input(in_data, in_cols, test_prop)
    X_df, Y_df = out[0]  # [0][0] is X dataframe, [0][1] is Y dataframe
    X_train, X_test, y_train, y_test = out[1]
    X_train.to_csv(out_folder + '\\X_train.csv')
    cross_cross(X_df, out_folder=out_folder)

    # use GridSearch CV to tune model hyper-parameters
    out_list = train_xgb(X_train, y_train, param_grid, k=k, scoring='r2')
    best_model = out_list[1]

    # plot model performance and feature importance
    model_test(X_test, y_test, best_model, out_list[2], out_folder)
    plot_feature_importance(best_model, X_train, out_folder)
    plot_hyperparams(out_list[0], param_grid, out_folder)

    # save model for predictions later
    joblib.dump(best_model, out_folder + '\\best_estimator.pkl')

    return


#  ########## SET XGBOOST PARAMETER RANGES ###########
CSV_DIR = r'C:\Users\xrnogueira\Documents\Data\NO2_stations'
main_csv = CSV_DIR + '\\master_monthly_no2_jan4.csv'
krig_csv = CSV_DIR + '\\master_monthly_no2_jan4_rk.csv'
#test_csv = CSV_DIR + '\\master_no2_daily_test_500_rows.csv'


keep_cols1 = ['mean_no2',  'sp', 'swvl1', 't2m', 'tp', 'u10', 'v10', 'blh', 'u100', 'v100', 'p_roads_1000',
                 's_roads_1700', 's_roads_3000', 'tropomi', 'pod_den_1100', 'Z_r', 'Z']
keep_cols2 = ['mean_no2', 'sp', 'swvl1', 't2m', 'tp', 'u10', 'v10', 'blh', 'u100', 'v100', 'p_roads_1000',
                 's_roads_1700', 's_roads_3000', 'tropomi', 'pod_den_1100', 'Z_r', 'Z', 'no2_krig']
keep_cols3 = ['mean_no2', 'sp', 't2m', 'tp', 'blh', 'tropomi', 'Z_r', 'Z']

drop_cols = ['u10', 'v10', 'swvl1', 'u100', 'v100', 'mean_no2']

keep_cols1 = [i for i in keep_cols1 if i not in drop_cols]
keep_cols2 = [i for i in keep_cols2 if i not in drop_cols]


gamma_range = list(np.arange(0, 1, 0.5))
eta_range = [round(i, 2) for i in list(np.arange(0.01, 0.31, 0.05))]
lambda_range = [round(i, 1) for i in list(np.arange(0.6, 1.4, 0.2))]
colsample_range = list(np.arange(0.5, 1.25, 0.25))
max_depth_range = list(np.arange(4, 9, 1))

params_list = [gamma_range, eta_range, lambda_range, colsample_range, max_depth_range]

if __name__ == "__main__":
    #train_and_run(main_csv, keep_cols1, params_list, test_prop=0.2)
    #train_and_run(krig_csv, keep_cols2, params_list, test_prop=0.2)
    train_and_run(main_csv, keep_cols3, params_list, test_prop=0.2)

    # for hardcoding SHAP
    #o_fol = r'C:\Users\xrnogueira\Documents\Data\NO2_stations\MODEL_RUNS\Run3'
    #X_train = pd.read_csv(o_fol + '\\X_train.csv')[keep_cols1]
    #saved_model = o_fol + '\\best_estimator.pkl'
    #model = joblib.load(saved_model)
    #shap_analytics(model, X_train, o_fol)



