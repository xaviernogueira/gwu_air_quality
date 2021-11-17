import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def train_xgb(X_train, y_train, gammas, etas, lambdas, min_child_weights, max_depths, scoring='r2'):
    """

    :param X_train: dataframe or XDarray with independent variable training columns
    :param y_train: dataframe or XDarray with dependent variable training columns
    :param gammas:
    :param etas:
    :param lambdas:
    :param min_child_weights:
    :param max_depths:
    :param scoring: a scikit-learn scorer string (default is r2)
    :return: a list containing [model.cv_results_, model.best_estimator_, model.best_params_, model.best_score_]
    """
    # set up parameter grid and scorers
    param_grid = {'gamma': gammas, 'eta': etas, 'reg_lambda': lambdas,
                  'min_child_weight': min_child_weights,
                  'max_depth': max_depths, 'booster': ['gbtree']}
    SCORERS = {'r2': make_scorer(r2_score), 'neg_mean_squared_error': make_scorer(mean_squared_error)}
    SCORERS_str = list(SCORERS.keys())

    # set up XGBoost regressor model
    xgb_model = xgb.XGBRegressor('reg:squarederror')
    xgb_model.fit(X_train, y_train)

    # iterate over all parameter combinations and use the best performer to fit
    xgb_iters = GridSearchCV(xgb_model, param_grid, cv=5, scoring=scoring, verbose=1, refit=True, return_train_score=True)
    xgb_iters.fit(X_train, y_train)

    grid_results = pd.from_dict(xgb_iters.cv_results_)
    print('Best params: %s, %s: %s' % (xgb_iters.best_params_, scoring, xgb_iters.best_score_))

    out_list = [grid_results, xgb_iters.best_estimator_, xgb_iters.best_params_, xgb_iters.best_score_]

    return out_list


####################################
gamma_range = list(np.arange(0, 2, 0.5))
eta_range = list(np.arange(0.05, 0.5, 0.05))
lambda_range = list(np.arange(0.6, 1.4, 0.2))
min_child_weight_range = list(np.arange(1, 21, 5))
max_depth_range = list(np.arange(4, 7, 1))