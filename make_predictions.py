"""
This contains a workflow to make a prediction map using lat/long csv for a urban point cloud
@xaviernogueira
"""
import logging
import os.path

import joblib
import pandas as pd

from useful_functions import init_logger


def apply_predictions(points_csv, saved_model, variables=None):
    """
    Apply model on the csv and create a csv for each month
    :param points_csv: a csv with a pointid field containing monthly variable names across the prediction AOI
    :param saved_model: a joblib saved ML model with headers matching the points_csv (minus _m#MONTH_NUMBER#)
    :param variables: (optional) a list of variables to exclusively use (subset of all variables)
    :return:
    """
    # set up logger and inputs/outputs
    init_logger(__file__)
    logging.info('Applying predictions')
    master_df = pd.read_csv(points_csv)
    master_df.sort_values('pointid', axis=1)

    out_folder = os.path.dirname(points_csv)

    # bring in saved model
    model = joblib.load(saved_model)

    # iterate over months
    for m in range(1, 13):
        master_copy = master_df.copy()
        m_str = 'm%s_' % m
        static_cols =  ['Z', 'Z_r']
        # 'p_roads_1000', 's_roads_1700', 's_roads_3000', 'pod_den_1100' PUT BACK FOR REAL RUN

        # grab monthly columns and change to match model headers
        if variables is None:
            m_cols = [i for i in list(master_copy.columns) if m_str in i]

        else:
            # remove non-specified variables if desired
            m_cols1 = [i for i in list(master_copy.columns) if m_str in i]
            m_cols = [i for i in m_cols1 if i.replace(m_str, '') in variables]
            static_cols = [i for i in static_cols if i in variables]

        keep_cols = m_cols + static_cols

        X_map = master_copy[keep_cols]

        for col in keep_cols:
            X_map.rename(columns={col: col.replace(m_str, '')}, inplace=True)

        # use model to make prediction and compare with tropomi
        prediction = model.predict(X_map)
        master_copy['no2_predict'] = prediction
        master_copy['vs_tropomi'] = master_copy['no2_predict'] - master_copy['tropomi']

        # save as csv
        master_copy.to_csv(out_folder + '%s_prediction.csv' % m_str)

    return


if __name__ == "__main__":
    model_file = r'C:\Users\xrnogueira\Documents\Data\NO2_stations\MODEL_RUNS\Run3\best_estimator.pkl'
    raster_dir = r'C:\Users\xrnogueira\Documents\Data\Chicago_prediction'
    raster_dict = {}
    #all_vars = ['sp', 't2m', 'tp', 'blh', 'p_roads_1000', 's_roads_1700', 's_roads_3000', 'tropomi', 'pod_den_1100', 'Z_r', 'Z']

    points_csv = []
    test_model = r'C:\Users\xrnogueira\Documents\Data\NO2_stations\MODEL_RUNS\Run7\best_estimator.pkl'
    apply_predictions(points_csv, test_model)