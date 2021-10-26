"""
Quick and useful functions for data science: pre-processing, plotting, etc.

author @xaviernogueira
"""
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader


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
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.gridlines(projection, draw_labels=True, alpha=0, linestyle='--')


