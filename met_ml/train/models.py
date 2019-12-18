import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler


def elevation_scaler(x, feature_range=(0, 1), data_range=(-420, 8848)):
    '''MinMaxScaler for elevations on Earth'''
    fmin, fmax = feature_range
    dmin, dmax = data_range
    scale = (fmax - fmin) / (dmax - dmin)
    x_scaled = scale * x + fmin - dmin * scale
    return x_scaled


def latitude_scaler(x):
    return np.sin(np.radians(x))


def day_of_year_scaler(x):
    return np.cos((x - 1) / 365.25 * 2 * np.pi)



def fit_transformers(dfs):
    """takes a list of dataframes, returns a fit transformer"""
    
    trans = {
        "P": FunctionTransformer(np.cbrt, validate=False),
        "elev": FunctionTransformer(elevation_scaler, validate=False),
        "lat": FunctionTransformer(latitude_scaler, validate=False),
        "t": FunctionTransformer(day_of_year_scaler, validate=False),
        "t_min": StandardScaler(),
        "t_max": StandardScaler(),
        "SW_IN_F":  MinMaxScaler(),
        "LW_IN_F": MinMaxScaler(),
        "PA_F": MinMaxScaler(),
        "RH": MinMaxScaler(),
    }
    
    df = pd.concat(dfs)
    
    transformers = {}
    for key in df.columns:
        transformers[key] = trans[key].fit(df[[key]])
    return transformers


def transform_df(transformers, df):
    out = pd.DataFrame(index=df.index)
    for key in df:
        out[key] = transformers[key].transform(df[[key]])
    return out