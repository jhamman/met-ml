import numpy as np
import pandas as pd
import xarray as xr

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


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
        "precip": FunctionTransformer(np.cbrt, validate=False),
        "elev": FunctionTransformer(elevation_scaler, validate=False),
        "lat": FunctionTransformer(latitude_scaler, validate=False),
        "doy": FunctionTransformer(day_of_year_scaler, validate=False),
        "t_min": StandardScaler(),
        "t_max": StandardScaler(),
        "SW_IN":  MinMaxScaler(),
        "LW_IN": MinMaxScaler(),
        "PA": MinMaxScaler(),
        "RH": MinMaxScaler(),
    }
    
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    df = pd.concat(dfs)
    
    transformers = {}
    for key in trans:
        transformers[key] = trans[key].fit(df[[key]])
    return transformers


def transform_df(transformers, df):
    out = pd.DataFrame(index=df.index)
    for key in df:
        if key in transformers:
            out[key] = transformers[key].transform(df[[key]])
        else:
            out[key] = df[key]
    return out


def save_transformers(transformers, file_path):
    # TODO: use ONNX for this
    with fsspec.open(file_path, mode='wb') as f:
        dump(transformers, f)


def plot_train_test_sites(train_sites, test_sites):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.scatter(train_sites.lon, train_sites.lat, transform=ccrs.PlateCarree(), label="Training Sites")
    ax.scatter(
        test_sites.lon,
        test_sites.lat,
        c="r",
        transform=ccrs.PlateCarree(),
        label="Validation Sites",
    )
    ax.set_global()
    ax.stock_img()
    ax.coastlines()
    ax.gridlines()
    ax.legend()
    plt.show()
    plt.close()

    
def train_test_split_based_on_lat(data, test_size, plot=True, random_state=None):
    """
    data is a pandas dataframe, splits based on stratified latitude 
    """
    all_sites = data[['Site', 'lat', 'lon']].drop_duplicates().reset_index(drop=True)
    all_sites['lat_rounded'] = all_sites.lat.round(-1)
    train_sites, test_sites = train_test_split(
        all_sites, 
        test_size=test_size, 
        stratify=all_sites['lat_rounded'], 
        random_state=random_state
    )
    
    if plot:
        plot_train_test_sites(train_sites, test_sites)
        
    train_data = data.loc[data.Site.isin(train_sites.Site.unique())].reset_index(drop=True)
    test_data = data.loc[data.Site.isin(test_sites.Site.unique())].reset_index(drop=True)
    
    return train_data, test_data


def scale_data(train, test):
    # fit transformer on train data 
    transformers = fit_transformers(train)
    
    # transform both train and test df 
    t_train = transform_df(transformers, train)
    t_test = transform_df(transformers, test)

    return t_train, t_test, transformers


def inverse_transform(transformers, df):
    out = pd.DataFrame(index=df.index)
    for key in df:
        out[key] = transformers[key].inverse_transform(df[[key]])
    return out


def make_lookback(df, variables, lookback=90):
    df = df[variables]  # sort columns
    coords = {'features': variables}
    da = xr.DataArray(df.values, dims=("samples", "features"), coords=coords)
    lba = da.rolling(samples=lookback).construct("lookback")
    lba.coords['lookback'] = np.linspace(-1 * (lookback - 1), 0, num=lookback, dtype=int)
    mask = lba.isnull().any(("lookback", "features"))
    return lba.where(~mask, drop=True).transpose("samples", "lookback", "features")


def get_features_and_labels(data, lookback=90, input_is_2D=True): 
    train_vars = ["precip", "t_min", "t_max", "doy", "lat", "elev"]
    target_vars = ["SW_IN", "LW_IN", "PA", "RH"]
    label_vars = ["Site", "TIMESTAMP_START"]
    all_vars = train_vars + target_vars + label_vars
    
    with_lookback = xr.concat(
        [make_lookback(sub_df, variables=all_vars, lookback=lookback) for site, sub_df in data.groupby('Site')],
        dim="samples",
    )
    
    X = with_lookback.sel(features=train_vars).chunk({'samples': 10000}).astype(float)
    y = with_lookback.sel(features=target_vars).sel(lookback=0).chunk({'samples': 10000}).astype(float)
    labels = with_lookback.sel(features=label_vars).sel(lookback=0).chunk({'samples': 10000})
    
    assert X.isnull().sum().values == 0
    assert y.isnull().sum().values == 0
    assert labels.isnull().sum().values == 0
    
    if input_is_2D:
        X = X.stack({'flatten_features': ['features', 'lookback']}).transpose("samples", "flatten_features")
    
    return X, y, labels
