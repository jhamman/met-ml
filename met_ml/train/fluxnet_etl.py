import math
import os.path
import random
from glob import glob

import dask
import intake
import numpy as np
import pandas as pd
import xarray as xr
from joblib import dump, load

train_vars = ["precip", "t_min", "t_max"]
meta_vars = ["doy", "lat", "lon", "elev"]
target_vars = ["SW_IN_F_MDS", "LW_IN_F_MDS", "PA", 'RH']
era_vars = ['SW_IN_ERA', 'LW_IN_ERA', 'PA_ERA', 'RH_ERA']

# these missing in the metadata and were looked up using google earth
elevs = {
    "AR-Vir": 105.0,
    "AU-Wac": 753.0,
    "AR-SLu": 508.0,
    "AU-Rig": 151.0,
    "AU-Stp": 228.0,
    "CN-Du2": 1321.0,
    "JP-SMF": 213.0,
    "AU-How": 39.0,
    "AU-DaP": 71.0,
    "CN-Sw2": 1445.0,
    "AU-Dry": 176.0,
    "AU-Emr": 175.0,
    "CN-Din": 292.0,
    "AU-DaS": 74.0,
    "CN-Cng": 143.0,
    "AU-Whr": 151.0,
    "AU-Fog": 4.0,
    "AU-RDF": 189.0,
    "RU-Sam": 11.0,
    "AU-Cum": 39.0,
    "CN-Qia": 112.0,
    "CN-Du3": 1313.0,
    "CN-Ha2": 3198.0,
    "CN-Cha": 767.0,
    "AU-Gin": 51.0,
    "AU-Ade": 76.0,
    "CN-HaM": 4004.0,
    "AU-GWW": 448.0,
    "AU-Ync": 126.0,
    "JP-MBF": 572.0,
    "MY-PSO": 147.0,
    "AU-TTE": 552.0,
    "AU-ASM": 606.0,
    "CN-Dan": 4313.0,
    "AU-Cpr": 63.0,
    "AU-Lox": 45.0,
    "AU-Rob": 710.0,
}


sites_to_skip = [
    "CA-Man",  # missing RH
    "DE-RuR",  # missing RH
    "CA-Man",  # missing RH
    "DE-RuR",  # missing RH
    "DE-RuS",  # missing RH
    "MY-PSO",  # missing RH
#     "CN-Cha",  # found nans in df
#     "CN-Dan",  # found nans in df
#     "CN-Din",  # found nans in df
#     "CN-Qia",  # found nans in df
#     "DK-ZaH",  # found nans in df
#     "FI-Lom",  # found nans in df
#     "IT-Isp",  # found nans in df
#     "IT-SR2",  # found nans in df
#     "US-Me5",  # found nans in df
#     "US-PFa",  # found nans in df
]


def get_fluxnet(cat, all_site_meta, from_cache=True):
    """load the fluxnet dataset"""
    if not from_cache:
        # use dask to speed things up
        fluxnet_df = load_fluxnet(cat, all_site_meta)

#         dump(x_data_computed, "./etl_data/x_data_computed.joblib")
#         dump(y_data_computed, "./etl_data/y_data_computed.joblib")
#         dump(meta, "./etl_data/meta.joblib")
    else:
        x_data_computed = load("../data/etl/x_data_computed.joblib")
        y_data_computed = load("../data/etl/y_data_computed.joblib")
        meta = load("../data/etl/meta.joblib")

    return fluxnet_df


def calculate_vp_sat(temp_celsius):
    """calculates vp_sat in hpa from temp in celsius"""
    temp_rankine = temp_celsius * 9 / 5 + 491.67
    A = -1.0440397 * 10**4
    B = -11.29465
    C = -2.7022355 * 10**-2
    D = 1.289036 * 10**-5
    E = -2.4780681 * 10**-9
    F = 6.5459673
    expo = (
        A / temp_rankine 
        + B 
        + C * temp_rankine 
        + D * temp_rankine ** 2 
        + E * temp_rankine ** 3
        + F * np.log(temp_rankine)
    )
    vp_sat = np.exp(expo)
    return vp_sat * 68.9476
    

@dask.delayed
def load_fluxnet_site(entry):
    try:
        df = entry.read()
        df.index = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
        
        # replace negative Precip and Relative humidity to nans 
        df.replace(-9999, np.nan, inplace=True)
        df.loc[df.P < 0, 'P'] = np.nan 
        df.loc[((df.RH < 0) | (df.RH > 100)), 'RH'] = np.nan 
        
        # calculate RH from VPD for ERA data
        vp_sat = calculate_vp_sat(df.TA_ERA)
        df['RH_ERA'] = (1 - df.VPD_ERA / vp_sat) * 100.

        out = pd.DataFrame()
        out['precip'] = df["P"].resample("1D").sum()
        out["t_min"] = df["TA_F_MDS"].resample("1D").min()
        out["t_max"] = df["TA_F_MDS"].resample("1D").max()
        out[target_vars] = df[target_vars].resample("1D").mean()
        out[era_vars] = df[era_vars].resample('1D').mean()
        return out
    except:
        return None


def add_meta(df, meta):
    df["doy"] = df.index.dayofyear
    df["lat"] = meta["lat"]
    df["lon"] = meta["lon"]
    df["elev"] = meta["elev"]
    return df



def get_meta(all_site_meta):
    all_sites = all_site_meta.index.get_level_values(0).unique()
    meta = {
        key: extract_site_meta(all_site_meta, key)
        for key in all_sites
        if key not in sites_to_skip
    }
    return meta


def load_fluxnet(cat, all_site_meta):
    """return lists of x and y data"""
    
    print('getting meta data')
    meta = get_meta(all_site_meta)
    meta_df = pd.DataFrame.from_dict(meta, orient="index")
    
    print('getting all jobs')
    site_data = {}
    for site, site_meta in meta.items():
        site_data[site] = load_fluxnet_site(cat["raw_fullset"](station=site.lower(), kind='fullset', freq='hh'))
    
    print('computing')
    site_data = dask.compute(site_data)[0]

    print('prep output')
    out = {}
    var_names = train_vars + target_vars + era_vars
    for name, df in site_data.items():

        if df is not None:
            out[name] = add_meta(df.loc[:, var_names], meta[name])
        else:
            print(f'failed to read {name}, look into this...')
    
    print('concat')
    df = pd.concat(out.values(), keys=out.keys())
    df.index.names = ['Site', 'TIMESTAMP_START']

    return df


def make_cyclic_doy(doy):
    # TODO: consider updating this to handle leap years
    return np.cos((doy - 1) / 365 * 2 * np.pi)


def first_entry(entry):
    try:
        return entry.astype(float).values[0]
    except:
        return float(entry)


def extract_site_meta(meta, site):
    out = {}
    out["lat"] = first_entry(meta[site]["LOCATION_LAT"])
    out["lon"] = first_entry(meta[site]["LOCATION_LONG"])

    try:
        out["elev"] = first_entry(meta[site]["LOCATION_ELEV"])
    except:
        try:
            out["elev"] = elevs[site]
        except KeyError:
            print(f"failed to get elevation for {site}")
    return out


def get_training_sites(glob_path, thresh_days):

    # pick a list of sites with sufficiently long temporal records
    thresh = pd.Timedelta(thresh_days, "D")  # ~10years

    paths = glob(glob_path)

    sites = []
    starts = []
    stops = []

    for f in paths:
        df = pd.read_csv(f)
        sites.append(os.path.split(f)[-1].split("_")[1])
        starts.append(df["TIMESTAMP"].values[0])
        stops.append(df["TIMESTAMP"].values[-1])

    site_df = pd.DataFrame(
        {
            "site": sites,
            "start": pd.to_datetime(starts, format="%Y%m%d"),
            "stop": pd.to_datetime(stops, format="%Y%m%d"),
        }
    ).set_index("site")
    site_df["dur"] = site_df["stop"] - site_df["start"]
    train_sites = site_df[site_df.dur > thresh]

    return train_sites
