import math
import random

import numpy as np
import pandas as pd
import xarray as xr
import dask
import intake

from utils import get_training_sites, extract_site_meta


@dask.delayed
def load_x_df(entry, meta):
    df = entry.read().set_index('TIMESTAMP_START')
    
    out = df[['P']].resample('1D').sum()
    out['t_min'] = df['TA_F'].resample('1D').min()
    out['t_max'] = df['TA_F'].resample('1D').max()

    out['t'] = make_cyclic_doy(out.index.dayofyear)
    out['lat'] = np.sin(np.radians(meta['lat']))
    out['elev'] = meta['elev']

    return out


@dask.delayed
def load_y_df(entry):
    # Fetch a batch of y data
    out = entry.read().set_index('TIMESTAMP')[['SW_IN_F']]
    return out


def load_fluxnet(compute=True):
    '''return lists of x and y data'''
    
    cat = intake.Catalog('./fluxnet/catalog.yml')
    
    glob_path = './fluxnet/FLX_*_FLUXNET2015_FULLSET_DD_*.csv'
    thresh_days = 365
    train_sites = get_training_sites(glob_path, thresh_days)
    train_sites.describe()

    all_site_meta = pd.read_excel(
        './fluxnet/FLX_AA-Flx_BIF_LATEST.xlsx').set_index(
            ['SITE_ID', 'VARIABLE'])['DATAVALUE']
    meta = {key: extract_site_meta(all_site_meta, key) for key in train_sites.index}
    
    x_data = [load_x_df(cat['subdaily'](site=site), site_meta)
          for site, site_meta in meta.items()]
    
    y_data = [load_y_df(cat['daily'](site=site))
              for site, site_meta in meta.items()]
    if compute:
        x_data = dask.compute(*x_data)
        y_data = dask.compute(*y_data)
    
    return x_data, y_data, meta


def make_cyclic_doy(doy):
    # TODO: consider updating this to handle leap years
    return np.cos((doy - 1) / 365 * 2 * np.pi)


def make_lookback(vals, features, lookback=90):
    out_shape = (vals.shape[0] - lookback, lookback, vals.shape[1])
    out = np.full(out_shape, np.nan)
    for i in range(lookback):
        start = lookback - i
        stop = len(vals) - i
        out[:, i, :] = vals[start:stop]
        
    out = xr.DataArray(out, dims=('samples', 'lookback', 'features'),
                       coords={'features': features})
    return out
