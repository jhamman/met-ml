from glob import glob
import os.path

import numpy as np
import pandas as pd


# these missing in the metadata and were looked up using google earth
elevs = {'AR-Vir': 105.,
         'AU-Wac': 753.,
         'AR-SLu': 508.,
         'AU-Rig': 151.,
         'AU-Stp': 228.,
         'CN-Du2': 1321.,
         'JP-SMF': 213.,
         'AU-How': 39.,
         'AU-DaP': 71.,
         'CN-Sw2': 1445.,
         'AU-Dry': 176.,
         'AU-Emr': 175.,
         'CN-Din': 292.,
         'AU-DaS': 74.,
         'CN-Cng': 143.,
         'AU-Whr': 151.,
         'AU-Fog': 4.,
         'AU-RDF': 189.,
         'RU-Sam': 11.,
         'AU-Cum': 39.,
         'CN-Qia': 112.,
         'CN-Du3': 1313.,
         'CN-Ha2': 3198.,
         'CN-Cha': 767.,
         'AU-Gin': 51.,
         'AU-Ade': 76.,
         'CN-HaM': 4004.,
         'AU-GWW': 448.,
         'AU-Ync': 126.,
         'JP-MBF': 572.,
         'MY-PSO': 147.,
         'AU-TTE': 552.,
         'AU-ASM': 606.,
         'CN-Dan': 4313.,
         'AU-Cpr': 63.,
         'AU-Lox': 45.,
        }


def first_entry(entry):
    try:
#         print(entry)
        return entry.astype(float).values[0]
    except:
        return float(entry)
    
    
def extract_site_meta(meta, site):
    out = {}
    out['lat'] = first_entry(meta[site]['LOCATION_LAT'])
    out['lon'] = first_entry(meta[site]['LOCATION_LONG'])
    
    try:
        out['elev'] = first_entry(meta[site]['LOCATION_ELEV'])
    except:
        out['elev'] = elevs.get(site)
    return out


def get_training_sites(glob_path, thresh_days):

    # pick a list of sites with sufficiently long temporal records
    thresh = pd.Timedelta(thresh_days, 'D')  # ~10years

    paths = glob(glob_path)

    sites = []
    starts = []
    stops = []

    for f in paths:
        df = pd.read_csv(f)
        sites.append(os.path.split(f)[-1].split('_')[1])
        starts.append(df['TIMESTAMP'].values[0])
        stops.append(df['TIMESTAMP'].values[-1])


    site_df = pd.DataFrame(
        {'site': sites,
         'start': pd.to_datetime(starts, format='%Y%m%d'),
         'stop': pd.to_datetime(stops, format='%Y%m%d')
        }).set_index('site')
    site_df['dur'] = (site_df['stop'] - site_df['start'])
    train_sites = site_df[site_df.dur > thresh]
    
    return train_sites
