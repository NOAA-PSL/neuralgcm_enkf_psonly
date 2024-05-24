import fsspec
import cftime
import gin
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import xarray
import os

import gcsfs
import fsspec
from netCDF4 import Dataset
from datetime import datetime, timedelta

from dinosaur import horizontal_interpolation
from dinosaur import pytree_utils
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
from dinosaur import xarray_utils
from neuralgcm import api
from dinosaur import scales
from neuralgcm import model_builder, physics_specifications
import sys

def getmean(diff,coslats):
    meancoslats = coslats.mean()
    return (coslats*diff).mean()/meancoslats

def splitdate(yyyymmddhh):
    """
 yyyy,mm,dd,hh = splitdate(yyyymmddhh)

 give an date string (yyyymmddhh) return integers yyyy,mm,dd,hh.
    """
    yyyy = int(yyyymmddhh[0:4])
    mm = int(yyyymmddhh[4:6])
    dd = int(yyyymmddhh[6:8])
    hh = int(yyyymmddhh[8:10])
    return yyyy,mm,dd,hh

expt_name = sys.argv[1]
init_date = sys.argv[2]
init_file_path = os.path.join(os.path.join('/work2/noaa/gsienkf/whitaker',expt_name),init_date)
init_file = os.path.join(init_file_path,'sanl_%s_fhr06_ensmean' % init_date)
init_year,init_month,init_day,init_hour = splitdate(init_date)

grav = scales.GRAVITY_ACCELERATION.magnitude
#print("reading from %s..." % init_file)
ncin = Dataset(init_file)
pfull = (ncin['pfull'][:]).astype(int)
nlevz500 = (pfull.tolist()).index(500)
#print(nlevz500)
z500 = ncin['z'][0,nlevz500,::-1,:]
#print(pfull[nlevz500],z500.shape,z500.min(),z500.max())
z500 = z500*grav
latsin = ncin['grid_yt'][::-1]
lonsin = ncin['grid_xt'][:]
ncin.close()
z500_ds = xarray.Dataset(data_vars=
        dict(
        geopotential=(["latitude", "longitude"], z500),
    ),
        coords=dict(
        longitude=("longitude", lonsin),
        latitude=("latitude", latsin),
    ))
#print(z500_ds)

# interpolate to 1-deg grid
new_lat = np.arange(-90,90.1,1.0)
new_lon = np.arange(0,360,1.0)
z500ds_1deg = z500_ds.interp(latitude=new_lat, longitude=new_lon, assume_sorted=True, kwargs={"fill_value": "extrapolate"})
#z500ds_1deg = z500ds_1deg.transpose('latitude','longitude')
#print(z500ds_1deg)
z500ds_1deg = z500ds_1deg['geopotential']
z500 = z500ds_1deg.values/grav
#print(z500.shape, z500.min(), z500.max())

# get era5 data
lats, lons = np.meshgrid(new_lat, new_lon)
lats = lats.transpose()
coslats = np.cos(np.radians(lats))
lat1 = 90; lat2 = 20. # NH
latmask_nh = np.logical_or(lats > lat1, lats < lat2)
coslats_nh = np.ma.masked_array(coslats, mask=latmask_nh)
lat1 = -20; lat2 = -90 # SH
latmask_sh = np.logical_or(lats > lat1, lats < lat2)
coslats_sh = np.ma.masked_array(coslats, mask=latmask_sh)
def gs_get_mapper(path):
  fs = gcsfs.GCSFileSystem(project='neuralgcm')
  return fs.get_mapper(path)
era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5_ds = xarray.open_zarr(gs_get_mapper(era5_path), chunks=None)
full_era5_ds = full_era5_ds['geopotential']
start_time = '%04i-%02i-%02i %02i:00:00' % (init_year,init_month,init_day,init_hour)
era5_ds = full_era5_ds.sel(time=slice(start_time,start_time))
era5_ds = era5_ds.thin(time=6) # thin to every 6 hours
era5_ds = era5_ds.sel(level=500) # level 21 is 500 hpa
if not (era5_ds['latitude'].diff('latitude') > 0).all():
  # Ensure ascending latitude
  era5_ds = era5_ds.isel(latitude=slice(None, None, -1))
era5_ds_1deg = era5_ds.interp(latitude=new_lat, longitude=new_lon, assume_sorted=True)
#print(era5_ds_1deg.shape)

z500_1deg = (z500ds_1deg.values/grav).squeeze()
z500era_1deg = (era5_ds_1deg.values/grav).squeeze()
#print(z500_1deg.shape, z500_1deg.min(), z500_1deg.max())
#print(z500era_1deg.shape, z500era_1deg.min(), z500era_1deg.max())
diff = z500era_1deg-z500_1deg
print(init_date, np.sqrt(getmean(diff**2,coslats)), np.sqrt(getmean(diff**2,coslats_nh)), np.sqrt(getmean(diff**2,coslats_sh)))
