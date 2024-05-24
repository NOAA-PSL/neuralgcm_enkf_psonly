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

# Helper function to load files from Google Cloud Storage
def gs_open(path):
  fs = gcsfs.GCSFileSystem(project='neuralgcm')
  return fs.open(path, 'rb')


era5path = '/work2/noaa/gsienkf/whitaker/python/run_neuralgcm'
#model_type = 'deterministic_0_7'
#model_type = 'stochastic_1_4'

expt_name = sys.argv[1]
init_date = sys.argv[2]
model_type = sys.argv[3]
model_name = 'neural_gcm_%s_deg_v0.pkl' % model_type
init_file_path = os.path.join(os.path.join('/work2/noaa/gsienkf/whitaker',expt_name),init_date)
init_file = os.path.join(init_file_path,'sanl_%s_fhr06_ensmean' % init_date)

#with gs_open(f'gs://gresearch/neuralgcm/03_04_2024/{model_name}') as f:
#    raise SystemExit
with open(f'{model_name}','rb') as f:
  ckpt = pickle.load(f)

neuralgcm_vars = [
    'u_component_of_wind',
    'v_component_of_wind',
    'geopotential',
    'temperature',
    'specific_humidity',
    'specific_cloud_ice_water_content',
    'specific_cloud_liquid_water_content',
    'sea_surface_temperature',
    'geopotential_at_surface',
    'sea_ice_cover'
]

neural_gcm_model = api.PressureLevelModel.from_checkpoint(ckpt)
physics_specs = neural_gcm_model._structure.specs.physics_specs
model_specs = neural_gcm_model._structure.specs
tscale = physics_specs.dimensionalize(1, scales.units.hour).magnitude
dt = tscale*model_specs.dt


if model_type[-3:] == '0_7':
    eval_ds = xarray.open_dataset(os.path.join(era5path,'era5_init_0p7deg_%s.nc' % init_date[0:8]))
elif model_type[-3:] == '1_4':
    eval_ds = xarray.open_dataset(os.path.join(era5path,'era5_init_1p4deg_%s.nc' % init_date[0:8]))
else:
    raise ValueError("unsupported resolution")
init_year,init_month,init_day,init_hour = splitdate(init_date)
init_time = datetime(init_year,init_month,init_day,init_hour)
demo_start_time = '%04i-%02i-%02i %02i:00:00' % (init_year,init_month,init_day,init_hour)

# Selecting input data and rollout parameters

start_time = '%04i-%02i-%02i %02i:00:00' % (init_year,init_month,init_day,init_hour)
input_data_ds = eval_ds.sel(time=slice(start_time,start_time))  # keep time axis for `forcings`.
grav = scales.GRAVITY_ACCELERATION.magnitude
z500era = input_data_ds['geopotential'].sel(level=500).values/grav
levels = input_data_ds.level.values.tolist()
lats = input_data_ds.latitude.values
lons = input_data_ds.longitude.values
lats, lons = np.meshgrid(lats, lons)
coslats = np.cos(np.radians(lats))
data_dict, forcings = neural_gcm_model.data_from_xarray(input_data_ds)
# slicing time=0 for initializing the model (encode expects no time axis)
inputs, input_forcings = pytree_utils.slice_along_axis(
    (data_dict, forcings), axis=0, idx=0)
ref_time = neural_gcm_model._structure.specs.aux_features['reference_datetime']
tscale = physics_specs.dimensionalize(1, scales.units.hour).magnitude
press_scale = physics_specs.dimensionalize(1.,scales.units.pascal).magnitude
wind_scale = physics_specs.dimensionalize(1.,scales.units.meter_per_second).magnitude

initial_state = neural_gcm_model.encode(
    inputs, forcings=input_forcings,rng_key=jax.random.PRNGKey(42))
state = initial_state
ref_temp = neural_gcm_model._structure.specs.aux_features['ref_temperatures']

# read in GFS history file
ncin = Dataset(init_file)
ncin.set_auto_mask(False)
pressfc = ncin['pressfc'][:]
pressfc = np.moveaxis(pressfc, -1, -2) # flip pos of lats and lons
pressfc = pressfc[:,:,::-1]/press_scale # reverse lats, scale
ugrd = ncin['ugrd'][0]
ugrd = np.moveaxis(ugrd, -1, -2)
ugrd = ugrd[:,:,::-1]/wind_scale
vgrd = ncin['vgrd'][0]
vgrd = np.moveaxis(vgrd, -1, -2)
vgrd = vgrd[:,:,::-1]/wind_scale
tmp = ncin['tmp'][0]
tmp = np.moveaxis(tmp, -1, -2)
tmp = tmp[:,:,::-1]
tmp = tmp-ref_temp[:,np.newaxis,np.newaxis]
spfh = ncin['spfh'][0]
spfh = np.moveaxis(spfh, -1, -2)
spfh = spfh[:,:,::-1]
clwmr = ncin['clwmr'][0]
clwmr = np.moveaxis(clwmr, -1, -2)
clwmr = clwmr[:,:,::-1]
icmr = ncin['icmr'][0]
icmr = np.moveaxis(icmr, -1, -2)
icmr = icmr[:,:,::-1]
ncin.close()
state.state.log_surface_pressure = neural_gcm_model.model_coords.horizontal.to_modal(jnp.log(pressfc))
state.state.temperature_variation = neural_gcm_model.model_coords.horizontal.to_modal(jnp.array(tmp))
vort, div = spherical_harmonic.uv_nodal_to_vor_div_modal(neural_gcm_model.model_coords.horizontal, jnp.array(ugrd), jnp.array(vgrd)) 
state.state.vorticity = vort; state.state.divergence = div
state.state.tracers['specific_humidity'] = neural_gcm_model.model_coords.horizontal.to_modal(jnp.array(spfh))
state.state.tracers['specific_cloud_liquid_water_content'] = neural_gcm_model.model_coords.horizontal.to_modal(jnp.array(clwmr))
state.state.tracers['specific_cloud_ice_water_content'] = neural_gcm_model.model_coords.horizontal.to_modal(jnp.array(icmr))

# run one time step
final_state, predictions = neural_gcm_model.unroll(
    state,
    forcings=forcings,
    steps=1,
    start_with_input=True,
)
#valid_time = (predictions['sim_time'].item()*tscale*np.timedelta64(1,'h') + ref_time).item()
#print(valid_time)
nlev = levels.index(500)
z500 = predictions['geopotential'][0,nlev,...]/grav
diff = z500era-z500
#print(init_date, np.sqrt(getmean(diff**2,coslats)))

predictions_ds = neural_gcm_model.data_to_xarray(predictions, times=np.arange(1))
z500_ds = predictions_ds['geopotential'].sel(level=500)
#print(z500_ds)

# interpolate to 1-deg grid
new_lat = np.arange(-90,90.1,1.0)
new_lon = np.arange(0,360,1.0)
if not (z500_ds['latitude'].diff('latitude') > 0).all():
  # Ensure ascending latitude
  z500_ds = z500_ds.isel(latitude=slice(None, None, -1))
z500ds_1deg = z500_ds.interp(latitude=new_lat, longitude=new_lon, assume_sorted=True, kwargs={"fill_value": "extrapolate"})
z500ds_1deg = z500ds_1deg.transpose('time','latitude','longitude')
#print(z500ds_1deg)
z500ds_1deg.to_netcdf(os.path.join(init_file_path,'z500_%s.nc' % init_date),'w')

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
iy,im,id,ih = splitdate(init_date)
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
