import jax.numpy as jnp
import numpy as np

import jax, functools, dataclasses, gcsfs, fsspec, cftime, pickle, xarray, sys
from netCDF4 import Dataset
from datetime import datetime, timedelta

from dinosaur import spherical_harmonic, horizontal_interpolation, vertical_interpolation
from dinosaur import xarray_utils, scales
from neuralgcm import api, orographies

import haiku as hk

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

model_type = sys.argv[1]
model_name = 'neural_gcm_dynamic_forcing_%s_deg.pkl' % model_type
init_date = sys.argv[2]
nmem = int(sys.argv[3])
rng_seed = int(sys.argv[4])
if len(sys.argv) > 5:
    init_file = sys.argv[5]
else:
    init_file = False

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

model = api.PressureLevelModel.from_checkpoint(ckpt)

if model_type[-3:] == '0_7':
    eval_era5 = xarray.open_dataset('era5_init_0p7deg_%s.nc' % init_date[0:8])
elif model_type[-3:] == '1_4':
    eval_era5 = xarray.open_dataset('era5_init_1p4deg_%s.nc' % init_date[0:8])
else:
    raise ValueError("unsupported resolution")
init_year,init_month,init_day,init_hour = splitdate(init_date)
init_time = datetime(init_year,init_month,init_day,init_hour)
valid_time = init_time + timedelta(hours=6)
valid_date = valid_time.strftime('%Y%m%d%H')

# Selecting input data and rollout parameters

inner_steps = 3  # save model outputs once every 6 hours
timedelta = np.timedelta64(1, 'h') * inner_steps
outer_steps = 3  # total of 9 hours
print('time step = ',model.timestep)
nsteps_per_timedelta = api._calculate_sub_steps(model.timestep,timedelta)
print('time steps per output time',nsteps_per_timedelta,timedelta)
times = (np.arange(outer_steps) * inner_steps)  # time axis in hours

start_time = '%04i-%02i-%02i %02i:00:00' % (init_year,init_month,init_day,init_hour)
print(start_time)
input_data_ds = eval_era5.sel(time=slice(start_time,start_time))  # keep time axis for `forcings`.
inputs = model.inputs_from_xarray(input_data_ds.isel(time=0))
#dict_keys(['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content', 'sim_time'])
input_forcings = model.forcings_from_xarray(input_data_ds.isel(time=0))

grav = scales.GRAVITY_ACCELERATION.magnitude

if init_file:
    print("initializing from %s..." % init_file)
    ncin = Dataset(init_file)
    ncin.set_auto_mask(False)
    ps = ncin['pressfc'][:,::-1,:]
    ps = model.to_nondim_units(ps, units='pascals')
    ps = np.moveaxis(jnp.array(ps),-1,-2)
    u = ncin['ugrd'][0,:,::-1,:]
    v = ncin['vgrd'][0,:,::-1,:]
    t = ncin['tmp'][0,:,::-1,:]
    z = ncin['z'][0,:,::-1,:]
    q = ncin['spfh'][0,:,::-1,:]
    clmr = ncin['clwmr'][0,:,::-1,:]
    icmr = ncin['icmr'][0,:,::-1,:]
    ncin.close()
    inputs['geopotential'] = np.moveaxis(jnp.array(grav*z),-1,-2)
    inputs['temperature'] = np.moveaxis(jnp.array(t),-1,-2)
    inputs['specific_humidity'] = np.moveaxis(jnp.array(q),-1,-2)
    inputs['u_component_of_wind'] = np.moveaxis(jnp.array(u),-1,-2)
    inputs['v_component_of_wind'] = np.moveaxis(jnp.array(v),-1,-2)
    inputs['specific_cloud_ice_water_content'] = np.moveaxis(jnp.array(icmr),-1,-2)
    inputs['specific_cloud_liquid_water_content'] = np.moveaxis(jnp.array(clmr),-1,-2)

rng_key = jax.random.PRNGKey(rng_seed)
state = model.encode(
    inputs, forcings=input_forcings, rng_key=rng_key)
if init_file:
    # use ps updated by EnKF (over-riding ps from encoder)
    state.state.log_surface_pressure = model.model_coords.horizontal.to_modal(jnp.log(ps))
    #pass # use ps from encoder inferred from updated geopotential height
else:
    print("initializing from ERA5")

lats1d = model._structure.specs.aux_features['xarray_dataset']['latitude'].values
lons1d = model._structure.specs.aux_features['xarray_dataset']['longitude'].values
pfull_arr=input_data_ds['level'].values
nlevs = len(pfull_arr)
phalf_arr=np.empty(nlevs+1,np.float32)
phalf_arr[0]=0.5
for k in range(1,nlevs):
    phalf_arr[k]=pfull_arr[k-1]+0.5*(pfull_arr[k]-pfull_arr[k-1])
phalf_arr[nlevs] = pfull_arr[nlevs-1] + 0.5*(pfull_arr[nlevs-1]-pfull_arr[nlevs-2])

z = inputs['geopotential'];  z = np.moveaxis(z/grav,-1,-2)
t = inputs['temperature']; t =  np.moveaxis(t,-1,-2)
q = inputs['specific_humidity']; q = np.moveaxis(q,-1,-2)
u = inputs['u_component_of_wind']; u  = np.moveaxis(u,-1,-2)
v = inputs['v_component_of_wind']; v = np.moveaxis(v,-1,-2)
icmr = inputs['specific_cloud_ice_water_content']; icmr = np.moveaxis(icmr,-1,-2)
clmr = inputs['specific_cloud_liquid_water_content']; clmr = np.moveaxis(clmr,-1,-2)
ps = np.exp(model.model_coords.horizontal.to_nodal(state.state.log_surface_pressure))
ps = model.from_nondim_units(ps, units='pascals')
ps = np.moveaxis(ps, -1, -2)

# use learned orography
if model_type[-3:] == '0_7':
    nwaves=254
    #orogpath='stochastic_modular_step_model/~/dimensional_learned_weatherbench_to_primitive_with_memory_encoder/~/learned_weatherbench_to_primitive_encoder_1/~/learned_orography'
    orogpath='stochastic_modular_step_model/~/stochastic_physics_parameterization_step/~/custom_coords_corrector/~/dycore_with_physics_corrector/~/learned_orography'
    correction_scale=2.e-6
else:
    nwaves=126
    orogpath='stochastic_modular_step_model/~/stochastic_physics_parameterization_step/~/custom_coords_corrector/~/dycore_with_physics_corrector/~/learned_orography'
    correction_scale=1.e-5
@hk.transform
def get_orography():
  base_orography = functools.partial(
      orographies.FilteredCustomOrography,
      orography_data_path=None,
      renaming_dict=dict(longitude='lon', latitude='lat'),
  )
  orography_coords = dataclasses.replace(
        model.model_coords,
        horizontal=spherical_harmonic.Grid.with_wavenumbers(longitude_wavenumbers=nwaves)
  )
  return orographies.LearnedOrography(
      orography_coords,
      model._structure.specs.dt,
      model._structure.specs.physics_specs,
      model._structure.specs.aux_features,
      base_orography_module=base_orography,
      correction_scale=correction_scale,
  )()

dycore_coords = spherical_harmonic.Grid.with_wavenumbers(longitude_wavenumbers=nwaves)
learned_correction = model.params[orogpath]['orography']
learned_orography_modal = get_orography.apply(
    {'learned_orography': {'orography': learned_correction}}, rng=None
)
hgtsfcin = model.from_nondim_units(
    dycore_coords.to_nodal(learned_orography_modal), units='meters'
)
#hgtsfcin = dycore_coords.to_nodal(learned_orography_modal)
print('hgtsfcin shape/min/max = ',hgtsfcin.shape, hgtsfcin.min(), hgtsfcin.max())
regridder = horizontal_interpolation.BilinearRegridder(dycore_coords, model.model_coords.horizontal)
hgtsfc = regridder(hgtsfcin)
hgtsfc = np.moveaxis(hgtsfc,-1,-2)
print('hgtsfc shape/min/max = ',hgtsfc.shape, hgtsfc.min(), hgtsfc.max())

ref_time = model._structure.specs.aux_features['reference_datetime']
valid_time = model.sim_time_to_datetime64(state.state.sim_time._value)
print(valid_time,'u,v,tmp,z,ps min/max',u.min(),u.max(),v.min(),v.max(),t.min(),t.max(),z.min(),z.max(),ps.min(),ps.max())
print(valid_time,'spfh,cldwat,cldice min/max',q.min(),q.max(),clmr.min(),clmr.max(),icmr.min(),icmr.max())
units='hours since %s' % ref_time
print(cftime.num2date(model.from_nondim_units(state.state.sim_time,units='hours'),units=units))

for nouter in range(outer_steps):

    for nt in range(nsteps_per_timedelta):
        sim_date = model.sim_time_to_datetime64(state.state.sim_time._value)
        print(nouter, nt, cftime.num2date(model.from_nondim_units(state.state.sim_time,units='hours'),units=units),sim_date)
        state = model.advance(state, forcings=input_forcings)

# note: GSI will require obs on 64 hybrid levels, do vertical interpolation given pressures.
# history file variable names are tmp,ugrd,vgrd,spfh,pressfc,clwmr,icmr,dpres,hgtsfc,delz,o3mr...
    decoded = model.decode(state, input_forcings)
    z = decoded['geopotential'];  z = np.moveaxis(z/grav,-1,-2)
    t = decoded['temperature']; t =  np.moveaxis(t,-1,-2)
    q = decoded['specific_humidity']; q = np.moveaxis(q,-1,-2)
    u = decoded['u_component_of_wind']; u  = np.moveaxis(u,-1,-2)
    v = decoded['v_component_of_wind']; v = np.moveaxis(v,-1,-2)
    icmr = decoded['specific_cloud_ice_water_content']; icmr = np.moveaxis(icmr,-1,-2)
    clmr = decoded['specific_cloud_liquid_water_content']; clmr = np.moveaxis(clmr,-1,-2)
    ps = np.exp(model.model_coords.horizontal.to_nodal(state.state.log_surface_pressure))
    ps = model.from_nondim_units(ps, units='pascals')
    ps = np.moveaxis(ps, -1, -2)

    valid_time = model.sim_time_to_datetime64(state.state.sim_time._value) 
    print(valid_time,'u,v,temp,z,ps min/max',u.min(),u.max(),v.min(),v.max(),t.min(),t.max(),z.min(),z.max(),ps.min(),ps.max())
    print(valid_time,'spfh,cldwat,cldice min/max',q.min(),q.max(),clmr.min(),clmr.max(),icmr.min(),icmr.max())

    forecast_hour = (nouter+1)*(timedelta.item()).total_seconds()/3600
    valid_time = init_time + (nouter+1)*timedelta.item()
    print(valid_time)

    # write GFS history file
    nc = Dataset('sfg_%s_fhr%02i_mem%03i' % (valid_date,forecast_hour,nmem),'w')
    x = nc.createDimension('grid_xt',len(lons1d))
    y = nc.createDimension('grid_yt',len(lats1d))
    zz = nc.createDimension('pfull',len(pfull_arr))
    zi = nc.createDimension('phalf',len(phalf_arr))
    tt = nc.createDimension('time',1)
    nchar = nc.createDimension('nchars',20)
    pfull = nc.createVariable('pfull',np.float32,zz)
    phalf = nc.createVariable('phalf',np.float32,zi)
    phalf[:] = phalf_arr
    phalf.units = 'mb'
    phalf.units = 'mb'
    pfull[:] = pfull_arr
    pfull.units = 'mb'
    grid_xt = nc.createVariable('grid_xt',np.float64,x)
    lon = nc.createVariable('lon',np.float64,(y,x))
    grid_yt = nc.createVariable('grid_yt',np.float32,y)
    lat = nc.createVariable('lat',np.float64,(y,x))
    time = nc.createVariable('time',np.float64,tt)
    time_iso = nc.createVariable('time_iso','S1',(tt,nchar))
    time_iso._Encoding = 'ascii'
    #time.units = 'hours since %04i-%02i-%02i %02i:00:00' % (valid_time.year,valid_time.month,valid_time.day,valid_time.hour)
    time.units = 'hours since %s' % start_time
    time[0] = forecast_hour
    time_iso[0] = valid_time.isoformat()+'Z'
    grid_xt[:] = lons1d
    grid_xt.units = 'degrees_E'
    lon.units = 'degrees_E'
    grid_yt[:] = lats1d[::-1]
    grid_yt.units = 'degrees_N'
    lons,lats = np.meshgrid(lons1d,lats1d)
    lon[:] = lons; lat[:] = lats[::-1]
    lat.units = 'degrees_N'
    tmp_var = nc.createVariable('tmp',np.float32,(tt,zz,y,x),fill_value=9.9e20)
    tmp_var.cell_methods = "time: point"
    tmp_var[0,...] = t[:,::-1,:]
    z_var = nc.createVariable('z',np.float32,(tt,zz,y,x),fill_value=9.9e20)
    z_var.cell_methods = "time: point"
    z_var[0,...] = z[:,::-1,:]
    ugrd_var = nc.createVariable('ugrd',np.float32,(tt,zz,y,x),fill_value=9.9e20)
    ugrd_var.cell_methods = "time: point"
    ugrd_var[0,...] = u[:,::-1,:]
    vgrd_var = nc.createVariable('vgrd',np.float32,(tt,zz,y,x),fill_value=9.9e20)
    vgrd_var.cell_methods = "time: point"
    vgrd_var[0,...] = v[:,::-1,:]
    spfh_var = nc.createVariable('spfh',np.float32,(tt,zz,y,x),fill_value=9.9e20)
    spfh_var.cell_methods = "time: point"
    spfh_var[0,...] = q[:,::-1,:]
    clwmr_var = nc.createVariable('clwmr',np.float32,(tt,zz,y,x),fill_value=9.9e20)
    clwmr_var.cell_methods = "time: point"
    clwmr_var[0,...] = clmr[:,::-1,:]
    icmr_var = nc.createVariable('icmr',np.float32,(tt,zz,y,x),fill_value=9.9e20)
    icmr_var.cell_methods = "time: point"
    icmr_var[0,...] = icmr[:,::-1,:]
    hgtsfc_var = nc.createVariable('hgtsfc',np.float32,(tt,y,x),fill_value=9.9e20)
    hgtsfc_var.cell_methods = "time: point"
    hgtsfc_var[0,...] = hgtsfc[::-1,:]
    pressfc_var = nc.createVariable('pressfc',np.float32,(tt,y,x),fill_value=9.9e20)
    pressfc_var.cell_methods = "time: point"
    pressfc_var[0,...] = ps[0,::-1,:]
    nc.grid='gaussian'
    nc.grid_id=1
    nc.ncnsto=3
    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.pyplot as plt
    #plt.imshow(hgtsfc_var[0], cmap=plt.cm.hot_r, vmin=-300, vmax=6000)
    #plt.savefig('orog.png')
    #plt.figure()
    #plt.imshow(pressfc_var[0], cmap=plt.cm.hot_r, vmin=50000, vmax=110000)
    #plt.savefig('ps.png')
    #raise SystemExit
    nc.close()
