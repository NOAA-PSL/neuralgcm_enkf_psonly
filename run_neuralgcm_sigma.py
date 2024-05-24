import fsspec
import cftime
import gin
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import xarray

import functools, dataclasses, gcsfs, fsspec
from netCDF4 import Dataset
from datetime import datetime, timedelta

from dinosaur import spherical_harmonic, pytree_utils
from dinosaur import horizontal_interpolation, vertical_interpolation
from dinosaur import xarray_utils, primitive_equations, filtering, scales
from neuralgcm import api, orographies, model_builder, physics_specifications
import haiku as hk
import sys

# Helper function to load files from Google Cloud Storage

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
#model_name = 'neural_gcm_%s_deg_v0.pkl' % model_type # older checkpoint
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
physics_specs = model._structure.specs.physics_specs

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
demo_start_time = '%04i-%02i-%02i %02i:00:00' % (init_year,init_month,init_day,init_hour)
data_inner_steps = 6  # process every 6 hour

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
input_forcings = model.forcings_from_xarray(input_data_ds.isel(time=0))

rng_key = jax.random.PRNGKey(rng_seed)
state = model.encode(
    inputs, forcings=input_forcings, rng_key=rng_key)
grav = scales.GRAVITY_ACCELERATION.magnitude
ref_temp = model._structure.specs.aux_features['ref_temperatures']

# over-ride ERA5 initialization if an init_file is specified (with data in model coordinates)
if init_file:
    print("initializing from %s..." % init_file)
    ncin = Dataset(init_file)
    ncin.set_auto_mask(False)
    pressfc = ncin['pressfc'][:]
    pressfc = np.moveaxis(pressfc, -1, -2) # flip pos of lats and lons
    pressfc = model.to_nondim_units(pressfc[:,:,::-1], units='pascals')
    ugrd = ncin['ugrd'][0]
    ugrd = np.moveaxis(ugrd, -1, -2)
    ugrd = model.to_nondim_units(ugrd[:,:,::-1],units='meters per second')
    vgrd = ncin['vgrd'][0]
    vgrd = np.moveaxis(vgrd, -1, -2)
    vgrd = model.to_nondim_units(vgrd[:,:,::-1],units='meters per second')
    tmp = ncin['tmp'][0]
    tmp = np.moveaxis(tmp, -1, -2)
    tmp = model.to_nondim_units(tmp[:,:,::-1],units='kelvin')
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
    psdim = ncin['pressfc'][:]
    ncin.close()
    state.state.log_surface_pressure = model.model_coords.horizontal.to_modal(jnp.log(pressfc))
    state.state.temperature_variation = model.model_coords.horizontal.to_modal(jnp.array(tmp))
    vort, div = spherical_harmonic.uv_nodal_to_vor_div_modal(model.model_coords.horizontal, jnp.array(ugrd), jnp.array(vgrd)) 
    state.state.vorticity = vort; state.state.divergence = div
    state.state.tracers['specific_humidity'] = model.model_coords.horizontal.to_modal(jnp.array(spfh))
    state.state.tracers['specific_cloud_liquid_water_content'] = model.model_coords.horizontal.to_modal(jnp.array(clwmr))
    state.state.tracers['specific_cloud_ice_water_content'] = model.model_coords.horizontal.to_modal(jnp.array(icmr))
else:
    print("initializing from ERA5")

# run forecasts, converting to grid in sigma coords every nsteps_per_timedelta time steps

#print(dir(state.state))
#print(state.state.tracers.keys())
#print(forcings.keys())
icec = input_forcings['sea_ice_cover'].squeeze()
print(icec.shape, icec.min(), icec.max())
tmpsfc = input_forcings['sea_surface_temperature'].squeeze()
print(tmpsfc.shape, tmpsfc.min(), tmpsfc.max())
#hgtsfc = model._structure.specs.aux_features['xarray_dataset']['geopotential_at_surface'].values/grav
orog = model._structure.specs.aux_features['xarray_dataset']['geopotential_at_surface']

# choice 0
#hgtsfc = orog.values/grav

# choice 1
#filter_fns = [filtering.exponential_filter(model.model_coords.horizontal)]
#orography = primitive_equations.filtered_modal_orography(
#    orog, model.model_coords, model.model_coords, filter_fns)
#hgtsfc = model.model_coords.horizontal.to_nodal(orography)/grav

# choice2
#modal_orography = primitive_equations.truncated_modal_orography(orog,model.model_coords)
#hgtsfc = model.model_coords.horizontal.to_nodal(modal_orography)/grav

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
print('hgtsfc shape/min/max = ',hgtsfc.shape, hgtsfc.min(), hgtsfc.max())

#diff = hgtsfc1-hgtsfc
#print('diff',diff.min(), diff.max())

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#plt.imshow((orog[:,::-1].values/grav).T, cmap=plt.cm.hot_r, vmin=-300, vmax=6000)
#plt.savefig('orog1.png')
#plt.figure()
#plt.imshow(hgtsfc[:,::-1].T, cmap=plt.cm.hot_r, vmin=-300, vmax=6000)
#plt.savefig('orog2.png')
#plt.figure()
#diff = (orog[:,::-1].values/grav).T - hgtsfc[:,::-1].T
#print(diff.min(), diff.max())
#plt.imshow(diff, cmap=plt.cm.bwr, vmin=-1000, vmax=1000)
#plt.savefig('diff.png')
#raise SystemExit

lats1d = model._structure.specs.aux_features['xarray_dataset']['latitude'].values
lons1d = model._structure.specs.aux_features['xarray_dataset']['longitude'].values
print('hgtsfc min/max',hgtsfc.min(),hgtsfc.max(),hgtsfc.shape)
#sigma_values = model.model_coords.vertical.centers
sigma_values = model.model_coords.vertical.boundaries
nlevs = len(sigma_values)-1
print('nlevs',nlevs)
# get ak,bk from hyblevs file
#siglev_data = np.loadtxt('global_hyblev.l65.txt')
#nlevs = int(siglev_data[0,1]-1)
#ak = siglev_data[1:nlevs+2,0]
#bk = siglev_data[1:nlevs+2,1]
# define ak,bk from sigma values (no vertical interpolation)
ak = np.zeros_like(sigma_values)
bk = sigma_values

def state_to_grid(model, state, ak, bk, sigma_values):
    ref_temp = model._structure.specs.aux_features['ref_temperatures']
    physics_specs = model._structure.specs.physics_specs
    ps = np.exp(model.model_coords.horizontal.to_nodal(state.state.log_surface_pressure))
    ps = model.from_nondim_units(ps, units='pascals')
    pressi = sigma_values[:,np.newaxis,np.newaxis]*ps
    pressi_target = bk[:,np.newaxis,np.newaxis]*ps + ak[:,np.newaxis,np.newaxis]
    press = 0.5*(pressi[0:-1]+pressi[1:])
    press_target = 0.5*(pressi_target[0:-1]+pressi_target[1:])
    ps = np.moveaxis(ps, -1, -2)
    if press.shape == press_target.shape and np.abs(press-press_target).max() < 1.:
        #print("no vertical interp needed")
        vert_interp=False
    else:
        vert_interp=True

    temp = ref_temp[:,np.newaxis,np.newaxis] + model.model_coords.horizontal.to_nodal(state.state.temperature_variation)
    if vert_interp: temp = vertical_interpolation._vertical_interp_3d(press_target,press,temp)
    temp = np.moveaxis(temp, -1, -2) # so dimensions are level, lat, lon
    temp = model.from_nondim_units(temp, units='kelvin')
    spfh = model.model_coords.horizontal.to_nodal(state.state.tracers['specific_humidity'])
    spfh = spfh.clip(np.finfo(spfh.dtype).eps)
    if vert_interp: spfh = vertical_interpolation._vertical_interp_3d(press_target,press,spfh)
    spfh = np.moveaxis(spfh, -1, -2)
    cldwat = model.model_coords.horizontal.to_nodal(state.state.tracers['specific_cloud_liquid_water_content'])
    cldwat = cldwat.clip(np.finfo(cldwat.dtype).eps)
    if vert_interp: cldwat = vertical_interpolation._vertical_interp_3d(press_target,press,cldwat)
    cldwat = np.moveaxis(cldwat, -1, -2)
    cldice = model.model_coords.horizontal.to_nodal(state.state.tracers['specific_cloud_ice_water_content'])
    cldice = cldice.clip(np.finfo(cldice.dtype).eps)
    if vert_interp: cldice = vertical_interpolation._vertical_interp_3d(press_target,press,cldice)
    cldice = np.moveaxis(cldice, -1, -2)
    u, v = spherical_harmonic.vor_div_to_uv_nodal(
    model.model_coords.horizontal, state.state.vorticity, state.state.divergence)
    if vert_interp: u = vertical_interpolation._vertical_interp_3d(press_target,press,u)
    if vert_interp: v = vertical_interpolation._vertical_interp_3d(press_target,press,v)
    u = np.moveaxis(u, -1, -2); v = np.moveaxis(v, -1, -2)
    u = model.from_nondim_units(u, units='meters per second')
    v = model.from_nondim_units(v, units='meters per second')

    return u,v,temp,ps,spfh,cldwat,cldice

ugrd,vgrd,tmp,pressfc,spfh,clwmr,icmr = state_to_grid(model, state, ak, bk, sigma_values)
ref_time = model._structure.specs.aux_features['reference_datetime']
valid_time = model.sim_time_to_datetime64(state.state.sim_time._value)
print(valid_time,'u,v,temp,ps min/max',ugrd.min(),ugrd.max(),vgrd.min(),vgrd.max(),tmp.min(),tmp.max(),pressfc.min(),pressfc.max())
print(valid_time,'spfh,cldwat,cldice min/max',spfh.min(),spfh.max(),clwmr.min(),clwmr.max(),icmr.min(),icmr.max())
units='hours since %s' % ref_time
print(units)
print(cftime.num2date(model.from_nondim_units(state.state.sim_time,units='hours'),units=units))

# for delz computation
rd     = 2.8705e+2
rv     = 4.6150e+2
fv     = rv/rd-1.    # used in virtual temperature equation 

for nouter in range(outer_steps):

    for nt in range(nsteps_per_timedelta):
        sim_date = model.sim_time_to_datetime64(state.state.sim_time._value)
        print(nouter, nt, cftime.num2date(model.from_nondim_units(state.state.sim_time,units='hours'),units=units),sim_date)
        state = model.advance(state, forcings=input_forcings)

# note: GSI will require obs on 64 hybrid levels, do vertical interpolation given pressures.
# history file variable names are tmp,ugrd,vgrd,spfh,pressfc,clwmr,icmr,dpres,hgtsfc,delz,o3mr...
    ugrd,vgrd,tmp,pressfc,spfh,clwmr,icmr = state_to_grid(model, state, ak, bk, sigma_values)
    valid_time = model.sim_time_to_datetime64(state.state.sim_time._value) 
    print(valid_time,'u,v,temp,ps min/max',ugrd.min(),ugrd.max(),vgrd.min(),vgrd.max(),tmp.min(),tmp.max(),pressfc.min(),pressfc.max())
    print(valid_time,'spfh,cldwat,cldice min/max',spfh.min(),spfh.max(),clwmr.min(),clwmr.max(),icmr.min(),icmr.max())

    forecast_hour = (nouter+1)*(timedelta.item()).total_seconds()/3600
    valid_time = init_time + (nouter+1)*timedelta.item()
    print(valid_time)

    # write GFS history file
    nc = Dataset('sfg_%s_fhr%02i_mem%03i' % (valid_date,forecast_hour,nmem),'w')
    x = nc.createDimension('grid_xt',len(lons1d))
    y = nc.createDimension('grid_yt',len(lats1d))
    z = nc.createDimension('pfull',len(ak)-1)
    zi = nc.createDimension('phalf',len(ak))
    t = nc.createDimension('time',1)
    nchar = nc.createDimension('nchars',20)
    pfull = nc.createVariable('pfull',np.float32,z)
    phalf = nc.createVariable('phalf',np.float32,zi)
    phalf[:] = bk*1.e3 + ak
    phalf.units = 'mb'
    phalf.units = 'mb'
    pfull[:] = 0.5*(phalf[:-1]+phalf[1:])
    pfull.units = 'mb'
    grid_xt = nc.createVariable('grid_xt',np.float64,x)
    lon = nc.createVariable('lon',np.float64,(y,x))
    grid_yt = nc.createVariable('grid_yt',np.float32,y)
    lat = nc.createVariable('lat',np.float64,(y,x))
    time = nc.createVariable('time',np.float64,t)
    time_iso = nc.createVariable('time_iso','S1',(t,nchar))
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
    tmp_var = nc.createVariable('tmp',np.float32,(t,z,y,x),fill_value=9.9e20)
    tmp_var.cell_methods = "time: point"
    tmp_var[0,...] = tmp[:,::-1,:]
    #for k in range(nlevs):
    #    print(k,ak[k],bk[k],tmp_var[0,k,...].min(),tmp_var[0,k,...].max())
    ugrd_var = nc.createVariable('ugrd',np.float32,(t,z,y,x),fill_value=9.9e20)
    ugrd_var.cell_methods = "time: point"
    ugrd_var[0,...] = ugrd[:,::-1,:]
    vgrd_var = nc.createVariable('vgrd',np.float32,(t,z,y,x),fill_value=9.9e20)
    vgrd_var.cell_methods = "time: point"
    vgrd_var[0,...] = vgrd[:,::-1,:]
    spfh_var = nc.createVariable('spfh',np.float32,(t,z,y,x),fill_value=9.9e20)
    spfh_var.cell_methods = "time: point"
    spfh_var[0,...] = spfh[:,::-1,:]
    clwmr_var = nc.createVariable('clwmr',np.float32,(t,z,y,x),fill_value=9.9e20)
    clwmr_var.cell_methods = "time: point"
    clwmr_var[0,...] = clwmr[:,::-1,:]
    icmr_var = nc.createVariable('icmr',np.float32,(t,z,y,x),fill_value=9.9e20)
    icmr_var.cell_methods = "time: point"
    icmr_var[0,...] = icmr[:,::-1,:]
    o3mr_var = nc.createVariable('o3mr',np.float32,(t,z,y,x),fill_value=9.9e20)
    o3mr_var.cell_methods = "time: point"
    hgtsfc_var = nc.createVariable('hgtsfc',np.float32,(t,y,x),fill_value=9.9e20)
    hgtsfc_var.cell_methods = "time: point"
    hgtsfc_var[0,...] = np.transpose(hgtsfc[:,::-1])
    pressfc_var = nc.createVariable('pressfc',np.float32,(t,y,x),fill_value=9.9e20)
    pressfc_var.cell_methods = "time: point"
    pressfc_var[0,...] = pressfc[0,::-1,:]
    tv = tmp * ( 1.0 + fv*spfh) # convert T to Tv
    workarr = (rd/grav)*tv
    dpres = np.zeros_like(workarr)
    delz = np.zeros_like(workarr)
    bbk = bk.copy()
    bbk[0]=1.e-7
    for k in range(nlevs):
        delz[k] = workarr[k]*np.log((ak[k]+bbk[k]*pressfc[0])/(ak[k+1]+bk[k+1]*pressfc[0]))
        dpres[k] = pressfc[0]*(bk[k+1]-bk[k])
        #print(k,delz[k].min(),delz[k].max(),dpres[k].min(),dpres[k].max())
    delz_var = nc.createVariable('delz',np.float32,(t,z,y,x),fill_value=9.9e20)
    delz_var.cell_methods = "time: point"
    delz_var[0,...] = delz[:,::-1,:]
    dpres_var = nc.createVariable('dpres',np.float32,(t,z,y,x),fill_value=9.9e20)
    dpres_var.cell_methods = "time: point"
    dpres_var[0,...] = dpres[:,::-1,:]
    nc.grid='gaussian'
    nc.grid_id=1
    nc.ak = ak[:]
    nc.bk = bk[:]
    nc.ncnsto = 4
    nc.close()
    nc = Dataset('bfg_%s_fhr%02i_mem%03i' % (valid_date,forecast_hour,nmem),'w')
    x = nc.createDimension('grid_xt',len(lons1d))
    y = nc.createDimension('grid_yt',len(lats1d))
    z = nc.createDimension('pfull',len(ak)-1)
    zi = nc.createDimension('phalf',len(ak))
    t = nc.createDimension('time',1)
    nchar = nc.createDimension('nchars',20)
    pfull = nc.createVariable('pfull',np.float32,z)
    phalf = nc.createVariable('phalf',np.float32,zi)
    phalf[:] = bk*1.e3 + ak
    phalf.units = 'mb'
    phalf.units = 'mb'
    pfull[:] = 0.5*(phalf[:-1]+phalf[1:])
    pfull.units = 'mb'
    grid_xt = nc.createVariable('grid_xt',np.float64,x)
    lon = nc.createVariable('lon',np.float64,(y,x))
    grid_yt = nc.createVariable('grid_yt',np.float32,y)
    lat = nc.createVariable('lat',np.float64,(y,x))
    time = nc.createVariable('time',np.float64,t)
    time_iso = nc.createVariable('time_iso','S1',(t,nchar))
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
    nc.grid='gaussian'
    nc.grid_id=1
    nc.fhzero=3
    hgtsfc_var = nc.createVariable('orog',np.float32,(t,y,x),fill_value=9.9e20)
    hgtsfc_var.cell_methods = "time: point"
    hgtsfc_var[0,...] = np.transpose(hgtsfc[:,::-1])
    pressfc_var = nc.createVariable('pressfc',np.float32,(t,y,x),fill_value=9.9e20)
    pressfc_var.cell_methods = "time: point"
    pressfc_var[0,...] = pressfc[0,::-1,:]
    tmpsfc_var = nc.createVariable('tmpsfc',np.float32,(t,y,x),fill_value=9.9e20)
    tmpsfc_var.cell_methods = "time: point"
    tmpsfc_var[0,...] = np.transpose(tmpsfc[:,::-1])
    icec_var = nc.createVariable('icec',np.float32,(t,y,x),fill_value=9.9e20)
    icec_var.cell_methods = "time: point"
    icec_var[0,...] = np.transpose(icec[:,::-1])
    nc.close()
