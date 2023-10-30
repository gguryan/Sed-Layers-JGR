# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:05:50 2022

@author: gjg882
"""

#imports

import numpy as np
from matplotlib import pyplot as plt

import pandas as pd

import xarray as xr

import time

from os import chdir
from pathlib import Path

import numpy as np
from landlab import RasterModelGrid
from landlab import load_params
from landlab.io.netcdf import read_netcdf
from landlab.io.netcdf import write_netcdf
from landlab.plot import imshow_grid
from landlab.components import (FlowAccumulator, 
                                DepressionFinderAndRouter,
                                FastscapeEroder,
                                Lithology,
                                LithoLayers,
                                ChannelProfiler,
                                ChiFinder,
                                Space, 
                                PriorityFloodFlowRouter, 
                                SpaceLargeScaleEroder)

#%%
#LOAD INPUT FILES HERE 

#Initial Topography
#mg = read_netcdf('drainage_maker_fsc_3000kyr_nx200_mg_attrs.nc')
mg = read_netcdf('NoLayers_20mdx_500kyr_n15_mg.nc')

#load Parameters
#inputs = load_params('dtch_params_1200kyr_nx200_Kr5.txt')
inputs = load_params('params_dtch_SS_layers_40m_1200kyr_kr5.txt')



#%%
#SET UP MODEL GRID


#TODO - Are these lines needed? Move elsewhere?
dx=inputs['dx']
nx=inputs['nx']
lith_cmap = plt.cm.get_cmap('Paired', 10)
model_name = inputs['model_name']

#%%
#Uncomment this section if initial grid was created with Fastscape eroder (only the 50x50 grids were made w SPACE)

# #Create a field for bedrock elevation
# mg.add_zeros('node', 'bedrock__elevation')

# #Initial soil depth, m
# soil_init = inputs['soil_init']

# #Create a grid field for soil depth
# mg.add_zeros('node', 'soil__depth')

# #Add initial layer of soil
# mg.at_node['soil__depth'][:] = soil_init


#%%    
#All boundaries are closed except outlet node
mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=True,
                                       left_is_closed=True,
                                       right_is_closed=True,
                                       top_is_closed=True)

#Setting Node 0 (bottom left corner) as outlet node 
mg.set_watershed_boundary_condition_outlet_id(0, 
                                              mg.at_node['topographic__elevation'],
                                              -9999.)

mg.at_node['topographic__elevation'][:] = mg.at_node['topographic__elevation'] * .001


#%%
plt.figure()
imshow_grid(mg, "topographic__elevation")
plt.title('Initial Topographic Elevation')
plt.show()
#%%
#CONFIGURE LITHOLOGY

lith_start_time = time.time()

print('configuring lithology')
K_soft = inputs['K_soft']
K_ratio = inputs['K_ratio']

#Erodibility
K_hard = K_soft / K_ratio

#Used for deposition only
K_avg = (K_soft + K_hard) / 2


lith_attrs = {'K_sp': {1: K_hard, 2: K_soft, 
                  3: K_hard, 4: K_soft,
                  5: K_hard, 6: K_soft,
                  7: K_hard, 8: K_soft, 
                  9: K_hard, 10: K_soft, 
                  11: K_avg}}

layer_thickness = inputs['layer_thickness']

layer_ids = np.tile(list(range(1, 11)), 4)

max_depth =(layer_thickness*len(layer_ids))+layer_thickness

#bottom layer is extra thick      
layer_depths = np.arange(layer_thickness, max_depth, layer_thickness)
layer_depths[-1] += 300


lith = LithoLayers(mg, 
                    layer_depths, 
                    layer_ids,
                    attrs=lith_attrs,
                    layer_type='MaterialLayers',
                    rock_id=11)


lith_end_time = time.time()
lith_time = round((lith_end_time - lith_start_time)  / 60)
print('Lithology complete, time =', lith_time, 'minutes')

dz_init = lith.dz.copy()

#%%
#INSTANTIATE FLOW COMPONENTS
m_sp = inputs['m_sp']
n_sp = inputs['n_sp']

print('Initiating flow router')

fa = FlowAccumulator(mg, flow_director='D8') #TODO Remove?
#fr = PriorityFloodFlowRouter(mg, flow_metric='D8', suppress_out = True)

fa.run_one_step()

print('Initial flow routing complete')

#%%
#Instantiate Fastcape Eroder



#space runtime parameters
fsc_dt = inputs['fsc_dt'] #years
fsc_uplift = inputs['fsc_uplift']
fsc_runtime = inputs['fsc_runtime']
fsc_runtime_kyr = int(fsc_runtime / 1000) #for labeling output files, plots

#Array of all time steps
t = np.arange(0, fsc_runtime+fsc_dt, fsc_dt)
nts = len(t)


#Telling lithology which rock type to deposit when deposition occurs
#Can change this to be spatially variable, setting at type 2 to see if/where deposition is happening
lith.rock_id=10

# if nx <= 50:
#     save_interval = 1000
# else:
#     save_interval = 10000

save_interval = 1000

out_times = np.arange(0, fsc_runtime+save_interval, save_interval)
out_count = len(out_times)



fsc = FastscapeEroder(mg, K_sp = 'K_sp', m_sp = m_sp, n_sp = n_sp)


#%% Create Xarray dataset to save model output

ds = xr.Dataset(
    data_vars={
        
        'topographic__elevation':  (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'meters',  # dictionary with data attributes
                'long_name': 'Topographic Elevation'
            }),
            
        'rock_type__id':
            (('time', 'y', 'x'), np.empty((out_count, mg.shape[0], mg.shape[1])), {
                'units': '-',
                'long_name': 'Rock Type ID Code'
            }),
        
        'bedrock__erosion' : (
            ('time', 'y', 'x'),  # tuple of dimensions
            np.empty((out_count, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                'units': 'meters/yr',
                'long_name': 'Bedrock Erosion'   
        })      
    },
        
    coords={
        'x': (
            ('x'),  # tuple of dimensions
            mg.x_of_node.reshape(
                mg.shape)[0, :],  # 1-d array of coordinate data
            {
                'units': 'meters'
            }),  # dictionary with data attributes
        'y': (('y'), mg.y_of_node.reshape(mg.shape)[:, 1], {
            'units': 'meters'
        }),
        'time': (('time'), out_times, {
            'units': 'years',
            'standard_name': 'time'
        })
    },
    attrs=dict(inputs))
    
    
#%%



file_id = f'dtch_{fsc_runtime_kyr}kyr_nx{nx}_Kr{K_ratio}_fsc_ero' 

print(file_id)


#%%

path = Path.cwd() / file_id
path.mkdir(exist_ok=True)
chdir(path)

#%%

fsc.run_one_step(dt=fsc_dt)
alpha_test = fsc._alpha


#%%
#Evolve the landscape!

#Track how long it takes loop to run
start_time = time.time()

elapsed_time = 0

outlet_Qs_arr = np.empty(t.shape)

#Note - both of these are [m/yr] - length per time
E_r = mg.add_zeros('bedrock__erosion', at='node')
vol_eroded = mg.add_zeros('volume__eroded', at='node')

#Output field for xarray dataset
out_fields = ['topographic__elevation', 'rock_type__id', 'bedrock__erosion']

#Save initial condition to xarray output
for of in out_fields:
    ds[of][0, :, :] = mg['node'][of].reshape(mg.shape)

ds_file = file_id + "_ds_qs.nc"

print('Starting LEM loop')

for i in range(nts):
    
    
    #create a copy of original topography before erosion
    
    topo_orig =  mg.at_node['topographic__elevation'].copy()
    
    #New priority flow router component
    fa.run_one_step()
    
    #erode with space
    _ = fsc.run_one_step(dt=fsc_dt)
    
    
    #E_r_term = fsc._alpha / fsc_dt
    E_r_term = (topo_orig - mg.at_node['topographic__elevation'][:]) / fsc_dt
    
    
    mg.at_node['bedrock__erosion'][:] = E_r_term

    vol_eroded[:] = mg.at_node['bedrock__erosion'][:] * (dx**2)
    
    outlet_Qs = np.sum(vol_eroded)
    
    outlet_Qs_arr[i] = outlet_Qs
    
    #Layers are advected upwards due to uplift
    dz_ad = np.zeros(mg.size('node'))
    dz_ad[mg.core_nodes] = fsc_uplift * fsc_dt
    mg.at_node['topographic__elevation'] += dz_ad
    lith.dz_advection=dz_ad
        
    #Update lithology
    lith.run_one_step()
    
    #Update space K values
    fsc.K_br = mg.at_node['K_sp']
    

    
    if elapsed_time %1000 == 0:
        print(elapsed_time)

    if elapsed_time %save_interval== 0:
        
        ds_ind = int((elapsed_time/save_interval))
    
        for of in out_fields:
            ds[of][ds_ind, :, :] = mg['node'][of].reshape(mg.shape)
        
        print(elapsed_time, ds_ind, outlet_Qs)
        #ds.to_netcdf(ds_file)
       
    
    elapsed_time += fsc_dt

end_time = time.time()
loop_time = round((end_time - start_time) / 60)
print('Loop time =', loop_time)
#%%

plt.figure()
imshow_grid(mg, "topographic__elevation")
plt.title('Final Topographic Elevation')

#%%

#save xr dataset to netcdf
ds_file = file_id + "_ds_final_Qs.nc"
#ds.to_netcdf(ds_file)

df = pd.DataFrame({ 'outlet_Qs': outlet_Qs_arr })
df['time']=pd.Series(t)

#%%
# plot_time = 40000
# plot_time_kyr = int(plot_time/1000)
# fig=plt.figure(figsize=(10,8))
# ds.topographic__elevation.sel(time=plot_time).plot(cmap='pink')
# plt.title(f'Topography at {plot_time_kyr} kyr, Vs = 3.0 m/yr', fontsize = 20)

#%%

'''
out_path = Path('C:\\Users\\gjg882\\Box\\UT\\Research\\Code\\space\\space_paper\\mixed_driver_master\\test_dir')
out_path.mkdir(exist_ok=True)


ds_out_file = str(out_path) + ds_file
ds.to_netcdf(ds_out_file)


#save xr dataset to netcdf
ds_file = file_id + "_ds_final.nc"
ds.to_netcdf(ds_file)

#save model grid to netcdf
mg_title_string = file_id + "_mg.nc"
mg_attrs = dict(inputs)
write_netcdf(mg_title_string, mg, attrs=mg_attrs)

'''

#%%

plt.figure()
plt.plot(t, outlet_Qs_arr)

#%%

df.to_csv('dtch_outlet_qs.csv')
