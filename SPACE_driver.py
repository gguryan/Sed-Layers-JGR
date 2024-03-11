# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:05:50 2022

@author: gjg882
"""

#imports

import numpy as np
from matplotlib import pyplot as plt

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

#200x200 grid inputs
#mg = read_netcdf('Inputs/topo_init_200x200.nc') #initial topography
#inputs = load_params('Inputs/SPACE_params_200x200.txt') #load params from text file
#ds_file = ('Output/SPACE_out_200x200.nc') #filename to save the model output to

#50x50 grid inputs
mg = read_netcdf('Inputs/topo_init_50x50.nc') #initial topography
inputs = load_params('Inputs/SPACE_params_50x50.txt') #load params from text file
ds_file = ('Output/SPACE_out_50x50.nc') #filename to save the model output to




#%%
#SET UP MODEL GRID


if mg.has_field("bedrock__elevation", at="node") == False: 

    # #Create a field for bedrock elevation
    mg.add_zeros('node', 'bedrock__elevation')
    
    # #Initial soil depth, m
    soil_init = inputs['soil_init']
    
    # #Create a grid field for soil depth
    mg.add_zeros('node', 'soil__depth')
    
    # #Add initial layer of soil
    mg.at_node['soil__depth'][:] = soil_init


#%% Set model grid boundary conditions

#All boundaries are closed except outlet node
mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=True,
                                       left_is_closed=True,
                                       right_is_closed=True,
                                       top_is_closed=True)

#Setting Node 0 (bottom left corner) as outlet node 
mg.set_watershed_boundary_condition_outlet_id(0, 
                                              mg.at_node['topographic__elevation'],
                                              -9999.)

#Shrink initial topography so drainage network is preserved but elevations are cm-scale
mg.at_node['topographic__elevation'][:] = mg.at_node['topographic__elevation'] * .001
mg.at_node['bedrock__elevation'][:] = mg.at_node['topographic__elevation']

#%%

#Look at initial topography
plt.figure(dpi=300)
imshow_grid(mg, "topographic__elevation")
plt.title('Initial Topographic Elevation')
#plt.show()
#plt.savefig('topo_init_nx200.svg')
#%%Configure lithology and layers


print('configuring lithology - may take several minutes')
K_soft = inputs['K_soft']
K_ratio = inputs['K_ratio']

#Erodibility
K_hard = K_soft / K_ratio

#Used for deposition only in case litholayers needs to deposit something (it shouldn't)
K_avg = (K_soft + K_hard) / 2


#Set the erodibility (K) of each rock layer using a dict
lith_attrs = {'K_sp': {1: K_hard, 2: K_soft, 
                  3: K_hard, 4: K_soft,
                  5: K_hard, 6: K_soft,
                  7: K_hard, 8: K_soft, 
                  9: K_hard, 10: K_soft, 
                  11: K_avg}}

layer_thickness = inputs['layer_thickness']

#layers repeat from 1-10 4 times
layer_ids = np.tile(list(range(1, 11)), 4)

#calculate max depth needed for given number of layers
max_depth =(layer_thickness*len(layer_ids))+layer_thickness

#make the bottom layer extra thick so it doesn't get eroded through      
layer_depths = np.arange(layer_thickness, max_depth, layer_thickness)
layer_depths[-1] += 300

#instantitae litholyaers
lith = LithoLayers(mg, 
                    layer_depths, 
                    layer_ids,
                    attrs=lith_attrs,
                    layer_type='MaterialLayers',
                    rock_id=10) 


#%% Instantiate flow components

#instantiate flow routing
fr = PriorityFloodFlowRouter(mg, flow_metric='D8', suppress_out = True)
fr.run_one_step()

#%% Instantiate SPACE component 


m_sp = inputs['m_sp'] #stream power exponent
n_sp = inputs['n_sp'] #stream power exponent
K_sed = inputs['K_sed'] #sediment erodibility
F_f = inputs['F_f'] #fraction of fines/wash load 
phi = inputs['phi'] #sediment porosity
H_star = inputs['H_star'] #sediment entrainment length scale
v_s = inputs['v_s'] #particle settling velocity
sp_crit_sed = inputs['sp_crit_sed'] #threshold for sediment erosion
sp_crit_br = inputs['sp_crit_br'] #threshold for bedrock erosion


space = SpaceLargeScaleEroder(mg,
           K_sed =K_sed,
           K_br = K_hard,
           F_f = F_f,
           phi = phi,
           H_star = H_star,
           v_s = v_s,
           m_sp = m_sp,
           n_sp = n_sp,
           sp_crit_sed = sp_crit_sed,
           sp_crit_br = sp_crit_br)

#space runtime parameters
space_dt = inputs['space_dt'] #timestep in years
space_uplift = inputs['space_uplift'] #m/yr
space_runtime = inputs['space_runtime'] #total model runtime in years
space_runtime_kyr = int(space_runtime / 1000) #for labeling output files, plots

#Array of all time steps
t = np.arange(0, space_runtime+space_dt, space_dt)
nts = len(t)


nx=mg.shape[0]

#set how often you want to save model output
if nx <= 50:
    save_interval = 1000
else:
    save_interval = 10000


#create an array of all the model times where data will be saved
out_times = np.arange(0, space_runtime+save_interval, save_interval)
out_count = len(out_times)

#%%
#Create Xarray dataset to save model output
#Each landlab model grid field is saved as a data_var

ds = xr.Dataset(
    data_vars={
        
        'topographic__elevation': (
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
        
        'soil__depth':
        (('time', 'y', 'x'), np.empty((out_count, mg.shape[0], mg.shape[1])), {
            'units': 'meters',
            'long_name': 'Sediment Depth'
        
        }),
        
        'sediment__flux':
        (('time', 'y', 'x'), np.empty((out_count, mg.shape[0], mg.shape[1])), {
            'units': '"m$^3$/yr"',
            'long_name': 'Sediment Flux'
            
        }),
        
        'bedrock__erosion':
        (('time', 'y', 'x'), np.empty((out_count, mg.shape[0], mg.shape[1])), {
            'units': '"m/yr"',
            'long_name': 'Bedrock Erosion'
            
        }),
        
        'sediment__erosion':
        (('time', 'y', 'x'), np.empty((out_count, mg.shape[0], mg.shape[1])), {
            'units': '"m/yr"',
            'long_name': 'Sediment Erosion'
        
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
    attrs=dict(inputs)) #save the model inputs to the dataset's metadata
    


#%% Run the model!

#Track how long it takes loop to run
start_time = time.time()

elapsed_time = 0

#Create model grid fields to store erosion rates at each cell
#Note - both of these are [m/yr] - length per time
E_s = mg.add_zeros('sediment__erosion', at='node')
E_r = mg.add_zeros('bedrock__erosion', at='node')

#Grid fields to be saved to xarray dataset
out_fields = ['topographic__elevation', 
              'rock_type__id', 
              'soil__depth', 
              'sediment__flux',
              'bedrock__erosion',
              'sediment__erosion']

#Save initial condition at time 0 to xarray output
for of in out_fields:
    ds[of][0, :, :] = mg['node'][of].reshape(mg.shape)


print('Starting SPACE loop')

for i in range(nts):
    
    #New priority flow router component
    fr.run_one_step()
    
    #erode with space
    _ = space.run_one_step(dt=space_dt)
    
    
    #Layers are advected upwards due to uplift
    dz_ad = np.zeros(mg.size('node'))
    dz_ad[mg.core_nodes] = space_uplift * space_dt
    mg.at_node['bedrock__elevation'] += dz_ad 
    lith.dz_advection=dz_ad
    
    #Recalculate topographic elevation to account for rock uplift
    mg.at_node['topographic__elevation'][:] = \
        mg.at_node['bedrock__elevation'][:] + mg.at_node['soil__depth'][:]
    
    #Update lithology
    lith.run_one_step()
    
    #Update space K values
    space.K_br = mg.at_node['K_sp']


    if elapsed_time %save_interval== 0:
        
        E_r_term = space._Er
        mg.at_node['bedrock__erosion'] = E_r_term.reshape(mg.shape[0], mg.shape[1])
        
        E_s_term = space._Es
        mg.at_node['sediment__erosion'] = E_s_term.reshape(mg.shape[0], mg.shape[1])
        
        
        ds_ind = int((elapsed_time/save_interval))
        

        for of in out_fields:
            ds[of][ds_ind, :, :] = mg['node'][of].reshape(mg.shape)
        
        us_outlet_Q = mg.at_node['surface_water__discharge'][51]
        outlet_Q = mg.at_node['surface_water__discharge'][0]
        
        end_time = time.time()
        loop_time = round((end_time - start_time) / 60)
        
        print(elapsed_time, ds_ind, 'code runtime =', loop_time, 'min')
        ds.to_netcdf(ds_file)
    
    elapsed_time += space_dt

end_time = time.time()
loop_time = round((end_time - start_time) / 60)
print('Loop time =', loop_time)
#%%plot final topography 

plt.figure()
imshow_grid(mg, "topographic__elevation")
plt.title('Final Topographic Elevation')
