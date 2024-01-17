# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 20:02:52 2024

@author: gjg882
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import  GridSpec

from matplotlib.lines import Line2D

from matplotlib.offsetbox import AnchoredText

import seaborn as sns

import xarray as xr

import pandas as pd

from landlab import RasterModelGrid
from landlab import load_params
from landlab.plot import imshow_grid
from landlab.components import (FlowAccumulator, 
                                DepressionFinderAndRouter,
                                FastscapeEroder,
                                ChannelProfiler,
                                SteepnessFinder,
                                ChiFinder)


#%%USER INPUTS

#string to name exported plot files 
file_id = 'dtch_1200kyr_nx200'

#open xarray dataset of saved model output
ds_file = 'C:/Users/gjg882/Box/UT/Research/Code/space/space_paper/big_grid_figs/dtch_1200kyr_nx200_Kr5_ds_final.nc'

# optional - select the time to plot
# plot_time = 800000

#%%
#load in model output
ds = xr.open_dataset(ds_file) 

#load in parameters
ds_attrs = ds.attrs

plot_time = ds.attrs['space_runtime']

rock = ds.rock_type__id.sel(time=plot_time)

#%%
#Define function to run channel profiler, calculate steepness, and calculate chi for a given time in xarray dataset

def channel_calcs(ds, plot_time, sf_min_DA, cf_min_DA):
    
    ds_attrs = ds.attrs
    
    nx = ds_attrs['nx']
    ny = ds_attrs['ny']
    dx = ds_attrs['dx']
    model_name = ds_attrs['model_name']

    #m and n are the same for both fastscape and space
    m_sp = ds_attrs['m_sp']
    n_sp = ds_attrs['n_sp']
    
    theta = m_sp / n_sp
    
    #make model grid using topographic elevation at desired time
    mg = RasterModelGrid((nx, ny), dx)
    z = ds.topographic__elevation.sel(time=plot_time)
    z = mg.add_field("topographic__elevation", z, at="node")
    
    #Setting Node 0 (bottom left corner) as outlet node 
    mg.set_watershed_boundary_condition_outlet_id(0, mg.at_node['topographic__elevation'],-9999.)
    
    #Add rock type at each node to model grid
    rock = ds.rock_type__id.sel(time=plot_time)
    mg.has_field('node', 'rock_type__id')
    rock is mg.add_field('node', 'rock_type__id', rock, dtype=int)
    
    var_list = list(ds.keys())
    
    if 'bedrock__erosion' in var_list:
    
        #Add bedrock erosion term
        Er = ds.bedrock__erosion.sel(time=plot_time)
        mg.has_field('node', 'bedrock__erosion')
        Er is mg.add_field('node', 'bedrock__erosion', Er, dtype=float)
    
    if model_name == 'SPACE':
        
        #Add soil depth to model grid
        sed_depth = ds.soil__depth.sel(time=plot_time)
        mg.has_field('node', 'soil__depth')
        sed_depth is mg.add_field('node', 'soil__depth', sed_depth, dtype=float)
        
        #Add sediment erosion term
        Es = ds.sediment__erosion.sel(time=plot_time)
        mg.has_field('node', 'sediment__erosion')
        Es is mg.add_field('node', 'sediment__erosion', Es, dtype=float)

    
    #Run flow accumulator
    fa = FlowAccumulator(mg, flow_director='D8')
    fa.run_one_step()

    #Run channel profiler to find main channel nodes
    prf = ChannelProfiler(mg, main_channel_only=True,minimum_channel_threshold=sf_min_DA)
    prf.run_one_step()
    
    #Calculate Channel Steepness
    sf = SteepnessFinder(mg, reference_concavity=theta, min_drainage_area=sf_min_DA)
    sf.calculate_steepnesses()    

    #Calculate Chi
    cf = ChiFinder(mg, min_drainage_area=cf_min_DA,reference_concavity=theta,use_true_dx=True)
    cf.calculate_chi()
    
    #Get nodes that define main channel
    prf_keys = list(prf.data_structure[0])
    
    

    channel_dist = prf.data_structure[0][prf_keys[0]]['distances']
    channel_dist_ids = prf.data_structure[0][prf_keys[0]]['ids']
    channel_elev = mg.at_node["topographic__elevation"][channel_dist_ids]
    channel_chi = mg.at_node["channel__chi_index"][channel_dist_ids]
    channel_ksn = mg.at_node["channel__steepness_index"][channel_dist_ids]

    channel_x = []
    channel_y = []
    
    for node in channel_dist_ids:
    
        channel_x.append(mg.x_of_node[node])
        channel_y.append(mg.y_of_node[node])

    channel_rock_ids = mg.at_node['rock_type__id'][channel_dist_ids]
    channel_rock_ids = channel_rock_ids.astype(int)


    df = pd.DataFrame({'channel_dist': channel_dist,
                   'channel_elev': channel_elev,
                   'channel_rock_id': channel_rock_ids, 
                   'channel_ksn' : channel_ksn,
                   'channel_chi' : channel_chi,
                   'channel_x' : channel_x,
                   'channel_y' : channel_y})
    
    df["uplift"] = .001
    
    if 'bedrock__erosion' in var_list:
        channel_Er = mg.at_node['bedrock__erosion'][channel_dist_ids] 
        df['channel_Er'] = channel_Er
    
    if model_name == 'SPACE':
        
        #add sediment data for SPACE model runs
        channel_sed_depth = mg.at_node['soil__depth'][channel_dist_ids]
        df['channel_sed_depth'] = channel_sed_depth
        
        channel_Es = mg.at_node['sediment__erosion'][channel_dist_ids]  
        df['channel_Es'] = channel_Es
            
    #drop first row of dataframe - this is the outlet node
    df1 = df.iloc[1: , :]    
    
    return mg, df1

#%%

def plot_prf_fig3 (df, plot_time, model_name, save_fig, file_id, panel_label):
    
    #colormap for plotting
    #lith_cmap=plt.cm.get_cmap('Paired', 10)
    lith_cmap = sns.color_palette("Paired", 10, as_cmap=True)
    
    #Convert time to kyr for labels
    plot_time_kyr = int(plot_time / 1000)
    plot_time_myr = plot_time / 1000000
    
    #group dataframe by rock type
    groups = df.groupby('channel_rock_id')
    
    #make the figure
    #size used to be 10,4
    fig = plt.figure(constrained_layout=True, figsize=(8, 4), dpi=300)

    
    #plot channel profile in top row
    ax1 = fig.add_subplot()
    
    for name, group in groups:
        
        if name %2 == 0:
        
            ax1.plot(group.channel_dist, group.channel_elev, 
                    marker='o', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
        else:
            ax1.plot(group.channel_dist, group.channel_elev, 
                    marker='s', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
        
    legend_elements_dtch = [Line2D([0], [0], marker='o', color='w', label='Soft Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Hard Rock',
                          markerfacecolor='dimgrey', markersize=10)]
    
    legend_elements_mixed = [Line2D([0], [0], marker='o', color='w', label='Soft Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Hard Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], color='dimgrey', lw=2, label='Sediment Thickness')]

    #ax1.legend(title="Layer ID")

    ax1.set_xlabel('Distance Upstream (m)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_ylim(top=450)
    ax1.set_xlim(left=0)
    
    anchored_text = AnchoredText(panel_label, loc='upper left', prop=dict(fontsize="14"))
    ax1.add_artist(anchored_text)   
    
    if model_name == 'SPM':
        
        title_string = f"{model_name} Model, Time={plot_time_myr} myr"
        
        
        #ax1.legend(loc='best', title='Layer ID', bbox_to_anchor=[0.09, 0.85],) #use this one to include layer colors in legend
        
        ax1.legend(handles=legend_elements_dtch, loc='best')
        ax=ax1.twinx()
        ax.set_ylabel("TEST")
        ax.yaxis.label.set_size(12)
    
        
        '''
        
        ax=ax1.twinx()
        
       
        ax.plot(df['channel_dist'], df['channel_Er'], '-', color = 'blue', label = 'Bedrock Erosion Rate')
        ax.set_ylabel("Erosion Rate (m/yr)")
        
        ax.plot(df['channel_dist'], df['uplift'], '--', color = 'dimgrey', label = 'Uplift Rate')
        '''
    
    if model_name == 'SPACE':
        
        v_s_round = np.around(ds.attrs['v_s'])
        
        #title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_kyr} kyr, V={v_s_round} m/yr"
        title_string = f"{model_name} Model, Time={plot_time_kyr} kyr, V={v_s_round} m/yr"
        
        #ax1.legend(handles=legend_elements_mixed, bbox_to_anchor=[0.1, 0.8], loc='center left')
        ax1.legend(loc='best', title='Layer ID', bbox_to_anchor=[0.1, 0.85],) #use this one to include layer colors in legend
        
        
        
        #Add sediment depth to channel profile w/ secondary y-axis
        ax=ax1.twinx()
        ax.plot(df['channel_dist'], df['channel_sed_depth'], '-', color = 'dimgrey', label = 'Sediment Thickness')
        ax.set_ylabel("Sediment Thickness H (m)")
        ax.set_ylim(top=2.25)
        #ax.legend(loc='best')
        
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
    
 
    ax1.set_title(title_string)
    
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    
    ax1.xaxis.label.set_size(12)
    ax1.yaxis.label.set_size(12)
    

    
    if save_fig == True:
        file_string = 'PRF_sed_depth' + file_id + '.svg'
        fig.savefig(file_string);
