# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:31:24 2024

@author: gjg882
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:41:44 2022

@author: gjg882
"""

#%%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors


from matplotlib.gridspec import  GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText

import seaborn as sns

import xarray as xr

import pandas as pd



from landlab import RasterModelGrid
from landlab.plot import imshow_grid
from landlab.components import (FlowAccumulator, 
                                DepressionFinderAndRouter,
                                FastscapeEroder,
                                ChannelProfiler,
                                SteepnessFinder,
                                ChiFinder)


#%%
# =============================================================================
# 
# #Label for output filenames - model type, runtime, grid size, bedrock erdobility ratio, misc. 
# 
# #file_id = 'Mixed_1200kyr_nx50_Kr5_Ksr1_Vs3_60mlayers'
# #ds_file = 'C:/Users/gjg882/Box/UT/Research/Code/space/space_paper/mixed_driver_master/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs3_60mlayers/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs3_60mlayers_ds_final.nc'
# 
# #file_id = 'dtch_1200kyr_nx50_Kr5'
# #ds_file = 'C:/Users/gjg882/Box/UT/Research/Code/space/space_paper/dtch_driver_master/dtch_1200kyr_nx50_Kr5_fsc_ero/dtch_1200kyr_nx50_Kr5_fsc_ero_ds_final.nc'
# 
# 
# file_id = 'mixed_1200kyr_nx50_Kr5_Ksr1_Vs1'
# ds_file = 'C:/Users/gjg882/Box/UT/Research/Code/space/space_paper/mixed_driver_master/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs1/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs1_ds_final.nc'
# 
# #file_id = 'mixed_1200kyr_nx50_Kr5_Ksr1_Vs3'
# #ds_file = 'C:/Users/gjg882/Box/UT/Research/Code/space/space_paper/mixed_driver_master/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs3/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs3_ds_final.nc'
# 
# #file_id = 'mixed_1200kyr_nx50_Kr5_Ksr1_Vs5'
# #ds_file = 'C:/Users/gjg882/Box/UT/Research/Code/space/space_paper/mixed_driver_master/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs5/Mixed_1200kyr_nx50_Kr5_Ksr1_Vs5_ds_final.nc'
# 
# 
# 
# #load in model output
# 
# ds = xr.open_dataset(ds_file) 
# 
# #load in parameters
# #ds_attrs = ds.attrs
# 
# #Select model time (in years) to plot
# plot_time = 1200000
# #plot_time = ds.attrs['space_runtime']
# 
# =============================================================================



#TODO

#%%Calculate the main channel profile and extract data from along the profile

def calc_main_channel(ds, plot_time, sf_min_DA=500, cf_min_DA=500):

    """
    Function runs channel profiler, calculates steepness, and calculates chi for a given time in xarray dataset

    Parameters
    ----------
    ds : xarray dataset with model output

    plot_time : the desired time point to plot from the xarray dataset

    sf_min_DA : the minimum drainage area to use for the steepness finder component
    
    cf_min_DA : the minimum drainage area to use for the chi finder component

    Returns
    -------
    out : mg, df
        mg : landlab model grid 
        df : data frame of points along the main channel profile with xy coords, ksn, erosion rates, rock type, etc
    """

    ds_attrs = ds.attrs
    
    model_name = ds.attrs['model_name']

    #update model names to make plot labels consistent with manuscript text
    #not necessary for anyone else trying to use this

    if model_name == 'Mixed':
        ds.attrs['model_name'] = 'SPACE'

    if model_name == 'Detachment Limited':
        ds.attrs['model_name'] = 'SPM'
    
    nx = ds_attrs['nx']
    ny = ds_attrs['ny']
    dx = ds_attrs['dx']

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
    
    if model_name == 'Mixed' or model_name == 'SPACE':
        
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
    
    if model_name == 'Mixed' or model_name == 'SPACE':
        
        #add sediment data for SPACE model runs
        channel_sed_depth = mg.at_node['soil__depth'][channel_dist_ids]
        df['channel_sed_depth'] = channel_sed_depth
        
        channel_Es = mg.at_node['sediment__erosion'][channel_dist_ids]  
        df['channel_Es'] = channel_Es
    
    #remove outlet node      
    df_out = df.iloc[1: , :]
        
    return mg, df_out


        
#%%

def plot_channel_prf(df, ds, plot_time, plot_sed, plot_ero, ax=None):
    
    
    """
    Function runs channel profiler, calculates steepness, and calculates chi for a given time in xarray dataset

    Parameters
    ----------
    df : dataframe of channel profile data created using calc_main_channel function
    
    ds : xarray dataset of saved model output

    plot_time : the desired time point to plot from the xarray dataset
    
    plot_sed : If True, plot sediment thickness on a secondary y-axis
    
    plot_ero: If True, plot erosion rates on a secondary y-axis
    
    ax : axis to plot on (optional)
    
    Note - the profile sediment or erosion rate should only be plotted one at a time

    Returns
    -------
    ax : figure axes - plot can be further modified after calling function
    

    """
    
    if ax is None:
        ax = plt.gca()
        
    
    model_name = ds.attrs['model_name']

    #update model names to make plot labels consistent with manuscript text
    #not necessary for anyone else trying to use this

    if model_name == 'Mixed':
        ds.attrs['model_name'] = 'SPACE'

    if model_name == 'Detachment Limited':
        ds.attrs['model_name'] = 'SPM'
    
    #new colormap using seaborn
    lith_cmap = sns.color_palette("Paired", 10, as_cmap=True)

    #Convert time to desired units for labels
    plot_time_kyr = int(plot_time / 1000)
    plot_time_myr = plot_time / 1000000
    
    #group dataframe by rock type
    groups = df.groupby('channel_rock_id')
    
    #make the figure
    #fig = plt.figure(constrained_layout=True, figsize=(14, 4), dpi=300)
    
    
    for name, group in groups:
        
        if name %2 == 0:
        
            ax.plot(group.channel_dist, group.channel_elev, 
                    marker='o', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
        else:
            ax.plot(group.channel_dist, group.channel_elev, 
                    marker='s', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
        
    legend_elements_dtch = [Line2D([0], [0], marker='o', color='w', label='Soft Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Hard Rock',
                          markerfacecolor='dimgrey', markersize=10)]
    

  
    ax.legend(title="Layer ID")
    ax.set_xlabel('Distance Upstream (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_ylim(top=600)
    
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
 
    

    ax.set_xlim(left=0)
    
    if plot_sed == True:
        
        legend_elements_sed = [Line2D([0], [0], marker='o', color='w', label='Soft Rock',
                              markerfacecolor='dimgrey', markersize=10),
                       Line2D([0], [0], marker='s', color='w', label='Hard Rock',
                              markerfacecolor='dimgrey', markersize=10),
                       Line2D([0], [0], color='dimgrey', lw=2, label='Sediment Depth')]  
        
        
        if model_name == 'Mixed' or model_name == 'SPACE': #either naming convention will work
            #Add sediment depth to channel profile w/ secondary y-axis
            title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_myr} myr, V={ds.attrs['v_s']}"
            ax1=ax.twinx()
            ax1.plot(df['channel_dist'], df['channel_sed_depth'], '-', color = 'dimgrey', label = 'Sediment Thickness')
            ax1.set_ylabel("Sediment Thickness H (m)")
            ax1.set_ylim(top=2.25)
            #ax.legend(loc='best')
            
            ax1.tick_params(axis="x", labelsize=12)
            ax1.tick_params(axis="y", labelsize=12)
            ax1.xaxis.label.set_size(12)
            ax1.yaxis.label.set_size(12)
            
            ax1.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
            
            ax.legend(handles=legend_elements_sed, loc='best')
            
        
        if model_name == 'Detachment Limited' or  model_name == 'SPM':
            print('No sediment to plot')

            
        
    
    if plot_ero == True and plot_sed == True:
        raise Exception('Cannot plot sediment and erosion rate superimposed on the same figure')
    
    if plot_ero == True:
        
        ax1=ax.twinx()
        
        
        legend_elements_ero = [Line2D([0], [0], marker='o', color='w', label='Soft Rock',
                              markerfacecolor='dimgrey', markersize=10),
                       Line2D([0], [0], marker='s', color='w', label='Hard Rock',
                              markerfacecolor='dimgrey', markersize=10),
                       Line2D([0], [0], color='b', lw=2, label='Bedrock Erosion Rate'),
                       Line2D([0], [0], color='r', lw=2, label='Sediment Entrainment Rate'),
                       Line2D([0], [0], color='dimgrey', linestyle='dashed', lw=2, label='Uplift Rate')]
        
        if model_name == 'Detachment Limited' or  model_name == 'SPM':
            
            title_string = f"{model_name}, Time={plot_time_myr} myr"
            
            ax.legend(handles=legend_elements_dtch, loc='best')
            
           
            ax1.plot(df['channel_dist'], df['channel_Er'], '-', color = 'blue', label = 'Bedrock Incision Rate')
            ax1.set_ylabel("Incision/Entrainment/Uplift Rate (m/yr)")
            
            
            ax1.plot(df['channel_dist'], df['uplift'], '--', color = 'dimgrey', label = 'Uplift Rate')

        
        if model_name == 'SPACE'or model_name == 'Mixed':
            
            
            v_s_round = np.around(ds.attrs['v_s'])       
            title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_myr} myr, V={v_s_round} m/yr"
            
            ax.legend(handles=legend_elements_ero, loc='best')
            
            #Add sediment depth to channel profile w/ secondary y-axis

            ax1.plot(df['channel_dist'], df['channel_Es'], '-', color = 'red', label = 'Sediment Entrainment Rate')
            
            
            ax1.plot(df['channel_dist'], df['uplift'], '--', color = 'dimgrey', label = 'Uplift Rate')
            
            ax1.plot(df['channel_dist'], df['channel_Er'], '-', color = 'blue', label = 'Bedrock Erosion Rate')
            
            ax1.set_ylabel("Incision/Uplift Rate (m/yr)")
            
    else:
        #title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_kyr} kyr"
        title_string = f"Main Channel Profile, {model_name} Model, Time={plot_time_myr} myr"
   
    ax.set_title(title_string)
    
    
    return ax


#%%Calculate expected steepness

def calc_ksn_expected(ds):
    
    """
    Function calculates the expected channel steepness value for the soft and hard rock layers
    Calculation uses equation 6 for stream power model runs and equation 7 for space model runs
    
    Parameters
    ----------
    ds : xarray dataset with model output 

    Returns
    -------
    out : ksn_soft, ksn_hard
            ksn_soft : expected steepness in soft rock layers
            ksn_hard : expected steepness in hard rock layers
    """
    
    #read in variables from xarray dataset metadata
    model_name = ds.attrs['model_name']
    K_soft = ds.attrs['K_soft']
    K_hard = K_soft / ds.attrs['K_ratio']
    n = ds.attrs['n_sp']
    
        
    #calculate expected ksn using equation 6 for SPM runs
    if model_name == 'SPM' or model_name == 'Detachment Limited':
        U = ds.attrs['fsc_uplift']
        ksn_soft = (U/K_soft)**(1/n)
        ksn_hard = (U/K_hard)**(1/n)
    
    #Calculate expected ksn using equation 7 for SPACE model runs
    if model_name == 'SPACE' or model_name == 'Mixed':
        
        U = ds.attrs['space_uplift']
        V = ds.attrs['v_s']
        K_sed = ds.attrs['K_sed']
        ksn_soft = (U * ((V/K_sed) + (1/K_soft)))**(1/n)
        ksn_hard = (U * ((V/K_sed) + (1/K_hard)))**(1/n)
        
    return ksn_soft, ksn_hard
        
#%%

def plot_ksn(ds, plot_time, ax=None):
    
    if ax is None:
        ax = plt.gca()
    
    ksn_soft, ksn_hard = calc_ksn_expected(ds)
    
    mg, df = calc_main_channel(ds, plot_time)
    
    lith_cmap = sns.color_palette("Paired", 10, as_cmap=True)
    
    groups = df.groupby('channel_rock_id')
    
    ax.axhline(y=ksn_soft ,c="k")
    ax.axhline(y=ksn_hard ,c="dimgrey", linestyle='--')
    
    
    #Set the marker for each point based on layer number 
    for name, group in groups:
        
        #Even-numbered layers are soft rock and get round markers
        if name %2 == 0:
        
            ax.plot(group.channel_dist, group.channel_ksn, 
                    marker='o', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
       
        #Odd-numbered layers are hard rock and are plotted with square markers
        else:
            ax.plot(group.channel_dist, group.channel_ksn, 
                    marker='s', markeredgewidth=0.5, markeredgecolor='dimgrey', 
                    linestyle='', markersize=5, label=name, 
                    color=lith_cmap(int(name)-1))
            
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Soft Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Hard Rock',
                          markerfacecolor='dimgrey', markersize=10),
                   Line2D([0], [0], color='dimgrey', lw=2, label='Hard Rock Exp. ksn', linestyle='--'),
                   Line2D([0], [0], color='k', lw=2, label='Soft Rock Exp. ksn')]
    
    
    ax.legend(handles=legend_elements, loc='upper left')

    
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
        
    title_string = "Main Channel Steepness Index"
    #ax3.legend(title="Layer ID", loc=1, ncol=2)
    ax.set_xlabel('Distance Upstream (m)')
    ax.set_ylabel("$k_{sn}$", style='italic')
    ax.set_title(title_string)
    ax.set_ylim(bottom=0, top=350)
    
    return ax
#%%

