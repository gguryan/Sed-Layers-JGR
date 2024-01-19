# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:35:18 2024

@author: gjg882
"""

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

from plot_funcs import calc_main_channel
from plot_funcs import plot_channel_prf
from plot_funcs import calc_ksn_expected
from plot_funcs import plot_ksn


from landlab import RasterModelGrid
from landlab.plot import imshow_grid
from landlab.components import (FlowAccumulator, 
                                DepressionFinderAndRouter,
                                FastscapeEroder,
                                ChannelProfiler,
                                SteepnessFinder,
                                ChiFinder)

#%%
#file ID string will be used to name any saved plots
file_id = "sample_plot_SPACE"

#load in the model output
ds_file = 'Inputs/SPACE_output_50x50.nc'

ds = xr.open_dataset(ds_file) 

#input plot time manually  
plot_time = 1200000

#Can also read in model runtime from ds attributes (all input parameters are saved in the ds metadata)
#plot_time = ds.attrs['space_runtime'] #for space model runs
#plot_time = ds.attrs['fsc_runtime'] #for SPM runs


#%%

#Calculate main channel, return landlab model grid with channel profiler fields, also returns pandas df with channel data
mg, df = calc_main_channel(ds, plot_time)

fig1 = plt.figure()
plot_ksn(ds, plot_time)

fig2 = plt.figure()
plot_channel_prf(df, ds, plot_time, plot_sed=True, plot_ero=False, ax=None)
