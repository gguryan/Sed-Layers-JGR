# Sed-Layers-JGR

Code for manuscript "Sediment cover modulates landscape erosion patterns and channel steepness in layered rocks: Insights from the SPACE Model" submitted to Journal of Geophysical Research: Earth Surface on 10/28/23 and currently availabile as a [preprint](https://essopenarchive.org/users/695126/articles/683952-sediment-cover-modulates-landscape-erosion-patterns-and-channel-steepness-in-layered-rocks-insights-from-the-space-model)

Both driver scripts require a .txt file with model inputs and a .netcdf file with some initial topography/drainage network.
Readers are referred to the [manual by Shobe et al. 2017](https://figshare.com/articles/dataset/pub_shobe_etal_GMD/5193478/1) for a more complete description of input parameters 

Model output is stored in an x-array dataset and automatically output into a netCDF at a user-defined interval. Before running the driver script, edit the ds_file string with the desired filename for the output dataset and set the save_int (default is to save output every 1,000 years on 50x50 model cell grids and every 10,000 years on 200x200 cell model grids). 

This model was developed using Landlab 2.6.0 and requires the SpaceLargeScaleEroder and PriorityFloodFlowRouter components. See the included environment.yml file for a complete list of packages and dependencies. 
