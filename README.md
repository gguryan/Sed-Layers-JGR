# Sed-Layers-JGR

Code for manuscript "Sediment cover modulates landscape erosion patterns and channel steepness in layered rocks: Insights from the SPACE Model" submitted to JGR: Earth Surface on 10/28/23

Both driver scripts require a .txt file with model inputs and a .netcdf file with the initial topography/drainage network.
Readers are referred to the [manual by Shobe et al. 2017](https://figshare.com/articles/dataset/pub_shobe_etal_GMD/5193478/1) for a more complete description of input parameters 

Model output is stored in an x-array dataset and automatically output into a netCDF at a user-defined interval. Before running the driver script, edit the ds_file string with the desired filename for the output dataset and set the save_int (default is to save output every 1000 years on 50x50 model cell grids and every 10000 years on 200x200 cell model grids). 
