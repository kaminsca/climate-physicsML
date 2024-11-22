#!/usr/bin/env python
import sys
# usage: python predict_and_visualize.py input_model_path out_figures_dir_path
input_model_path = sys.argv[1]
out_figures_dir_path = sys.argv[2]


import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle

import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from combined_loss_model import CustomModel  # Assuming your class is in this file

def plot_contour_map(lat, lon, values, cmap='viridis', vmin=None, vmax=None, clev=11, title='', save_path=None):
    """
    Plot a contour map with latitude, longitude, and values on a global map.

    Parameters:
    - lat: Array-like, latitude values.
    - lon: Array-like, longitude values.
    - values: Array-like, data values corresponding to lat/lon.
    - cmap: Colormap for the plot. Default is 'viridis'.
    - vmin: Minimum value for the colormap. Default is min(values).
    - vmax: Maximum value for the colormap. Default is max(values).
    - clev: Number of contour levels. Default is 11.
    - title: Title of the plot.
    - save_path: Path to save the plot. If None, the plot is shown interactively.
    """
    # Set up the plot
    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.PlateCarree()},
        figsize=(10, 5)
    )
    
    # Set global map features
    ax.set_global()
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Handle colormap limits and contour levels
    vmin = vmin if vmin is not None else np.min(values)
    vmax = vmax if vmax is not None else np.max(values)
    clevels = np.linspace(vmin, vmax, clev)

    # Plot the contour map
    contour = ax.tricontourf(
        lon, lat, values, levels=clevels, cmap=cmap, transform=ccrs.PlateCarree()
    )

    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', shrink=0.5, pad=0.05)
    cbar.set_label('Value')

    # Add title
    ax.set_title(title, fontsize=14)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    # close completely:
    plt.clf()

def load_ds_and_dso_from_file(ds_file, vars_mli, vars_mlo, vars_mlo_0, mli_mean, mli_max, mli_min, mlo_scale):
    dso_file = ds_file.replace(".mli.", ".mlo.")
    ds = xr.open_dataset(ds_file, engine="netcdf4")
    dso = xr.open_dataset(dso_file, engine="netcdf4")

    state_t_dso = dso['state_t']
    state_q0001_ds0 = dso['state_q0001']

    for kvar in ['state_t','state_q0001','state_q0002', 'state_q0003', 'state_u', 'state_v']:
        dso[kvar.replace('state','ptend')] = (dso[kvar] - ds[kvar])/1200 # timestep=1200[sec]

    epsilon = 1e-8
    ds = (ds - mli_mean) / (mli_max - mli_min + epsilon)

    dso = dso * mlo_scale

    index=59
    for var in vars_mlo_0:
        # print(var)
        if len(dso[var].shape) == 2:
            dso[var] = dso[var][index]
            # print("changed")
    dso=dso[vars_mlo_0]
    # now lets add the additional variables: state_t
    dso["state_t"] = state_t_dso[index]
    dso["state_q0001"] = state_q0001_ds0[index]
    # scale it knowing that max T on earth in Kelvin is 329.85 and min is 174.55, and the mean is 
    # Define the min and max temperature values in Kelvin
    global min_temp, max_temp
    min_temp = 174.55
    max_temp = 329.85

    # Normalize the "ptend_t" variable in the dataset
    # dso["ptend_t"] = dso["ptend_t"] / (max_temp - min_temp)           # remove "state_xxxx"
    dso = dso[vars_mlo]

    for var in vars_mli:
        # print(var)
        if len(ds[var].shape) == 2:
            ds[var] = ds[var][index]
            # print("changed")
    ds=ds[vars_mli]

    # flatten input variables #
    #ds = ds.stack({'batch':{'sample','ncol'}})
    ds = ds.stack({'batch':{'ncol'}})
    ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
    #dso = dso.stack({'batch':{'sample','ncol'}})
    dso = dso.stack({'batch':{'ncol'}})
    dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')
    return ds.values, dso.values

def build_model():
    initializer = tf.keras.initializers.GlorotUniform()
    # input_length = 2 * 60 + 5
    input_length = len(vars_mli)
    # output_length_lin = 2 * 60 - 118
    output_length_relu = len(vars_mlo)
    # output_length = output_length_lin + output_length_relu
    # output_length =  output_length_lin + output_length_relu
    input_layer = keras.layers.Input(shape=(input_length,), name='input')
    hidden_0 = keras.layers.Dense(768, activation='relu', kernel_initializer=initializer)(input_layer)
    hidden_1 = keras.layers.Dense(640, activation='relu', kernel_initializer=initializer)(hidden_0)
    hidden_2 = keras.layers.Dense(512, activation='relu', kernel_initializer=initializer)(hidden_1)
    hidden_3 = keras.layers.Dense(640, activation='relu', kernel_initializer=initializer)(hidden_2)
    hidden_4 = keras.layers.Dense(640, activation='relu', kernel_initializer=initializer)(hidden_3)
    output_pre = keras.layers.Dense(output_length_relu, activation='elu', kernel_initializer=initializer)(hidden_4)
    output_relu = keras.layers.Dense(output_length_relu, activation='relu', kernel_initializer=initializer)(output_pre)

    model = keras.Model(input_layer, output_relu, name='Emulator')
    model.summary()
# print dimensions fo input and output layers of model and exit
    print("Model dimensions of input and output layers:")
    print(model.input.shape)
    print(model.output.shape)
    return model

def load_model(model_path):
    """Load the trained model."""
    print(f"Loading model from {model_path}...")
    # model = keras.models.load_model(model_path, compile=False, custom_objects={'CustomModel': CustomModel})
    custom_objects = {'CustomModel': CustomModel}
    model = keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    # model.summary()
    return model

# def preprocess_data(input_file, vars_mli, mli_mean, mli_max, mli_min):
#     """Preprocess input data for prediction."""
#     ds = xr.open_dataset(  input_file , engine="netcdf4")
#     ds = ds[vars_mli]
#     # normalize
#     epsilon = 1e-8
#     ds = (ds - mli_mean) / (mli_max - mli_min + epsilon)
#     index=59
#     for var in vars_mli:
#         # print(var)
#         if len(ds[var].shape) == 2:
#             ds[var] = ds[var][index]
#             # print("changed")
#     ds=ds[vars_mli]
#     # print(ds.shape)
    

    # return ds.stack({"batch": {"ncol"}}).to_stacked_array(
    #     "mlvar", sample_dims=["batch"], name="mli"
    # ).values

# def postprocess_predictions(predictions, mlo_scale):
#     """Postprocess model predictions to scale them back to original units."""
#     predictions_rescaled = predictions / mlo_scale
#     return predictions_rescaled
def postprocess_predictions(predictions, mlo_scale, vars_mlo):
    """Postprocess model predictions to scale them back to original units."""
    # Ensure mlo_scale contains only the relevant variables
    mlo_scale = mlo_scale[vars_mlo]

    # Stack scaling factors to align with predictions
    mlo_scale_stacked = mlo_scale.stack({'new_dim': mlo_scale.dims}).to_stacked_array(
        'mlvar', sample_dims=["new_dim"], name='mlo'
    )
    # Extract scaling factors as numpy array
    scaling_factors = mlo_scale_stacked.values
    
    # Ensure scaling factors align with predictions
    if scaling_factors.shape[0] != predictions.shape[1]:
        raise ValueError(
            f"Shape mismatch: scaling_factors shape {scaling_factors.shape} "
            f"does not match predictions shape {predictions.shape}. Ensure the variables align correctly."
        )
    
    # Rescale predictions
    predictions_rescaled = predictions * scaling_factors
    return predictions_rescaled

def make_predictions(input_file, model, vars_mli, vars_mlo, mli_mean, mli_max, mli_min, mlo_scale, output_dir):
    """Make predictions and save results."""
    print(f"Processing input file: {input_file}")
    ds, dso = load_ds_and_dso_from_file(input_file, vars_mli, vars_mlo, vars_mlo_0, mli_mean, mli_max, mli_min, mlo_scale)
    predictions = model.predict(ds)
    return predictions, ds, vars_mlo, mlo_scale, dso
    predictions_rescaled = postprocess_predictions(predictions, mlo_scale, vars_mlo)

    # Load ground truth
    ds_ground_truth = xr.open_dataset(
        input_file.replace(".mli.", ".mlo."), engine="netcdf4"
    )
    ds_ground_truth = ds_ground_truth[vars_mlo].stack({"batch": {"lev"}}).to_stacked_array(
        "mlvar", sample_dims=["batch"], name="mlo"
    ).values

    # Save predictions and ground truth for comparison
    output_file = os.path.join(output_dir, "predictions_vs_ground_truth.nc")
    print(f"Saving predictions and ground truth to {output_file}")
    xr.Dataset(
        {
            "predictions": (["batch", "mlvar"], predictions_rescaled),
            "ground_truth": (["batch", "mlvar"], ds_ground_truth),
        }
    ).to_netcdf(output_file)

    return predictions_rescaled, ds_ground_truth

def plot_comparison(predictions, ground_truth, vars_mlo, output_dir):
    """Visualize predictions vs ground truth for key variables."""
    for i, var in enumerate(vars_mlo):
        plt.figure(figsize=(12, 8))
        plt.plot(ground_truth[:, i], label=f"Ground Truth ({var})", alpha=0.7)
        plt.plot(predictions[:, i], label=f"Predictions ({var})", alpha=0.7)
        plt.title(f"Comparison: {var}", fontsize=16)
        plt.xlabel("Batch", fontsize=14)
        plt.ylabel(var, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)

        output_file = os.path.join(output_dir, f"comparison_{var}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved comparison plot: {output_file}")


# In[46]:


with open("file_list.pkl", "rb") as f:
    all_files = pickle.load(f)
    print("Loaded file list from file_list.pkl.")


# In[47]:


all_files[0]


# In[48]:


# Define paths
input_file = all_files[-1]  # Replace with your .mli file
groundtruth_file = input_file.replace(".mli.", ".mlo.")  # Replace with your .mlo file
norm_path = "/home/alvarovh/code/cse598_climate_proj/ClimSim/preprocessing/normalizations/"  # Replace with your normalization files path
# Variables for input (mli) and output (mlo)
vars_mli  = ['cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_LWUP', 'cam_in_OCNFRAC', 'cam_in_SNOWHICE', 'cam_in_SNOWHLAND', 'pbuf_COSZRS', 'pbuf_LHFLX', 'pbuf_SHFLX', 'pbuf_SOLIN', 'pbuf_TAUX', 'pbuf_TAUY', 'state_pmid', 'state_ps', 'state_q0001', 'state_q0002', 'state_q0003', 'state_t', 'state_u', 'state_v', 'pbuf_CH4', 'pbuf_N2O', 'pbuf_ozone']

vars_mlo      = ['ptend_t','ptend_q0001','ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v',
                    'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 
                    'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD', "state_t", "state_q0001"]
vars_mlo_0   = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v',
                     'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 
                     'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

# Load normalization data
mli_mean = xr.open_dataset(os.path.join(norm_path, "inputs/input_mean.nc"))
mli_max = xr.open_dataset(os.path.join(norm_path, "inputs/input_max.nc"))
mli_min = xr.open_dataset(os.path.join(norm_path, "inputs/input_min.nc"))
mlo_scale = xr.open_dataset(os.path.join(norm_path, "outputs/output_scale.nc"))
fn_grid = '/home/alvarovh/code/cse598_climate_proj/ClimSim/grid_info/ClimSim_low-res_grid-info.nc'
ds_grid = xr.open_dataset(fn_grid, engine='netcdf4')
latitudes = ds_grid['lat'].values

# Load the trained model
model = load_model(input_model_path)


predictions, input_data, vars_mlo, mlo_scale, dso = make_predictions(
    input_file, model, vars_mli, vars_mlo, mli_mean, mli_max, mli_min, mlo_scale, out_figures_dir_path
)



prediction_figures_dirpath = f"{out_figures_dir_path}/predictions_figures/"
groundtruth_figures_dirpath = f"{out_figures_dir_path}/groundtruth_figures/"
error_figures_dirpath = f"{out_figures_dir_path}/error_figures/"
os.makedirs(prediction_figures_dirpath, exist_ok=True)
os.makedirs(groundtruth_figures_dirpath, exist_ok=True)
os.makedirs(error_figures_dirpath, exist_ok=True)

print("Will save prediction figures to: ", prediction_figures_dirpath)

with open("error_log.txt", "w") as f:
    f.write("Starting with predictions\n")
success_predictions_count = 0
for i in range(0, len(vars_mlo)):
    try:
        plot_contour_map(ds_grid['lat'], ds_grid['lon'], predictions[:,i], title=f'Predictions: {vars_mlo[i]}', save_path=f'{prediction_figures_dirpath}/predictions_{vars_mlo[i]}.png')
        print(f"Plotted predictions for {vars_mlo[i]} in {prediction_figures_dirpath}/predictions_{vars_mlo[i]}.png")
        success_predictions_count += 1
    except Exception as e:
        with open("error_log.txt", "a") as f:
            f.write(f"Error plotting {vars_mlo[i]}\n")
            f.write(str(e) + "\n")

print("done with predictions\nStarting with ground truth plots:\n")
success_groundtruth_count = 0
for i in range(0, len(vars_mlo)):
    try:
        # plot_contour_map(ds_grid['lat'], ds_grid['lon'], predictions[:,i], title=f'Predictions: {vars_mlo[i]}', save_path=f'out_model_figures/predictions_{vars_mlo[i]}.png')
        plot_contour_map(ds_grid['lat'], ds_grid['lon'], dso[:,i], title=f'Ground Truth: {vars_mlo[i]}', save_path=f'{groundtruth_figures_dirpath}/groundtruth_{vars_mlo[i]}.png')
        print(f"Plotted ground truth for {vars_mlo[i]} in {groundtruth_figures_dirpath}/groundtruth_{vars_mlo[i]}.png")
        success_groundtruth_count += 1
    except:
        with open("error_log.txt", "a") as f:
            f.write(f"Error plotting {vars_mlo[i]}\n")

print("done with ground truth\nStarting with error plots:\n")
success_error_count = 0
for i in range(0, len(vars_mlo)):
    try:
        plot_contour_map(ds_grid['lat'], ds_grid['lon'], dso[:,i] - predictions[:,i], title=f'Error: {vars_mlo[i]}', save_path=f'{error_figures_dirpath}/error_{vars_mlo[i]}.png')
        print(f"Plotted error for {vars_mlo[i]} in {error_figures_dirpath}/error_{vars_mlo[i]}.png")
        success_error_count += 1
    except:
        with open("error_log.txt", "a") as f:
            f.write(f"Error plotting {vars_mlo[i]}\n")
print("N successful plots ground truth: ", success_groundtruth_count)
print("N successful plots predictions: ", success_predictions_count)
print("N successful plots errors: ", success_error_count)
# EEEND
# # In[ ]:


# # In[ ]:


# plot_contour_map(ds_grid['lat'], ds_grid['lon'], dso[:,i].values, title='Predictions')


# # In[ ]:


# i = 0
# plot_contour_map(ds_grid['lat'], ds_grid['lon'], predictions[:,i], title='Predictions')


# # In[ ]:


# i = 8
# plot_contour_map(ds_grid['lat'], ds_grid['lon'], dso.values[:,i], title=vars_mlo[i])


# # In[ ]:


# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.io.shapereader import Reader
# import cartopy.io.shapereader as shpreader

# # Set up the map projection and area of interest
# fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
# ax.set_extent([-90, -80, 40, 50], crs=ccrs.PlateCarree())  # Michigan region

# # Add geographic features
# ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgrey")
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.LAKES, alpha=0.5)
# ax.add_feature(cfeature.RIVERS)

# # Plot Michigan boundary from shapefile
# shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces_lakes')
# for record in Reader(shapefile).records():
#     if record.attributes['name'] == 'Michigan':  # Filter for Michigan
#         geometry = record.geometry
#         ax.add_geometries([geometry], ccrs.PlateCarree(), edgecolor='blue', facecolor='none', linewidth=1.5, label="Michigan Boundary")

# # Define the grid box dimensions (5.625° x 5.625°) in Michigan area
# latitude = 45  # Center of Michigan latitude for demonstration
# longitude = -85  # Center of Michigan longitude for demonstration

# # Calculate the box corners
# lat_min = latitude - 5.625 / 2
# lat_max = latitude + 5.625 / 2
# lon_min = longitude - 5.625 / 2
# lon_max = longitude + 5.625 / 2

# # Plot the box on the map
# ax.plot(
#     [lon_min, lon_max, lon_max, lon_min, lon_min],
#     [lat_min, lat_min, lat_max, lat_max, lat_min],
#     color="red",
#     transform=ccrs.PlateCarree(),
#     label="5.625° x 5.625° Grid"
# )

# # Add labels and legend
# ax.set_title("5.625° x 5.625° Grid Box Over Michigan")
# ax.legend(loc="lower left")

# plt.show()


# # In[ ]:


# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.io.shapereader import Reader
# import cartopy.io.shapereader as shpreader

# # Set up the map projection and area of interest
# fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
# ax.set_extent([-90, -80, 40, 50], crs=ccrs.PlateCarree())  # Michigan region

# # Add geographic features
# ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgrey")
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.LAKES, alpha=0.5)
# ax.add_feature(cfeature.RIVERS)

# # Plot Michigan boundary from shapefile
# shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces_lakes')
# for record in Reader(shapefile).records():
#     if record.attributes['name'] == 'Michigan':  # Filter for Michigan
#         geometry = record.geometry
#         ax.add_geometries([geometry], ccrs.PlateCarree(), edgecolor='blue', facecolor='none', linewidth=1.5, label="Michigan Boundary")

# # Define the grid box dimensions (1.40625° x 1.40625°) in Michigan area
# latitude = 45  # Center of Michigan latitude for demonstration
# longitude = -85  # Center of Michigan longitude for demonstration

# # Calculate the box corners
# lat_min = latitude - 1.40625 / 2
# lat_max = latitude + 1.40625 / 2
# lon_min = longitude - 1.40625 / 2
# lon_max = longitude + 1.40625 / 2

# # Plot the box on the map
# ax.plot(
#     [lon_min, lon_max, lon_max, lon_min, lon_min],
#     [lat_min, lat_min, lat_max, lat_max, lat_min],
#     color="green",
#     transform=ccrs.PlateCarree(),
#     label="1.40625° x 1.40625° Grid"
# )

# # Add labels and legend
# ax.set_title("1.40625° x 1.40625° Grid Box Over Michigan")
# ax.legend


# # In[ ]:


# plot_lat_lon_data(ds_grid['lat'], ds_grid['lon'], input_data[:,-1], title='Latitude', save_path=None)


# # In[ ]:


# def plot_metric_map(ds_plotdata, kmetric='R2', kvar='cam_out_PRECSC', klev=-1,
#                     cmap='jet', vmin=0., vmax=1., clev=11,
#                     fn_fig=''):
#     # Define the figure and each axis for the 2 rows and 3 columns
#     fig, ax = plt.subplots(nrows=1,ncols=1,
#                             subplot_kw={'projection': ccrs.Robinson(central_longitude=179.5)},
#                             figsize=(8.,3.))

#     # axs is a 2 dimensional array of `GeoAxes`.  We will flatten it into a 1-D array
#     ax=ax.flatten()

#     #Loop over all of the models
#     # for k, kmodel in enumerate(plot_this_models):
#         # ax = axs[k]
#     # ax.set_global()

#     # ds_plotdata = PLOTDATA[kmetric][kmodel]

#     x = ds_grid['lon']
#     y = ds_grid['lat']
#     z = ds_plotdata[vars_mlo[0]]
    
#     # if klev>=0: z = z.isel(lev=klev)
    
#     # if kmetric=='R2':
#     #     z = z.where(z>0,-1e-5)
#     clevels = np.linspace(vmin,vmax,clev)
#     h = ax.tricontourf(x, y, z, transform=ccrs.PlateCarree(),
#                         levels = clevels, 
#                         cmap=cmap, 
#                         extend='min' if kmetric=='R2' else 'both')
#     h.cmap.set_under('silver')
#     if kmetric=='R2':        h.set_clim(1e-5, 1.)

#     # # Contour plot
#     # cs=axs[i].contourf(lons,ds['lat'],data,
#     #                   transform = ccrs.PlateCarree(),
#     #                   cmap='coolwarm',extend='both')

#     # ax.set_title(f'({abc[k]}) {kmodel}')
#     ax.coastlines()

# # # fig.suptitle(kvar, y=1.025)
# # fig.suptitle(vars_longname[kvar] + (f' (level={klev})' if klev>=0 else ''),
# #              y=1.02)
# tmp = vars_longname[kvar]
# if klev>=0:
#     tmp = tmp + (f' (level={klev})' if klev>=0 else '')
# tmp = tmp.split(',')
# tmp = f'{tmp[1]}\n{tmp[0]}'
# fig.text(.087,.5, tmp,ha='center', va='center', rotation=90)
    
# fig.subplots_adjust(right=0.9)
# cbar_ax = fig.add_axes([0.925, 0.145, 0.020, 0.71])
# cbar_ax.set_title(kmetric)
# fig.colorbar(h, cax=cbar_ax, ticks=[0,0.25,0.5,0.75,1])
# fig.set_facecolor('w')

# if len(fn_fig)>0:
#     fig.savefig(fn_fig)


# # In[ ]:


# groundtruth[vars_mlo]


# # In[ ]:


# # save to file
# np.savez("predictions.npz", predictions=predictions, input_data=input_data, vars_mlo=vars_mlo, groundtruth=groundtruth)
# # can be loaded with np.load("predictions.npz")


# # In[ ]:





# # In[ ]:


# input_data.shape


# # In[ ]:


# predictions.shape


# # In[ ]:


# mlo_scale


# # In[ ]:


# # Ensure output directory exists
# os.makedirs(output_dir, exist_ok=True)

# # Make predictions
# # predictions, input_data, vars_mlo, mlo_scale = make_predictions(
# #     input_file, model, vars_mli, vars_mlo, mli_mean, mli_max, mli_min, mlo_scale, output_dir
# # )
# predictions, ground_truth = make_predictions(
#     input_file, model, vars_mli, vars_mlo, mli_mean, mli_max, mli_min, mlo_scale, output_dir
# )


# # In[ ]:


# input_data.shape


# # In[ ]:


# predictions.shape


# # In[ ]:


# mlo_scale


# # In[ ]:


# #xarray to numpy array
# mloscalenp= mlo_scale.to_array().values
# mloscalenp.shape


# # In[ ]:


# mloscalenp


# # In[ ]:


# # Flatten `mlo_scale` to extract variables
# scaling_factors = []

# # Add variables with multiple levels
# for var in ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v']:
#     scaling_factors.extend(mlo_scale[var].values)  # Add all levels

# # Add scalar variables
# for var in ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 
#             'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']:
#     scaling_factors.append(mlo_scale[var].values)

# # Convert to numpy array
# scaling_factors = np.array(scaling_factors)

# # Ensure shape matches predictions
# if scaling_factors.shape[0] != predictions.shape[1]:
#     raise ValueError(f"Mismatch in scaling factors ({scaling_factors.shape[0]}) and predictions ({predictions.shape[1]})!")


# # In[ ]:





# # In[ ]:





# # In[10]:


# import xarray as xr

# root_climsim_dirpath = "/home/alvarovh/code/cse598_climate_proj/ClimSim/"
# norm_path = f"{root_climsim_dirpath}/preprocessing/normalizations/"


# # In[11]:


# mlo = xr.open_dataset(groundtruth_file, engine="netcdf4")
# mli = xr.open_dataset(input_file, engine="netcdf4")


# # In[12]:


# mlo


# # In[13]:


# vars_mlo_0=list(mlo.data_vars.keys())[2:]


# # In[14]:


# vars_mlo_0


# # In[15]:


# vars_mlo_0


# # In[16]:


# vars_mlo      = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v',
#                      'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 
#                      'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD', 'state_t', 'state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_v']


# # In[17]:


# mlo_scale


# # In[19]:


# var='ptend_t'
# (list(mlo[var].values))


# # In[20]:


# vars_mlo      = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v',
#                  'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 
#                  'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']


# # In[21]:


# vars_mlo_dims = [(mlo_scale[var].values.size) for var in vars_mlo]


# # In[49]:


# mlo["state_t"].values


# # In[22]:


# vars_mlo_dims


# # In[45]:


# mlo.dims


# # In[ ]:





# # In[26]:


# index=59
# for var in vars_mlo_0:
#     print(var)
#     if len(dso[var].shape) == 2:
#         dso[var] = dso[var][index]
#         print("changed")
# dso=dso[vars_mlo_0]



# # In[7]:


# dso


# # In[8]:


# # Load normalization data
# mli_mean = xr.open_dataset(os.path.join(norm_path, "inputs/input_mean.nc"))
# mli_max = xr.open_dataset(os.path.join(norm_path, "inputs/input_max.nc"))
# mli_min = xr.open_dataset(os.path.join(norm_path, "inputs/input_min.nc"))
# mlo_scale = xr.open_dataset(os.path.join(norm_path, "outputs/output_scale.nc"))

# mlo_scale
# # mlo_scale = mlo_scale[vars_mlo]


# # In[10]:


# list(mlo_scale.keys())


# # In[119]:


# dso['ptend_t'] = (dso['state_t'] - ds['state_t']) / 1200  # T tendency [K/s]
# dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001']) / 1200  # Q tendency [kg/kg/s]
# dso['ptend_q0002'] = (dso['state_q0002'] - ds['state_q0002'])/1200 # Q2 tendency [kg/kg/s]
# dso['ptend_q0003'] = (dso['state_q0003'] - ds['state_q0003'])/1200 # Q3 tendency [kg/kg/s]
# dso['ptend_u'] = (dso['state_u'] - ds['state_u'])/1200 # U tendency [m/s/s]
# dso['ptend_v'] = (dso['state_v'] - ds['state_v'])/1200 # V tendency [m/s/s] 


# # In[143]:


# vars_mlo=list(dso.data_vars.keys())[2:]
# vars_mlo_dims = [(i.values.size) for i in mlo_scale.data_vars.values()]
# vars_mli = list(ds.data_vars.keys())[2:]
# vars_mli_dims = [(i.values.size) for i in mli_min.data_vars.values()]



# # In[144]:


# vars_mli


# # In[145]:


# vars_mlo


# # In[114]:


# ds


# # In[115]:


# mli_min


# # In[102]:


# index=59
# for var in vars_mli:
#     print(ds[var].shape)
#     if len(ds[var].shape) == 2:
#         ds[var] = ds[var][index]
#         print("changed")
# ds=ds[vars_mli]


# # In[103]:


# dso


# # In[104]:





# # In[105]:


# ds


# # In[107]:


# dso


# # In[57]:


# len(ds[mli_vars[22]].shape)


# # In[ ]:





# # In[35]:


# mli_vars, mli_dims


# # In[31]:


# mlo_vars = list(mlo_scale.data_vars.keys())

# mlo_dims = [(i.values.size) for i in mlo_scale.data_vars.values()]


# # In[33]:


# mlo_vars,mlo_dims


# # In[ ]:


# mlo_scale # this is an xarray, lets get a list of how many levels each variable has



# # In[26]:


# d=[(i.values.size) for i in mlo_scale.data_vars.values()]


# # In[10]:


# list(mlo_scale.data_vars.keys())

