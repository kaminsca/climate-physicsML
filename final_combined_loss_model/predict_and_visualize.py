#!/usr/bin/env python
# import sys
# # usage: python predict_and_visualize.py input_model_path out_figures_dir_path
# input_model_path = sys.argv[1]
# out_figures_dir_path = sys.argv[2]


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
from combined_loss_model import CustomModel, build_model

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

# def load_model(model_path):
# if we want to pass other objects to load_model, we can use **kwargs and when we call the function we can pass the objects
# like this: load_model(model_path, custom_objects={'CustomModel': CustomModel})
def load_model(model_path, **kwargs):
    """Load the trained model."""
    # if exclude_these_losses and constant_lambdas were passed, we need to pass them to the model
    print(f"Loading model from {model_path}...")
    # model = keras.models.load_model(model_path, compile=False, custom_objects={'CustomModel': CustomModel})
    custom_objects = {'CustomModel': CustomModel}
    model = keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    # model.summary()
    return model

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

def make_predictions(input_file, model, vars_mli, vars_mlo, mli_mean, mli_max, mli_min, mlo_scale, output_dir, vars_mlo_0):
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


def make_predictions_and_save_plots(input_model_path, figures_folder):
    with open("file_list.pkl", "rb") as f:
        all_files = pickle.load(f)
        print("Loaded file list from file_list.pkl.")

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

    constant_lambdas = ["nonneg", "radiation"]
    exclude_these_losses = ["nonneg", "radiation"]
    model = load_model(input_model_path, constant_lambdas=constant_lambdas, exclude_these_losses=exclude_these_losses)
    predictions, input_data, vars_mlo, mlo_scale, dso = make_predictions(
        input_file, model, vars_mli, vars_mlo, mli_mean, mli_max, mli_min, mlo_scale, figures_folder, vars_mlo_0
    )

    prediction_figures_dirpath = f"{figures_folder}/predictions_figures/"
    groundtruth_figures_dirpath = f"{figures_folder}/groundtruth_figures/"
    error_figures_dirpath = f"{figures_folder}/error_figures/"
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

            # lets get the error of state_t
            if vars_mlo[i] == "state_t":
                state_t_error = dso[:,i] - predictions[:,i]


        except:
            with open("error_log.txt", "a") as f:
                f.write(f"Error plotting {vars_mlo[i]}\n")
    print("N successful plots ground truth: ", success_groundtruth_count)
    print("N successful plots predictions: ", success_predictions_count)
    print("N successful plots errors: ", success_error_count)

    # lets return the error of state_t
    return state_t_error
    


# with open("file_list.pkl", "rb") as f:
#     all_files = pickle.load(f)
#     print("Loaded file list from file_list.pkl.")



# all_files[0]


# # In[48]:


# # Define paths
# input_file = all_files[-1]  # Replace with your .mli file
# groundtruth_file = input_file.replace(".mli.", ".mlo.")  # Replace with your .mlo file
# norm_path = "/home/alvarovh/code/cse598_climate_proj/ClimSim/preprocessing/normalizations/"  # Replace with your normalization files path
# # Variables for input (mli) and output (mlo)
# vars_mli  = ['cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_LWUP', 'cam_in_OCNFRAC', 'cam_in_SNOWHICE', 'cam_in_SNOWHLAND', 'pbuf_COSZRS', 'pbuf_LHFLX', 'pbuf_SHFLX', 'pbuf_SOLIN', 'pbuf_TAUX', 'pbuf_TAUY', 'state_pmid', 'state_ps', 'state_q0001', 'state_q0002', 'state_q0003', 'state_t', 'state_u', 'state_v', 'pbuf_CH4', 'pbuf_N2O', 'pbuf_ozone']

# vars_mlo      = ['ptend_t','ptend_q0001','ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v',
#                     'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 
#                     'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD', "state_t", "state_q0001"]
# vars_mlo_0   = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v',
#                      'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 
#                      'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

# # Load normalization data
# mli_mean = xr.open_dataset(os.path.join(norm_path, "inputs/input_mean.nc"))
# mli_max = xr.open_dataset(os.path.join(norm_path, "inputs/input_max.nc"))
# mli_min = xr.open_dataset(os.path.join(norm_path, "inputs/input_min.nc"))
# mlo_scale = xr.open_dataset(os.path.join(norm_path, "outputs/output_scale.nc"))
# fn_grid = '/home/alvarovh/code/cse598_climate_proj/ClimSim/grid_info/ClimSim_low-res_grid-info.nc'
# ds_grid = xr.open_dataset(fn_grid, engine='netcdf4')
# latitudes = ds_grid['lat'].values

# # Load the trained model
# model = load_model(input_model_path)


# predictions, input_data, vars_mlo, mlo_scale, dso = make_predictions(
#     input_file, model, vars_mli, vars_mlo, mli_mean, mli_max, mli_min, mlo_scale, out_figures_dir_path
# )



# prediction_figures_dirpath = f"{out_figures_dir_path}/predictions_figures/"
# groundtruth_figures_dirpath = f"{out_figures_dir_path}/groundtruth_figures/"
# error_figures_dirpath = f"{out_figures_dir_path}/error_figures/"
# os.makedirs(prediction_figures_dirpath, exist_ok=True)
# os.makedirs(groundtruth_figures_dirpath, exist_ok=True)
# os.makedirs(error_figures_dirpath, exist_ok=True)

# print("Will save prediction figures to: ", prediction_figures_dirpath)

# with open("error_log.txt", "w") as f:
#     f.write("Starting with predictions\n")
# success_predictions_count = 0
# for i in range(0, len(vars_mlo)):
#     try:
#         plot_contour_map(ds_grid['lat'], ds_grid['lon'], predictions[:,i], title=f'Predictions: {vars_mlo[i]}', save_path=f'{prediction_figures_dirpath}/predictions_{vars_mlo[i]}.png')
#         print(f"Plotted predictions for {vars_mlo[i]} in {prediction_figures_dirpath}/predictions_{vars_mlo[i]}.png")
#         success_predictions_count += 1
#     except Exception as e:
#         with open("error_log.txt", "a") as f:
#             f.write(f"Error plotting {vars_mlo[i]}\n")
#             f.write(str(e) + "\n")

# print("done with predictions\nStarting with ground truth plots:\n")
# success_groundtruth_count = 0
# for i in range(0, len(vars_mlo)):
#     try:
#         # plot_contour_map(ds_grid['lat'], ds_grid['lon'], predictions[:,i], title=f'Predictions: {vars_mlo[i]}', save_path=f'out_model_figures/predictions_{vars_mlo[i]}.png')
#         plot_contour_map(ds_grid['lat'], ds_grid['lon'], dso[:,i], title=f'Ground Truth: {vars_mlo[i]}', save_path=f'{groundtruth_figures_dirpath}/groundtruth_{vars_mlo[i]}.png')
#         print(f"Plotted ground truth for {vars_mlo[i]} in {groundtruth_figures_dirpath}/groundtruth_{vars_mlo[i]}.png")
#         success_groundtruth_count += 1
#     except:
#         with open("error_log.txt", "a") as f:
#             f.write(f"Error plotting {vars_mlo[i]}\n")

# print("done with ground truth\nStarting with error plots:\n")
# success_error_count = 0
# for i in range(0, len(vars_mlo)):
#     try:
#         plot_contour_map(ds_grid['lat'], ds_grid['lon'], dso[:,i] - predictions[:,i], title=f'Error: {vars_mlo[i]}', save_path=f'{error_figures_dirpath}/error_{vars_mlo[i]}.png')
#         print(f"Plotted error for {vars_mlo[i]} in {error_figures_dirpath}/error_{vars_mlo[i]}.png")
#         success_error_count += 1
#     except:
#         with open("error_log.txt", "a") as f:
#             f.write(f"Error plotting {vars_mlo[i]}\n")
# print("N successful plots ground truth: ", success_groundtruth_count)
# print("N successful plots predictions: ", success_predictions_count)
# print("N successful plots errors: ", success_error_count)
