#!/usr/bin/env python
# coding: utf-8
import time
import argparse
import glob
import random
import pickle
import os
import math
import tqdm
from tqdm import tqdm
import re
import random
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
import numpy as np

DEBUG=False

root_huggingface_data_dirpath = "/nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_all/"
root_climsim_dirpath = "/home/alvarovh/code/cse598_climate_proj/ClimSim/"


def compute_mass_loss(x, y_true, y_pred):
    LHFLX = x[:, input_var_indices['pbuf_LHFLX']]  # Surface latent heat flux (W/m²)
    # rescale
    LHFLX = from_input_normalized_to_original(LHFLX, "pbuf_LHFLX")
    L_v = 2.5e6  # Latent heat of vaporization (J/kg)
    E = LHFLX / L_v  # Evaporation rate (kg/m²/s)

    PRECC = y_pred[:, output_var_indices['cam_out_PRECC']]  # Rain rate (m/s)
    PRECSC = y_pred[:, output_var_indices['cam_out_PRECSC']]  # Snow rate (m/s)
    # Rescale
    PRECC = PRECC / mlo_scale['cam_out_PRECC']
    PRECSC = PRECSC / mlo_scale['cam_out_PRECSC']
    P = PRECC + PRECSC  # Total precipitation rate (m/s)

    ptend_q = y_pred[:, output_var_indices['ptend_q0001']]  # Specific humidity tendency (kg/kg/s)
    # It is already in the original units, so we dont need to rescale it
    delta_t = 1200  # Time step in seconds
    delta_q = ptend_q * delta_t  # Change in specific humidity (kg/kg)

    # Assume air density ρ = 1.225 kg/m³ and scale delta_q accordingly if needed
    loss_mass = tf.reduce_mean(tf.abs(delta_q - (E - P)))
    return loss_mass
def compute_radiation_loss(x, y_true, y_pred):
    NETSW = y_pred[:, output_var_indices['cam_out_NETSW']]
    FLWDS = y_pred[:, output_var_indices['cam_out_FLWDS']]
    # rescale them to the original units
    NETSW = NETSW / mlo_scale['cam_out_NETSW']
    FLWDS = FLWDS / mlo_scale['cam_out_FLWDS']
    SOLIN = x[:, input_var_indices['pbuf_SOLIN']]
    LWUP = x[:, input_var_indices['cam_in_LWUP']]
    SHFLX = x[:, input_var_indices['pbuf_SHFLX']]
    LHFLX = x[:, input_var_indices['pbuf_LHFLX']]
    SOLIN = from_input_normalized_to_original(SOLIN, "pbuf_SOLIN")
    LWUP = from_input_normalized_to_original(LWUP, "cam_in_LWUP")
    SHFLX = from_input_normalized_to_original(SHFLX, "pbuf_SHFLX")
    LHFLX = from_input_normalized_to_original(LHFLX, "pbuf_LHFLX")


    NETR = (SOLIN - NETSW) + (FLWDS - LWUP)
    SURFACE_FLUXES = SHFLX + LHFLX

    loss_radiation = tf.reduce_mean(tf.abs(NETR - SURFACE_FLUXES))
    return loss_radiation

def mse_loss(x, y_true, y_pred):
    loss_fn = tf.keras.losses.MeanSquaredError()
    return loss_fn(y_true, y_pred)

def combined_loss(x, y_true, y_pred, loss_funcs):
    total_loss = 0.0
    loss_components = {}
    for loss_name, (loss_func, weight) in loss_funcs.items():
        if weight == 0:
            # Define loss_value as zero when weight is zero
            loss_value = tf.constant(0.0)
        else:
            loss_value = loss_func(x, y_true, y_pred)
            total_loss += weight * loss_value
        loss_components[loss_name] = loss_value
        # assert loss_value.shape == (), f"{loss_name} LOSS shape: {loss_value.shape}"
        # Check for NaNs and Infs
        # assert not tf.math.is_nan(loss_value), f"{loss_name} LOSS is nan"
        # assert not tf.math.is_inf(loss_value), f"{loss_name} LOSS is inf"
        if DEBUG:
            print(f"{loss_name} LOSS: {loss_value}")
    # Ensure total_loss is a scalar tensor
    total_loss = tf.convert_to_tensor(total_loss)
    assert total_loss.shape == (), f"total_loss shape: {total_loss.shape}"
    return total_loss, loss_components

def compute_initial_loss_values(model, dataset, loss_functions, num_batches=5):
    initial_loss_values = {loss_name: 0.0 for loss_name in loss_functions}
    count = 0
    for x_batch, y_batch in dataset:
        if count >= num_batches:
            break
        y_pred = model(x_batch, training=False)
        for loss_name, (loss_func, _) in loss_functions.items():
            loss_value = loss_func(x_batch, y_batch, y_pred).numpy()
            initial_loss_values[loss_name] += loss_value
        count += 1
    # Compute mean initial loss values
    for loss_name in initial_loss_values:
        initial_loss_values[loss_name] /= count
    return initial_loss_values


def compute_nonneg_loss(x, y_true, y_pred):
    # vars_to_check = ['state_q0001', 'cam_out_PRECC', 'cam_out_PRECSC', 'cam_out_NETSW', 'cam_out_FLWDS']
    vars_in_always_positive = ['cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_LWUP', 'cam_in_OCNFRAC', 'cam_in_SNOWHICE', 'cam_in_SNOWHLAND', 'pbuf_COSZRS', 'pbuf_SOLIN', 'state_pmid', 'state_ps', 'state_q0001', 'state_q0002', 'state_q0003', 'state_t', 'pbuf_CH4', 'pbuf_N2O', 'pbuf_ozone']
    vars_out_always_positive = ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 
                     'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD',"state_t"]
    # lets rescale the output variables

    loss_nonneg = 0.0
    for var in vars_in_always_positive + vars_out_always_positive:
        if var in vars_out_always_positive:
            v = y_pred[:, output_var_indices[var]]
            # rescale them if is not 'state_q0001' or 'state_t'
            if var != 'state_q0001' and var != 'state_t':
                v = v / mlo_scale[var]
        elif var in input_var_indices:
            v = x[:, input_var_indices[var]]
            # rescale them
            v = from_input_normalized_to_original(v, var)
        else:
            continue
        loss_nonneg += tf.reduce_mean(tf.nn.relu(-v))
    return loss_nonneg

def train_model(model, dataset, val_dataset, model_optimizer, initial_lambdas, lambda_optimizer, epochs, steps_per_epoch, validation_steps, output_results_dirpath, data_subset_fraction=1.0, patience=5, min_delta=0.001):
    # Ensure datasets are repeated
    dataset = dataset.repeat()
    val_dataset = val_dataset.repeat()
    
    best_val_mse = np.inf  # Initialize best validation MSE
    no_improvement_epochs = 0  # Counter for early stopping

    # Get the initial loss values for each loss function
    if len(constant_lambdas) < 3:
        lambda_mass_param = tf.Variable(initial_value=initial_loss_functions['mass'][1], trainable=True, dtype=tf.float32)
        lambda_radiation_param = tf.Variable(initial_value=initial_loss_functions['radiation'][1], trainable=True, dtype=tf.float32)
        lambda_nonneg_param = tf.Variable(initial_value=initial_loss_functions['nonneg'][1], trainable=True, dtype=tf.float32)
        
        lambda_mass = tf.constant(0.0)  if 'mass' in exclude_these_losses else tf.nn.softplus(lambda_mass_param)
        lambda_radiation = tf.constant(0.0)  if 'radiation' in exclude_these_losses else tf.nn.softplus(lambda_radiation_param)
        lambda_nonneg = tf.constant(0.0)  if 'nonneg' in exclude_these_losses else tf.nn.softplus(lambda_nonneg_param)
    
    # Batch-level logging file paths
    train_batch_log_path = f"{output_results_dirpath}/batch_train_log_lambdas_{lambdas_string_with_names}_datafrac_{data_subset_fraction}.csv"
    val_batch_log_path = f"{output_results_dirpath}/batch_val_log_lambdas_{lambdas_string_with_names}_datafrac_{data_subset_fraction}.csv"

    # Create and initialize batch logging CSVs
    with open(train_batch_log_path, "w") as f:
        f.write("epoch,batch,loss,mae,mse,mass_loss,radiation_loss,nonneg_loss,lambda_mass,lambda_radiation,lambda_nonneg\n")
    with open(val_batch_log_path, "w") as f:
        f.write("epoch,batch,loss,mae,mse,mass_loss,radiation_loss,nonneg_loss,lambda_mass,lambda_radiation,lambda_nonneg\n")
    
    train_variable_metrics_path = f"{output_results_dirpath}/train_variable_metrics_lambdas_{lambdas_string_with_names}_datafrac_{data_subset_fraction}.csv"
    with open(train_variable_metrics_path, "w") as f:
        f.write("epoch,batch,variable_name,mae,mse\n")
    val_variable_metrics_path = f"{output_results_dirpath}/val_variable_metrics_lambdas_{lambdas_string_with_names}_datafrac_{data_subset_fraction}.csv"
    with open(val_variable_metrics_path, "w") as f:
        f.write("epoch,batch,variable_name,mae,mse\n")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("=" * 20)

        # Initialize accumulators for epoch-level metrics
        epoch_loss = 0.0
        step_count = 0

        # Initialize metrics for training
        train_mae = tf.keras.metrics.MeanAbsoluteError()
        train_mse = tf.keras.metrics.MeanSquaredError()
        train_mass_loss = 0.0
        train_radiation_loss = 0.0
        train_nonneg_loss = 0.0


        # Training loop with tqdm
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1} Training", unit="step") as pbar:
            for step, (x_batch_train, y_batch_train) in enumerate(dataset):
                if step >= steps_per_epoch:
                    break

                # Variables of the base model (weights)
                model_variables = model.base_model.trainable_variables

                # Lambda parameters
                lambda_variables = [
                    model.lambda_mass_param,
                    model.lambda_radiation_param,
                    model.lambda_nonneg_param
                ]
                lambda_variables = [var for var in lambda_variables if var.trainable]


                with tf.GradientTape(persistent=True) as tape:
                    # Forward pass and loss computation
                    # Access lambda parameters from the model
                    lambda_mass_param = model.lambda_mass_param
                    lambda_radiation_param = model.lambda_radiation_param
                    lambda_nonneg_param = model.lambda_nonneg_param

                    # Compute lambdas
                    lambda_mass = tf.constant(0.0)  if 'mass' in exclude_these_losses else tf.nn.softplus(lambda_mass_param)
                    lambda_radiation = tf.constant(0.0)  if 'radiation' in exclude_these_losses else tf.nn.softplus(lambda_radiation_param)
                    lambda_nonneg = tf.constant(0.0)  if 'nonneg' in exclude_these_losses else tf.nn.softplus(lambda_nonneg_param)
                    
                    # Update loss_functions inside the tape
                    loss_functions = {
                        'mse': (mse_loss, 1.0),  # Keep MSE fixed if desired
                        'mass': (compute_mass_loss, lambda_mass),
                        'radiation': (compute_radiation_loss, lambda_radiation),
                        'nonneg': (compute_nonneg_loss, lambda_nonneg),
                    }

                    y_pred = model(x_batch_train, training=True)
                    total_loss, loss_components = combined_loss(x_batch_train, y_batch_train, y_pred, loss_functions)

                if DEBUG:
                    print("Total loss: ", total_loss)

                # Compute gradients with respect to all trainable variables
                grads = tape.gradient(total_loss, model.trainable_variables)
                
                # Compute gradients for model variables and lambda variables separately
                model_grads = tape.gradient(total_loss, model_variables)
                # Compute gradients for lambda parameters
                if lambda_variables:  # Check if there are trainable lambdas
                    lambda_grads = tape.gradient(total_loss, lambda_variables)
                    lambda_optimizer.apply_gradients(zip(lambda_grads, lambda_variables))
                    # After computing gradients
                    for var, grad in zip(lambda_variables, lambda_grads):
                        if DEBUG:
                            print(f"Gradient norm for {var.name}: {tf.norm(grad).numpy()}")

                # Apply gradients to model weights
                model_optimizer.apply_gradients(zip(model_grads, model_variables))

                # Apply gradients to lambda parameters
                # skip if no lambdas are trainableL
                if lambda_variables:
                    lambda_optimizer.apply_gradients(zip(lambda_grads, lambda_variables))
                del tape

                for grad in grads:
                    if grad is not None:
                        if tf.reduce_any(tf.math.is_nan(grad)):
                            print("NaN detected in gradients")
                            # You can choose to handle NaNs here
                            with open("nan_gradients.txt", "a") as f:
                                f.write(f"ERROR in epoch {epoch + 1}, step {step + 1}, lambdas: {lambdas_string_with_names}, datafrac: {data_subset_fraction}\n")
                            exit()
                    else:
                        # Handle or log the None gradients if necessary
                        print("None gradient detected")
                        pass
                if DEBUG:
                    print(f"Lambda Mass: {lambda_mass.numpy()}, Lambda Radiation: {lambda_radiation.numpy()}, Lambda Non-negativity: {lambda_nonneg.numpy()}")

                # Access specific loss components if needed
                mass_loss_value = loss_components.get('mass', tf.constant(0.0)).numpy()
                radiation_loss_value = loss_components.get('radiation', tf.constant(0.0)).numpy()
                nonneg_loss_value = loss_components.get('nonneg', tf.constant(0.0)).numpy()

                # PER VARIABLE METRICS ------------------------------------------------------------------------------------------------------
                with open(train_batch_log_path, "a") as f:
                    # Construct the line for CSV
                    line = f"{epoch + 1},{step + 1},{total_loss.numpy()},{train_mae.result().numpy()},{train_mse.result().numpy()},{mass_loss_value},{radiation_loss_value},{nonneg_loss_value}," \
                        f"{lambda_mass.numpy()},{lambda_radiation.numpy()},{lambda_nonneg.numpy()}\n"
                    f.write(line)
                
                variable_mae = {}
                variable_mse = {}
                mae_metric = tf.keras.metrics.MeanAbsoluteError()
                mse_metric = tf.keras.metrics.MeanSquaredError()
                # Loop through each variable and compute its MAE and MSE
                for i, var_name in enumerate(vars_mlo):  # Assuming the output corresponds to vars_mli
                    mae_metric = tf.keras.metrics.MeanAbsoluteError()
                    
                    # Calculate MAE and MSE for the specific variable
                    mae_metric.update_state(y_batch_train[:, i], y_pred[:, i])
                    mse_metric.update_state(y_batch_train[:, i], y_pred[:, i])

                    # Store the computed values
                    variable_mae[var_name] = mae_metric.result().numpy()
                    variable_mse[var_name] = mse_metric.result().numpy()

                with open(train_variable_metrics_path, "a") as f:
                    for var_name in vars_mlo:
                        f.write(f"{epoch + 1},{step + 1},{var_name},{variable_mae[var_name]:.6f},{variable_mse[var_name]:.6f}\n")
                
                # Accumulate epoch-level metrics
                # Update epoch-level metrics
                epoch_loss += total_loss.numpy()
                train_mass_loss += mass_loss_value
                train_radiation_loss += radiation_loss_value
                train_nonneg_loss += nonneg_loss_value
                train_mae.update_state(y_batch_train, y_pred)
                train_mse.update_state(y_batch_train, y_pred)
                step_count += 1
                
                # Update tqdm progress bar
                pbar.set_postfix({
                    "Loss": f"{total_loss.numpy():.4f}",
                    "MAE": f"{train_mae.result().numpy():.4f}",
                    "MSE": f"{train_mse.result().numpy():.4f}",
                # lets monitor also one specific variable mse:
                    "state_t_MSE": f"{variable_mse['state_t']:.4f}",
                    "cam_out_NETSW_MSE": f"{variable_mse['cam_out_NETSW']:.4f}",
                })
                pbar.update(1)


        # Compute averages for the epoch
        avg_epoch_loss = epoch_loss / step_count
        avg_epoch_mae = train_mae.result().numpy()
        avg_epoch_mse = train_mse.result().numpy()
        avg_epoch_mass_loss = train_mass_loss / step_count
        avg_epoch_radiation_loss = train_radiation_loss / step_count
        avg_epoch_nonneg_loss = train_nonneg_loss / step_count

        print(f"Average Training Loss for Epoch {epoch + 1}: {avg_epoch_loss:.4f}, "
              f"MAE: {avg_epoch_mae:.4f}, MSE: {avg_epoch_mse:.4f} "
                f"Mass Loss: {avg_epoch_mass_loss:.4f}, "
                f"Radiation Loss: {avg_epoch_radiation_loss:.4f}, "
                f"Non-negativity Loss: {avg_epoch_nonneg_loss:.4f}")

        for loss_name, loss_value in loss_components.items():
            tf.summary.scalar(f'loss/{loss_name}', loss_value, step=epoch)

        # Reset metrics for the next epoch
        train_mae.reset_state()
        train_mse.reset_state()

        # Validation loop
        val_loss = 0.0
        val_mae = tf.keras.metrics.MeanAbsoluteError()
        val_mse = tf.keras.metrics.MeanSquaredError()
        val_mass_loss = 0.0
        val_radiation_loss = 0.0
        val_nonneg_loss = 0.0
        val_step_count = 0

        print("Validation")
        with tqdm(total=validation_steps, desc=f"Epoch {epoch+1} Validation", unit="step") as pbar:
            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                if step >= validation_steps:
                    break
                y_val_pred = model(x_batch_val, training=False)
                val_loss_value, loss_components = combined_loss(x_batch_val, y_batch_val, y_val_pred, loss_functions)

                # Update overall metrics for the batch
                val_mae.update_state(y_batch_val, y_val_pred)
                val_mse.update_state(y_batch_val, y_val_pred)

                # Initialize dictionaries to store per-variable MAE and MSE for validation
                variable_val_mae = {}
                variable_val_mse = {}

                # Loop through each variable and compute its MAE and MSE
                for i, var_name in enumerate(vars_mlo):  
                    val_mae_metric = tf.keras.metrics.MeanAbsoluteError()
                    val_mse_metric = tf.keras.metrics.MeanSquaredError()

                    # Calculate MAE and MSE for the specific variable
                    val_mae_metric.update_state(y_batch_val[:, i], y_val_pred[:, i])
                    val_mse_metric.update_state(y_batch_val[:, i], y_val_pred[:, i])

                    # Store the computed values
                    variable_val_mae[var_name] = val_mae_metric.result().numpy()
                    variable_val_mse[var_name] = val_mse_metric.result().numpy()

                # Log batch metrics for validation
                with open(val_batch_log_path, "a") as f:
                    line = f"{epoch + 1},{step + 1},{val_loss_value},{val_mae.result().numpy()},{val_mse.result().numpy()}, {val_mass_loss},{val_radiation_loss},{val_nonneg_loss},"
                    for loss_name, loss_value in loss_components.items():
                        line += f"{loss_value.numpy()},"
                    line = line[:-1] + "\n"
                    f.write(line)
                
                # Append per-variable metrics for validation
                with open(val_variable_metrics_path, "a") as f:
                    for var_name in vars_mlo:
                        f.write(f"{epoch + 1},{step + 1},{var_name},{variable_val_mae[var_name]:.6f},{variable_val_mse[var_name]:.6f}\n")

                # Accumulate validation metrics
                val_loss += val_loss_value
                val_mass_loss += loss_components.get('mass', tf.constant(0.0)).numpy()
                val_radiation_loss += loss_components.get('radiation', tf.constant(0.0)).numpy()
                val_nonneg_loss += loss_components.get('nonneg', tf.constant(0.0)).numpy()
                val_step_count += 1
                pbar.update(1)

        # Calculate average validation metrics
        avg_val_loss = val_loss / val_step_count
        avg_val_mae = val_mae.result().numpy()
        avg_val_mse = val_mse.result().numpy()
        avg_val_mass_loss = val_mass_loss / val_step_count
        avg_val_radiation_loss = val_radiation_loss / val_step_count
        avg_val_nonneg_loss = val_nonneg_loss / val_step_count


        print(f"Validation Loss for Epoch {epoch + 1}: {avg_val_loss:.4f}, "
              f"MAE: {avg_val_mae:.4f}, MSE: {avg_val_mse:.4f}, "
                f"Mass Loss: {avg_val_mass_loss:.4f}, "
                f"Radiation Loss: {avg_val_radiation_loss:.4f}, "
                f"Non-negativity Loss: {avg_val_nonneg_loss:.4f}")
        
        # Early stopping logic based on MSE 
        if avg_val_mse < best_val_mse - min_delta:
            best_val_mse = avg_val_loss
            no_improvement_epochs = 0  # Reset counter
            # Save the best model
            model_save_path = os.path.join(output_results_dirpath, f"best_model_lambdas_{lambdas_string_with_names}_datafrac_{data_subset_fraction}_epoch_{epoch + 1}.keras")
            model.save(model_save_path)
            print(f"Model saved at {model_save_path} with improved validation MSE: {best_val_mse:.4f}")
        else:
            no_improvement_epochs += 1
            print(f"No improvement in validation MSE for {no_improvement_epochs} epochs.")

        # Stop training if no improvement exceeds patience
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered after {no_improvement_epochs} epochs with no improvement.")
            break


def calculate_dataset_size(file_list, vars_mli):
    # Load one representative file to determine sample size
    try:
        example_file = file_list[0]
        # ds = xr.open_dataset(example_file, engine='netcdf4')
        with xr.open_dataset(example_file, engine='netcdf4') as ds:
            samples_per_file = ds[vars_mli[0]].shape[0]  # Assuming all vars_mli have the same shape
        print(f"Samples per file: {samples_per_file}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return 0

    # Total dataset size
    total_samples = samples_per_file * len(file_list)
    print(f"Total number of samples in dataset: {total_samples}")
    return total_samples


def setup_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Enable memory growth for the GPU
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
    else:
        print("No GPUs found. Using CPU.")

def load_file_list(root_train_path):
    if os.path.exists("file_list.pkl"):
        with open("file_list.pkl", "rb") as f:
            all_files = pickle.load(f)
            print("Loaded file list from file_list.pkl.")
    else:
        all_files = glob.glob(os.path.join(root_train_path, '*/*MMF.mli*.nc'))
        with open("file_list.pkl", "wb") as f:
            pickle.dump(all_files, f)
            print("File list generated and saved to file_list.pkl.")
    return all_files

def from_input_normalized_to_original(normalized, var, input_var_norm_epsilon = 1e-5):
    '''
    Function to convert normalized input variables back to original scale
    
    Args:
    normalized: tf.Tensor, normalized input variable
    var: str, variable name
    input_var_norm_epsilon: float, epsilon value for normalization
    
    Returns:
    original: tf.Tensor, original scale input variable
        
    '''
    # mli_mean, mli_max, mli_min, mlo_scale
    # nrormalized by..
    # ds = (ds - mli_mean) / (mli_max - mli_min + epsilon)
    # rescaled by...
    # print(f"Variable: {var}")
    # print(f"normalized shape: {normalized.shape}")
    # print(f"scaling factor shape: {(mli_max[var] - mli_min[var] + input_var_norm_epsilon).shape}")
    # print(f"mean shape: {mli_mean[var].shape}")

    # it seems like sometimes it fails to get the key frmo the dictionaries, so I will print the key if it fails

    try:
        original = normalized * (mli_max[var] - mli_min[var] + input_var_norm_epsilon) + mli_mean[var]
    except:
        print(f"ERROR Failed to get key {var}")
        with open("failed_keys.txt", "a") as f:
            f.write(f"Failed to get key {var}\n")
        

    return normalized * (mli_max[var] - mli_min[var] + input_var_norm_epsilon) + mli_mean[var]

def create_dataset(
    file_list, vars_mli, vars_mlo, mli_mean,
    mli_max, mli_min, mlo_scale, shuffle_buffer,
    batch_size,
    shuffle=True
):
    ds = load_nc_dir_with_generator(
        file_list, vars_mli, vars_mlo, mli_mean,
        mli_max, mli_min, mlo_scale
    )
    ds = ds.unbatch()
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=int(shuffle_buffer / 384))
    return ds

def load_nc_dir_with_generator(
    filelist, vars_mli, vars_mlo,
    mli_mean, mli_max, mli_min, mlo_scale
):
    def gen():
        for file in filelist:
            # input read / preprocess #
            # read mli (-> ds)
            # ds = xr.open_dataset(file, engine='netcdf4')
            with xr.open_dataset(file, engine='netcdf4') as ds:
                with xr.open_dataset(file.replace('.mli.','.mlo.'), engine='netcdf4') as dso:
                    # subset ozone, ch4, n2o

                    # ds_utls = ds[vars_mli_utls]\
                    #             .isel(lev=slice(5,21)).rename({'lev':'lev2'})
                    # combine ds and ds_utls
                    ds = ds[vars_mli]

                    # ds = ds.merge(ds_utls)
                    
                    # output read / preprocess #
                    # read mlo (-> dso)
                    # make mlo tendency variales ("ptend_xxxx"):
                    # state_t_ds = ds['state_t']
                    # state_q0001_ds = ds['state_q0001']
                    state_t_dso = dso['state_t']
                    state_q0001_dso = dso['state_q0001']
                    
                    for kvar in ['state_t','state_q0001','state_q0002', 'state_q0003', 'state_u', 'state_v']:
                        dso[kvar.replace('state','ptend')] = (dso[kvar] - ds[kvar])/1200 # timestep=1200[sec]
                    
                    # normalizatoin, scaling #
                    # ds = (ds-mli_mean)/(mli_max-mli_min)
                    input_var_norm_epsilon = 1e-5
                    ds = (ds - mli_mean) / (mli_max - mli_min + input_var_norm_epsilon)
                    
                    # print if this was indifined:
                    dso = dso * mlo_scale

                    # get index 59 for variables that have more than 1 level
                    index=59
                    for var in vars_mlo_0:
                        # print(var)
                        if len(dso[var].shape) == 2:
                            dso[var] = dso[var][index]
                            # print("changed")
                    dso=dso[vars_mlo_0]
                    # now lets add the additional variables: state_t
                    dso["state_t"] = state_t_dso[index]
                    dso["state_q0001"] = state_q0001_dso[index] # I DONT SCALE THESE VARIABLES
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

                    # Check dataset for NaNs
                    # for x_batch, y_batch in zip(ds.values, dso.values):
                    #     if tf.reduce_any(tf.math.is_nan(x_batch)) or tf.reduce_any(tf.math.is_nan(y_batch)):
                    #         print("NaNs detected in dataset!")
                    denominator = (mli_max - mli_min + input_var_norm_epsilon)
                    for var in vars_mli:
                        if np.any(denominator[var] == 0):
                            print(f"Zero range detected in variable {var}")


                    yield (ds.values, dso.values)

    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.float32),
        # output_shapes=((None, 125), (None, 128))
        # output_shapes=((None,7),(None,12)) dynamically compute this
        output_shapes=((None, len(vars_mli)), (None, len(vars_mlo)))
    )

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

@keras.utils.register_keras_serializable()
class CustomModel(tf.keras.Model):
    def __init__(self, base_model, initial_lambdas):
        super(CustomModel, self).__init__()
        self.base_model = base_model

        # corrected_initial_lambdas = {}
        # for var in initial_lambdas:
        #     if initial_lambdas[var] ==0:
        #         corrected_initial_lambdas[var] = (-1e-10)
        #     else:
        #         corrected_initial_lambdas[var] = initial_lambdas[var]

        # Initialize trainable lambda parameters
        self.lambda_mass_param = self.add_weight(
            name='lambda_mass_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_lambdas['mass']),
            trainable=('mass' not in constant_lambdas and 'mass' not in exclude_these_losses),
        )
        self.lambda_radiation_param = self.add_weight(
            name='lambda_radiation_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_lambdas['radiation']),
            trainable=('radiation' not in constant_lambdas and 'radiation' not in exclude_these_losses),
        )
        self.lambda_nonneg_param = self.add_weight(
            name='lambda_nonneg_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_lambdas['nonneg']),
            trainable=('nonneg' not in constant_lambdas and 'nonneg' not in exclude_these_losses),
        )

    def call(self, inputs):
        # Pass inputs through the base model
        return self.base_model(inputs)

    def get_config(self):
        # Include all arguments required to reconstruct the model
        config = super(CustomModel, self).get_config()
        config.update({
            'base_model': keras.layers.serialize(self.base_model),
            'initial_lambdas': {
                'mass': self.lambda_mass_param.numpy(),
                'radiation': self.lambda_radiation_param.numpy(),
                'nonneg': self.lambda_nonneg_param.numpy(),
            }
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Reconstruct the base_model and initial_lambdas from the config
        base_model = keras.layers.deserialize(config['base_model'])
        initial_lambdas = config['initial_lambdas']
        return cls(base_model=base_model, initial_lambdas=initial_lambdas)


def prepare_training_files(root_path):
    """
    Prepare training files: every 10th sample for the first 5 days of each month 
    for the first 6 years, plus January of year 0007.
    """
    file_list = load_file_list(root_path)

    # Define regex patterns
    # Match files for years 0001 to 0006, any month, and any day

    # f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.000[123456]-*-*-*.nc')
    # f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0007-01-*-*.nc')
    pattern_f_mli1 = re.compile(r'.*/E3SM-MMF\.mli\.000[1-6]-*-*-.*\.nc$')
    # Match files for year 0007, month 01, and all days
    pattern_f_mli2 = re.compile(r'.*/E3SM-MMF\.mli\.0007-01-*-*.nc$')

    # Filter files matching each pattern
    f_mli1 = [file for file in file_list if pattern_f_mli1.match(file)]
    f_mli2 = [file for file in file_list if pattern_f_mli2.match(file)]

    # Combine and sort the filtered files
    training_files = sorted(f_mli1 + f_mli2)

    # Global shuffle and select every 10th file
    random.shuffle(training_files)
    training_files = training_files[::10]

    return training_files

def prepare_validation_files(root_path):
    """
    Prepare validation files: every 10th sample for the first 5 days of each month 
    for the following 2 years.
    """
    file_list = load_file_list(root_path)
    
    # f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0007-0[23456789]-0[12345]-*.nc')
    # f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0007-1[012]-0[12345]-*.nc')
    # f_mli3 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.000[89]-*-0[12345]-*.nc')
    
    # Compile regex patterns for matching desired files
    pattern1 = re.compile(r'E3SM-MMF\.mli\.0007-0[23456789]-0[12345]-.*\.nc')
    pattern2 = re.compile(r'E3SM-MMF\.mli\.0007-1[012]-0[12345]-.*\.nc')
    pattern3 = re.compile(r'E3SM-MMF\.mli\.000[89]-.*-0[12345]-.*\.nc')
    
    # Filter files matching the patterns
    f_mli1 = [f for f in file_list if pattern1.search(f)]
    f_mli2 = [f for f in file_list if pattern2.search(f)]
    f_mli3 = [f for f in file_list if pattern3.search(f)]
    
    # Combine and sort validation files
    validation_files = sorted([*f_mli1, *f_mli2, *f_mli3])
    
    # Shuffle and select every 10th file
    random.shuffle(validation_files)  # Global shuffle
    validation_files = validation_files[::10]  # Select every 10th file
    
    return validation_files

def load_training_and_validation_datasets(root_path, shuffle_buffer, batch_size):
    """
    Create training and validation datasets using the prepared files.
    """
    training_files = prepare_training_files(root_path)
    validation_files = prepare_validation_files(root_path)
    
    print(f'[TRAIN] Total # of input files: {len(training_files)}')
    print(f'[VAL] Total # of input files: {len(validation_files)}')

    # Load datasets using the generator function
    tds = load_nc_dir_with_generator(training_files)
    tds = tds.unbatch()
    tds = tds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    tds = tds.batch(batch_size)
    tds = tds.prefetch(buffer_size=int(shuffle_buffer / 384))

    tds_val = load_nc_dir_with_generator(validation_files)
    tds_val = tds.unbatch()
    tds_val = tds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    tds_val = tds.batch(batch_size)
    tds_val = tds.prefetch(buffer_size=int(shuffle_buffer / 384))

    return tds, tds_val

def main(constant_lambdas_str, exclude_these_losses_str, relative_scale_lambda_mse, relative_scale_lambda_mass, relative_scale_lambda_radiation, relative_scale_lambda_nonneg,
        output_results_dirpath, data_subset_fraction, n_epochs, lr, batch_size=32):
    start_time = time.time()
    N_EPOCHS = n_epochs
    shuffle_buffer = 12 * 384  # ncol=384

    output_results_dirpath = f"{output_results_dirpath}/results_{data_subset_fraction}"

    # Set up GPU memory growth
    setup_gpu()

    # Paths
    norm_path = f"{root_climsim_dirpath}/preprocessing/normalizations/"
    root_train_path = (
        f"{root_huggingface_data_dirpath}/datasets--LEAP--ClimSim_low-res/snapshots/"
        "bab82a2ebdc750a0134ddcd0d5813867b92eed2a/train/"
    )

    global vars_mlo, vars_mlo_0, vars_mli, vars_mlo_dims, vars_mli_dims, input_var_indices, output_var_indices, initial_loss_functions, lambdas_string_with_names, mlo_scale, mli_mean, mli_max, mli_min, constant_lambdas, exclude_these_losses
    if len(exclude_these_losses_str) > 1:
        exclude_these_losses = exclude_these_losses_str.split("_")
    else:
        exclude_these_losses = []
    if len(constant_lambdas_str) > 1:
        constant_lambdas = constant_lambdas_str.split("_")
    else:
        constant_lambdas = []
    assert all([l in ['mass', 'radiation', 'nonneg'] for l in constant_lambdas]), "Invalid constant lambda name(s) provided."
    assert all([l in ['mass', 'radiation', 'nonneg'] for l in exclude_these_losses]), "Invalid loss name(s) provided."

    # Load normalization datasets
    mli_mean = xr.open_dataset(norm_path + 'inputs/input_mean.nc')
    mli_max = xr.open_dataset(norm_path + 'inputs/input_max.nc')
    mli_min = xr.open_dataset(norm_path + 'inputs/input_min.nc')
    mlo_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc')

    # get only the 59 index of the variables that have more than 1 level
    mli_mean = mli_mean.isel(lev=59)
    mli_max = mli_max.isel(lev=59)
    mli_min = mli_min.isel(lev=59)
    mlo_scale = mlo_scale.isel(lev=59)

    training_files = prepare_training_files(root_train_path)
    validation_files = prepare_validation_files(root_train_path)

    # import one file to get the variables
    mli = xr.open_dataset(training_files[0], engine='netcdf4')
    mlo = xr.open_dataset(training_files[0].replace('.mli.', '.mlo.'), engine='netcdf4')
    vars_mlo_0   = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v',
                     'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 
                     'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
    # vars_mlo_0 is the same but without state_t and state_q0001. vars_mlo_0 corresponds with the scaling factor mlo_scale
    vars_mlo      = vars_mlo_0 + ['state_t', 'state_q0001']

    print(f"Output Vars Included: vars_mlo: {vars_mlo}")
    vars_mlo_dims = [(mlo_scale[var].values.size) for var in vars_mlo_0]

    assert len(vars_mlo_0) == len(vars_mlo_dims), f"vars_mlo and vars mlo_dims dont share the same length: {len(vars_mlo_0)} != {len(vars_mlo_dims)}"
    vars_mli = list(mli.data_vars.keys())[2:] # ['cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_LWUP', 'cam_in_OCNFRAC', 'cam_in_SNOWHICE', 'cam_in_SNOWHLAND', 'pbuf_COSZRS', 'pbuf_LHFLX', 'pbuf_SHFLX', 'pbuf_SOLIN', 'pbuf_TAUX', 'pbuf_TAUY', 'state_pmid', 'state_ps', 'state_q0001', 'state_q0002', 'state_q0003', 'state_t', 'state_u', 'state_v', 'pbuf_CH4', 'pbuf_N2O', 'pbuf_ozone']
    print(f"Input Vars Included: vars_mli: {vars_mli}")
    vars_mli_dims = [(i.values.size) for i in mli_min.data_vars.values()]

    input_var_indices = {name: idx for idx, name in enumerate(vars_mli)}
    output_var_indices = {name: idx for idx, name in enumerate(vars_mlo)}
    
    print(f'[TRAIN] Total # of input files: {len(training_files)}')
    print(f'[VAL] Total # of input files: {len(validation_files)}')

    #subset to specified fraction
    training_files = training_files[:int(len(training_files) * data_subset_fraction)]
    validation_files = validation_files[:int(len(validation_files) * data_subset_fraction)]

    print(f'[TRAIN] Total # of input files AFTER SUBSET OF {data_subset_fraction}: {len(training_files)}')
    print(f'[VAL] Total # of input files AFTER SUBSET OF {data_subset_fraction}: {len(validation_files)}')

    # Create training and validation datasets
    tds = create_dataset(
        training_files, vars_mli, vars_mlo, mli_mean,
        mli_max, mli_min, mlo_scale, shuffle_buffer, batch_size
    )
    tds_val = create_dataset(
        validation_files, vars_mli, vars_mlo, mli_mean,
        mli_max, mli_min, mlo_scale, shuffle_buffer, batch_size
    )

    # Calculate dataset size
    total_training_samples = calculate_dataset_size(training_files, vars_mli)
    steps_per_epoch = math.ceil(total_training_samples / batch_size)
    print(f"Steps per epoch: {steps_per_epoch}")

    total_validation_samples = calculate_dataset_size(validation_files, vars_mli)
    validation_steps = math.ceil(total_validation_samples / batch_size)
    print(f"Validation steps: {validation_steps}")

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # Prepare lambdas -----------------------------------------------------------
    # Compute initial loss values
    initial_loss_functions = {
        'mse': (mse_loss, 1.0),
        'mass': (compute_mass_loss, 0),
        'radiation': (compute_radiation_loss, 0),
        'nonneg': (compute_nonneg_loss, 0),
    }
    initial_lambdas_0 = {
        'mse': 1,
        'mass': 1,
        'radiation': 1,
        'nonneg': 1,
    }
    relative_scale_lambda = {
        'mse': relative_scale_lambda_mse,
        'mass': relative_scale_lambda_mass,
        'radiation': relative_scale_lambda_radiation,
        'nonneg': relative_scale_lambda_nonneg,
    }

    print(f"###RELATIVE SCALE LAMBDA: {relative_scale_lambda}")
    print(f"###CONSTANT LAMBDAS: {constant_lambdas}")
    print(f"###EXCLUDE THESE LOSSES: {exclude_these_losses}")

    base_model_0 = build_model()
    model_0 = CustomModel(base_model_0, initial_lambdas_0) # naive model with no lambdas
    initial_loss_values = compute_initial_loss_values(model_0, tds, initial_loss_functions, num_batches=100)
    print(f"INITIAL LOSS VALUES: {initial_loss_values}")
    
    # Number of loss functions
    N = len(initial_loss_values) - len(exclude_these_losses)

    # Compute constant c
    epsilon = 1e-8  # To prevent division by zero
    c = 100000.0 / N  # Total desired combined loss divided by number of loss functions

    # Compute lambdas
    initial_lambdas = {}
    for loss_name, loss_value in initial_loss_values.items():
        initial_lambdas[loss_name] = relative_scale_lambda[loss_name] * (c / (loss_value + epsilon))  # Scale by relative scale factor
    print(f"INITIAL LAMBDAS SCALED: {initial_lambdas}")
    
    # Compute scaled losses to verify
    scaled_losses = {loss_name: initial_lambdas[loss_name] * loss_value for loss_name, loss_value in initial_loss_values.items()}
    total_scaled_loss = sum(scaled_losses.values())

    print(f"SCALED LOSSES: {scaled_losses}")
    print(f"TOTAL SCALED LOSS: {total_scaled_loss}")

    # print how much are the initial losses in comparison to the mse loss
    print(f"Initial loss values in comparison to MSE loss:")
    for loss_name, loss_value in initial_loss_values.items():
        print(f"{loss_name}: {loss_value / initial_loss_values['mse']:.4f}")

    # update initial loss functions with the scaled values
    initial_loss_functions = {
        'mse': (mse_loss, 1.0),
        'mass': (compute_mass_loss, initial_lambdas['mass']),
        'radiation': (compute_radiation_loss, initial_lambdas['radiation']),
        'nonneg': (compute_nonneg_loss, initial_lambdas['nonneg']),
    }
    
    # Define model --------------------------------------------------------------
    base_model = build_model()
    print(f"initial_lambdas: {initial_lambdas}")
    # exit()
    model = CustomModel(base_model, initial_lambdas)
    # ----
    # Set up optimizer ---------------------------------------------------------
    # Optimizer for model weights
    model_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)

    # Optimizer for lambda parameters with a higher learning rate
    lambda_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, clipvalue=1.0)

    # Ensure Log directory exists
    os.makedirs(output_results_dirpath, exist_ok=True)
    lambdas_string_with_names = "_".join([f"excluded_{loss}" for loss in exclude_these_losses] + [f"constant_{loss}" for loss in constant_lambdas] + [f"scaledfactorof{loss}_{scale:.2f}" for loss, scale in relative_scale_lambda.items()])

    # Train model --------------------------------------------------------------
    train_model(
    model, 
    tds, 
    tds_val, 
    model_optimizer,
    initial_lambdas,
    lambda_optimizer,
    N_EPOCHS, 
    steps_per_epoch, 
    validation_steps, 
    output_results_dirpath,
    data_subset_fraction
    )
    print("DONE")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument(
        '--constant_lambdas_str',
        type=str,
        default='mass radiation nonneg',
        help='Space-separated list of loss names to keep constant'
    )
    parser.add_argument(
        '--exclude_these_losses_str',
        type=str,
        default='',
        help='Space-separated list of loss names to exclude'
    )
    parser.add_argument(
        '--relative_scale_lambda_mse',
        type=float,
        default=1.0,
        help='Relative scale factor for MSE loss'
    )
    parser.add_argument(
        '--relative_scale_lambda_radiation',
        type=float,
        default=1.0,
        help='Relative scale factor for Radiation loss'
    )
    parser.add_argument(
        '--relative_scale_lambda_mass',
        type=float,
        default=1.0,
        help='Relative scale factor for Mass loss'
    )
    parser.add_argument(
        '--relative_scale_lambda_nonneg',
        type=float,
        default=1.0,
        help='Relative scale factor for Non-negativity loss'
    )    
    parser.add_argument(
        '--output_results_dirpath',
        type=str,
        default='',
        help='Output results directory path'
    )
    parser.add_argument(
        '--data_subset_fraction',
        type=float,
        default=0.01,
        help='Fraction of data to use (between 0 and 1)'
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=1,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    args = parser.parse_args()

    if not args.output_results_dirpath:
        args.output_results_dirpath = (
            f"/home/alvarovh/code/cse598_climate_proj/results/"
        )

    main(
        args.constant_lambdas_str,
        args.exclude_these_losses_str,
        args.relative_scale_lambda_mse,
        args.relative_scale_lambda_mass,
        args.relative_scale_lambda_radiation,
        args.relative_scale_lambda_nonneg,
        args.output_results_dirpath,
        args.data_subset_fraction,
        args.n_epochs,
        args.lr,
        args.batch_size
    )
# python combined_loss_model.py --constant_lambdas_str="mass nonneg" --exclude_these_losses_str="mass nonneg" --relative_scale_lambda_mse=0.25 --output_results_dirpath=/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov24/initiallambdasscaled --data_subset_fraction=0.01 --n_epochs=10
