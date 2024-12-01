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
from preprocess_data import prepare_training_files, prepare_validation_files, create_dataset, calculate_dataset_size, from_input_normalized_to_original, load_normalization_data
from model import build_base_model, CustomModel
from config import vars_mli, vars_mlo, vars_mlo_0, norm_path, data_fraction

# setup seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)



DEBUG=False

def compute_mass_loss(x, y_true, y_pred):
    LHFLX = x[:, input_var_indices['pbuf_LHFLX']]  # Surface latent heat flux (W/m²)
    # rescale
    LHFLX = from_input_normalized_to_original(LHFLX, "pbuf_LHFLX", mli_mean, mli_max, mli_min)
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
    SOLIN = from_input_normalized_to_original(SOLIN, "pbuf_SOLIN", mli_mean, mli_max, mli_min)
    LWUP = from_input_normalized_to_original(LWUP, "cam_in_LWUP", mli_mean, mli_max, mli_min)
    SHFLX = from_input_normalized_to_original(SHFLX, "pbuf_SHFLX", mli_mean, mli_max, mli_min)
    LHFLX = from_input_normalized_to_original(LHFLX, "pbuf_LHFLX", mli_mean, mli_max, mli_min)

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
            v = from_input_normalized_to_original(v, var, mli_mean, mli_max, mli_min)
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
                    print(f"These are the trainable lambdas: {lambda_variables}")
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

def main(constant_lambdas_str, exclude_these_losses_str, relative_scale_lambda_mse, relative_scale_lambda_mass, relative_scale_lambda_radiation, relative_scale_lambda_nonneg,
        output_results_dirpath, n_epochs, lr, batch_size=32):
    
    data_subset_fraction = data_fraction
    start_time = time.time()
    N_EPOCHS = n_epochs
    shuffle_buffer = 12 * 384  # ncol=384

    output_results_dirpath = f"{output_results_dirpath}/results_{data_subset_fraction}"

    # Set up GPU memory growth
    setup_gpu()

    # Paths


    global input_var_indices, output_var_indices, initial_loss_functions, lambdas_string_with_names, mlo_scale, mli_mean, mli_max, mli_min, constant_lambdas, exclude_these_losses

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

    training_files = prepare_training_files(data_subset_fraction = data_subset_fraction)
    validation_files = prepare_validation_files(data_subset_fraction = data_subset_fraction)

    # Load normalization parameters
    mli_mean, mli_max, mli_min, mlo_scale = load_normalization_data(norm_path)
    input_var_indices = {name: idx for idx, name in enumerate(vars_mli)}
    output_var_indices = {name: idx for idx, name in enumerate(vars_mlo)}

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

    base_model_0 = build_base_model()
    model_0 = CustomModel(base_model_0, initial_lambdas_0, constant_lambdas=constant_lambdas, exclude_these_losses=exclude_these_losses)
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
    base_model = build_base_model()
    print(f"initial_lambdas: {initial_lambdas}")
    model = CustomModel(base_model, initial_lambdas, constant_lambdas=constant_lambdas, exclude_these_losses=exclude_these_losses)
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
        args.n_epochs,
        args.lr,
        args.batch_size
    )
# python train.py --constant_lambdas_str "mass_radiation_nonneg" --exclude_these_losses_str="mass_radiation_nonneg" --relative_scale_lambda_mse 1.0 --relative_scale_lambda_mass 1.0 --relative_scale_lambda_radiation 1.0 --relative_scale_lambda_nonneg 1.0 --n_epochs 1 --lr 1e-4 --batch_size 32 --output_results_dirpath=/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov30/zeromodel
