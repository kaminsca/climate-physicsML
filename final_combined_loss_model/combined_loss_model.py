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

# vars_mlo = ["state_t","state_q0001",'ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']
# # vars_mli = [
# #     'state_t', 'state_q0001', 'state_ps', 'pbuf_SOLIN',
# #     'pbuf_LHFLX', 'pbuf_SHFLX', 'cam_in_LWUP'
# # ]


# Get the CUDA_VISIBLE_DEVICES environment variable to determine available GPUs
visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')

# Configure logical GPUs based on the visible ones
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Create logical GPUs only for the visible devices
        for gpu_id, gpu in enumerate(gpus):
            if str(gpu_id) in visible_devices:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048) for _ in range(7)]
                )
    except RuntimeError as e:
        print(f"Error setting up logical GPUs: {e}")

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

def compute_humidity_loss(x, y_true, y_pred):
    ps = x[:, input_var_indices['state_ps']]  # Surface pressure (Pa)
    # lets rescale it:
    ps = from_input_normalized_to_original(ps, "state_ps")
    T = y_pred[:, output_var_indices['state_t']]  # Temperature (K)
    q = y_pred[:, output_var_indices['state_q0001']]  # Specific humidity (kg/kg)
    # T and q are already in the original units, so we dont need to rescale them

    # Clamp temperature to a reasonable range (e.g., 200K to 350K)
    T = tf.clip_by_value(T, 200.0, 350.0)

    # Compute saturation vapor pressure using Tetens formula
    Es = 610.94 * tf.exp((17.625 * (T - 273.15)) / (T - 30.11))  # Pa

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-6
    ps_minus_Es = tf.maximum(ps - Es, epsilon)  # Ensure ps - Es > 0
    qs = 0.622 * Es / ps_minus_Es  # Saturation specific humidity (kg/kg)

    # Compute the humidity loss with clamped values
    loss_humidity = tf.reduce_mean(tf.nn.relu(q - qs))

    # Add checks for NaNs/Infs in intermediate values (for debugging)
    tf.debugging.check_numerics(loss_humidity, "NaN or Inf in humidity loss")
    return loss_humidity


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


def compute_energy_loss(x, y_true, y_pred):
    r_out = x[:, input_var_indices['cam_in_LWUP']]
    lh = x[:, input_var_indices['pbuf_LHFLX']]
    sh = x[:, input_var_indices['pbuf_SHFLX']]
    # scale them to the original units, 
    # using mlo_scale, for vars in vars_mlo_0
    # and mli_mean, mli_max, mli_min for vars in vars_mli
    # cam_in_LWUP, pbuf_LHFLX, pbuf_SHFLX are in vars_mli:
    r_out = from_input_normalized_to_original(r_out, "cam_in_LWUP")
    lh = from_input_normalized_to_original(lh, "pbuf_LHFLX")
    sh = from_input_normalized_to_original(sh, "pbuf_SHFLX")

    # Convert sum_vars to a tf.Tensor and use tf.gather
    # sum_vars = [output_var_indices[var] for var in ['cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']]
    # sum_vars_tensor = tf.constant(sum_vars, dtype=tf.int32)

    # r_in = tf.reduce_sum(tf.gather(y_pred, sum_vars_tensor, axis=1), axis=1)
    # first lets get them, then  we rescale and then we sum them
    cam_out_SOLS = y_pred[:, output_var_indices['cam_out_SOLS']]
    cam_out_SOLL = y_pred[:, output_var_indices['cam_out_SOLL']]
    cam_out_SOLSD = y_pred[:, output_var_indices['cam_out_SOLSD']]
    cam_out_SOLLD = y_pred[:, output_var_indices['cam_out_SOLLD']]

    # now we rescale them, because they are outputs, we use mlo_scale
    # we normalized by using mlo_scale like this: 
    # cam_out_SOLS = cam_out_SOLS * mlo_scale
    # so we have to do the opposite:
    cam_out_SOLS = cam_out_SOLS / mlo_scale['cam_out_SOLS']
    cam_out_SOLL = cam_out_SOLL / mlo_scale['cam_out_SOLL']
    cam_out_SOLSD = cam_out_SOLSD / mlo_scale['cam_out_SOLSD']
    cam_out_SOLLD = cam_out_SOLLD / mlo_scale['cam_out_SOLLD']

    # now we sum them
    r_in = cam_out_SOLS + cam_out_SOLL + cam_out_SOLSD + cam_out_SOLLD

    loss_ec = r_in - (r_out + lh + sh)
    return tf.reduce_mean(tf.abs(loss_ec))


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

initial_loss_functions = {
    'mse': (mse_loss, 1.0),
    'energy': (compute_energy_loss, 0),
    'mass': (compute_mass_loss, 0),
    'radiation': (compute_radiation_loss, 0),
    'humidity': (compute_humidity_loss, 0),
    'nonneg': (compute_nonneg_loss, 0),
}
def train_model(model, dataset, val_dataset, model_optimizer, initial_lambdas, lambda_optimizer, epochs, steps_per_epoch, validation_steps, filepath_csv, output_results_dirpath, data_subset_fraction=1.0, patience=5, min_delta=0.001):
    import numpy as np  # Ensure numpy is imported
    # vars_mlo = ["state_t","state_q0001",'ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']
    # lambda_energy, lambda_mass, lambda_radiation, lambda_humidity, lambda_nonneg = lambdas
    TRAINABLE = True
    DONT_TRAIN_THESE = ["mass", "radiation", "humidity", "nonneg"]
    # train_lambda_energy = True if 'energy' not in DONT_TRAIN_THESE else False
    # train_lambda_mass = True if 'mass' not in DONT_TRAIN_THESE else False
    # train_lambda_radiation = True if 'radiation' not in DONT_TRAIN_THESE else False
    # train_lambda_humidity = True if 'humidity' not in DONT_TRAIN_THESE else False
    # train_lambda_nonneg = True if 'nonneg' not in DONT_TRAIN_THESE else False

    if TRAINABLE:
        # Initialize lambda parameters (before training loop)
        lambda_energy_param = tf.Variable(initial_value=initial_loss_functions['energy'][1], trainable=True, dtype=tf.float32)
        lambda_mass_param = tf.Variable(initial_value=initial_loss_functions['mass'][1], trainable=True, dtype=tf.float32)
        lambda_radiation_param = tf.Variable(initial_value=initial_loss_functions['radiation'][1], trainable=True, dtype=tf.float32)
        lambda_humidity_param = tf.Variable(initial_value=initial_loss_functions['humidity'][1], trainable=True, dtype=tf.float32)
        lambda_nonneg_param = tf.Variable(initial_value=initial_loss_functions['nonneg'][1], trainable=True, dtype=tf.float32)
        # lambda_energy_param = tf.Variable(initial_value=initial_loss_functions['energy'][1], trainable= train_lambda_energy, dtype=tf.float32)
        # lambda_mass_param = tf.Variable(initial_value=initial_loss_functions['mass'][1], trainable= train_lambda_mass, dtype=tf.float32)
        # lambda_radiation_param = tf.Variable(initial_value=initial_loss_functions['radiation'][1], trainable= train_lambda_radiation, dtype=tf.float32)
        # lambda_humidity_param = tf.Variable(initial_value=initial_loss_functions['humidity'][1], trainable= train_lambda_humidity, dtype=tf.float32)
        # lambda_nonneg_param = tf.Variable(initial_value=initial_loss_functions['nonneg'][1], trainable= train_lambda_nonneg, dtype=tf.float32)
        # for dont_train_these in DONT_TRAIN_THESE:
        #     if DONT_TRAIN_THESE[dont_train_these]:
        #         if dont_train_these == 'energy':
        #             lambda_energy_param = tf.constant(0.0)
        #         elif dont_train_these == 'mass':
        #             lambda_mass_param = tf.constant(0.0)
        #         elif dont_train_these == 'radiation':
        #             lambda_radiation_param = tf.constant(0.0)
        #         elif dont_train_these == 'humidity':
        #             lambda_humidity_param = tf.constant(0.0)
        #         elif dont_train_these == 'nonneg':
        #             lambda_nonneg_param = tf.constant(0.0)
        # Define lambdas as positive values using softplus
        lambda_energy = tf.constant(0.0)  if 'energy' in DONT_TRAIN_THESE else tf.nn.softplus(lambda_energy_param)
        lambda_mass = tf.constant(0.0)  if 'mass' in DONT_TRAIN_THESE else tf.nn.softplus(lambda_mass_param)
        lambda_radiation = tf.constant(0.0)  if 'radiation' in DONT_TRAIN_THESE else tf.nn.softplus(lambda_radiation_param)
        lambda_humidity = tf.constant(0.0)  if 'humidity' in DONT_TRAIN_THESE else tf.nn.softplus(lambda_humidity_param)
        lambda_nonneg = tf.constant(0.0)  if 'nonneg' in DONT_TRAIN_THESE else tf.nn.softplus(lambda_nonneg_param)
    


    # exclude those that are 0

    # Ensure datasets are repeated
    dataset = dataset.repeat()
    val_dataset = val_dataset.repeat()

    best_val_mse = np.inf  # Initialize best validation MSE
    no_improvement_epochs = 0  # Counter for early stopping

    # Batch-level logging file paths
    train_batch_log_path = f"{output_results_dirpath}/batch_train_log_lambdas_{lambdas_string_with_names}_datafrac_{data_subset_fraction}.csv"
    val_batch_log_path = f"{output_results_dirpath}/batch_val_log_lambdas_{lambdas_string_with_names}_datafrac_{data_subset_fraction}.csv"

    # Create and initialize batch logging CSVs
    with open(train_batch_log_path, "w") as f:
        f.write("epoch,batch,loss,mae,mse,energy_loss,mass_loss,radiation_loss,humidity_loss,nonneg_loss,lambda_energy,lambda_mass,lambda_radiation,lambda_humidity,lambda_nonneg\n")
    with open(val_batch_log_path, "w") as f:
        f.write("epoch,batch,loss,mae,mse,energy_loss,mass_loss,radiation_loss,humidity_loss,nonneg_loss,lambda_energy,lambda_mass,lambda_radiation,lambda_humidity,lambda_nonneg\n")
    
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
        train_energy_loss = 0.0
        train_mass_loss = 0.0
        train_radiation_loss = 0.0
        train_humidity_loss = 0.0
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
                    model.lambda_energy_param,
                    model.lambda_mass_param,
                    model.lambda_radiation_param,
                    model.lambda_humidity_param,
                    model.lambda_nonneg_param
                ]


                with tf.GradientTape(persistent=True) as tape:
                    # Forward pass and loss computation
                    # Access lambda parameters from the model
                    lambda_energy_param = model.lambda_energy_param
                    lambda_mass_param = model.lambda_mass_param
                    lambda_radiation_param = model.lambda_radiation_param
                    lambda_humidity_param = model.lambda_humidity_param
                    lambda_nonneg_param = model.lambda_nonneg_param

                    # Compute lambdas
                    lambda_energy = tf.constant(0.0)  if 'energy' in DONT_TRAIN_THESE else tf.nn.softplus(lambda_energy_param)
                    lambda_mass = tf.constant(0.0)  if 'mass' in DONT_TRAIN_THESE else tf.nn.softplus(lambda_mass_param)
                    lambda_radiation = tf.constant(0.0)  if 'radiation' in DONT_TRAIN_THESE else tf.nn.softplus(lambda_radiation_param)
                    lambda_humidity = tf.constant(0.0)  if 'humidity' in DONT_TRAIN_THESE else tf.nn.softplus(lambda_humidity_param)
                    lambda_nonneg = tf.constant(0.0)  if 'nonneg' in DONT_TRAIN_THESE else tf.nn.softplus(lambda_nonneg_param)

                    # Update loss_functions inside the tape
                    loss_functions = {
                        'mse': (mse_loss, 1.0),  # Keep MSE fixed if desired
                        'energy': (compute_energy_loss, lambda_energy),
                        'mass': (compute_mass_loss, lambda_mass),
                        'radiation': (compute_radiation_loss, lambda_radiation),
                        'humidity': (compute_humidity_loss, lambda_humidity),
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
                lambda_grads = tape.gradient(total_loss, lambda_variables)
                # After computing gradients
                for var, grad in zip(lambda_variables, lambda_grads):
                    if DEBUG:
                        print(f"Gradient norm for {var.name}: {tf.norm(grad).numpy()}")

                # Apply gradients to model weights
                model_optimizer.apply_gradients(zip(model_grads, model_variables))

                # Apply gradients to lambda parameters
                # skip if no lambdas are trainableL
                if len(constant_lambdas) < 5:
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
                    print(f"Lambda Energy: {lambda_energy.numpy()}, Lambda Mass: {lambda_mass.numpy()}, Lambda Radiation: {lambda_radiation.numpy()}, Lambda Humidity: {lambda_humidity.numpy()}, Lambda Non-negativity: {lambda_nonneg.numpy()}")

                # Now, total_loss is your overall loss
                # loss_value_scalar = total_loss.numpy()
                #print
                # print(f"Loss value scalar: {loss_value_scalar}, {loss_value_scalar.shape}, len of loss_components: {len(loss_components)}")

                # Access specific loss components if needed
                energy_loss_value = loss_components.get('energy', tf.constant(0.0)).numpy()
                mass_loss_value = loss_components.get('mass', tf.constant(0.0)).numpy()
                radiation_loss_value = loss_components.get('radiation', tf.constant(0.0)).numpy()
                humidity_loss_value = loss_components.get('humidity', tf.constant(0.0)).numpy()
                nonneg_loss_value = loss_components.get('nonneg', tf.constant(0.0)).numpy()

                # PER VARIABLE METRICS ------------------------------------------------------------------------------------------------------
                # Initialize dictionaries to store per-variable MAE and MSE
                
                # Log batch metrics for training
                # with open(train_batch_log_path, "a") as f:
                #     line = f"{epoch + 1},{step + 1},{total_loss.numpy()},{train_mae.result().numpy()},{train_mse.result().numpy()}, {energy_loss_value},{mass_loss_value},{radiation_loss_value},{humidity_loss_value},{nonneg_loss_value},"
                #     for loss_name, loss_value in loss_components.items():
                #         line += f"{loss_value.numpy()},"
                #     line = line[:-1] + "\n"
                #     f.write(line)
                # Log batch metrics for training
                with open(train_batch_log_path, "a") as f:
                    # Construct the line for CSV
                    line = f"{epoch + 1},{step + 1},{total_loss.numpy()},{train_mae.result().numpy()},{train_mse.result().numpy()},{energy_loss_value},{mass_loss_value},{radiation_loss_value},{humidity_loss_value},{nonneg_loss_value}," \
                        f"{lambda_energy.numpy()},{lambda_mass.numpy()},{lambda_radiation.numpy()},{lambda_humidity.numpy()},{lambda_nonneg.numpy()}\n"
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
                train_energy_loss += energy_loss_value
                train_mass_loss += mass_loss_value
                train_radiation_loss += radiation_loss_value
                train_humidity_loss += humidity_loss_value
                train_nonneg_loss += nonneg_loss_value
                train_mae.update_state(y_batch_train, y_pred)
                train_mse.update_state(y_batch_train, y_pred)
                step_count += 1

                
                # Update tqdm progress bar
                pbar.set_postfix({
                    "Loss": f"{total_loss.numpy():.4f}",
                    # "E. Loss": f"{energy_loss_value:.4f}",
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
        avg_epoch_energy_loss = train_energy_loss / step_count
        avg_epoch_mass_loss = train_mass_loss / step_count
        avg_epoch_radiation_loss = train_radiation_loss / step_count
        avg_epoch_humidity_loss = train_humidity_loss / step_count
        avg_epoch_nonneg_loss = train_nonneg_loss / step_count

        print(f"Average Training Loss for Epoch {epoch + 1}: {avg_epoch_loss:.4f}, "
              f"MAE: {avg_epoch_mae:.4f}, MSE: {avg_epoch_mse:.4f} "
                f"E. Loss: {avg_epoch_energy_loss:.4f}, "
                f"Mass Loss: {avg_epoch_mass_loss:.4f}, "
                f"Radiation Loss: {avg_epoch_radiation_loss:.4f}, "
                f"Humidity Loss: {avg_epoch_humidity_loss:.4f}, "
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
        val_energy_loss = 0.0
        val_mass_loss = 0.0
        val_radiation_loss = 0.0
        val_humidity_loss = 0.0
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
                    line = f"{epoch + 1},{step + 1},{val_loss_value},{val_mae.result().numpy()},{val_mse.result().numpy()}, {val_energy_loss},{val_mass_loss},{val_radiation_loss},{val_humidity_loss},{val_nonneg_loss},"
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
                val_energy_loss += loss_components.get('energy', tf.constant(0.0)).numpy()
                val_mass_loss += loss_components.get('mass', tf.constant(0.0)).numpy()
                val_radiation_loss += loss_components.get('radiation', tf.constant(0.0)).numpy()
                val_humidity_loss += loss_components.get('humidity', tf.constant(0.0)).numpy()
                val_nonneg_loss += loss_components.get('nonneg', tf.constant(0.0)).numpy()
                val_step_count += 1
                pbar.update(1)

        # Calculate average validation metrics
        avg_val_loss = val_loss / val_step_count
        avg_val_mae = val_mae.result().numpy()
        avg_val_mse = val_mse.result().numpy()
        avg_val_energy_loss = val_energy_loss / val_step_count
        avg_val_mass_loss = val_mass_loss / val_step_count
        avg_val_radiation_loss = val_radiation_loss / val_step_count
        avg_val_humidity_loss = val_humidity_loss / val_step_count
        avg_val_nonneg_loss = val_nonneg_loss / val_step_count


        print(f"Validation Loss for Epoch {epoch + 1}: {avg_val_loss:.4f}, "
              f"MAE: {avg_val_mae:.4f}, MSE: {avg_val_mse:.4f}, "
              f"E. Loss: {avg_val_energy_loss:.4f}, "
                f"Mass Loss: {avg_val_mass_loss:.4f}, "
                f"Radiation Loss: {avg_val_radiation_loss:.4f}, "
                f"Humidity Loss: {avg_val_humidity_loss:.4f}, "
                f"Non-negativity Loss: {avg_val_nonneg_loss:.4f}")
        

        # Log epoch metrics to the epoch-level CSV
        with open(filepath_csv, "a") as f:
            line = f"{epoch + 1},{avg_epoch_loss},{avg_epoch_mae},{avg_epoch_mse}," + \
            f"{avg_epoch_energy_loss},{avg_val_loss},{avg_val_mae},{avg_val_mse},"
            for loss_name, loss_value in loss_components.items():
                line += f"{loss_value.numpy()},"
            line = line[:-1] + "\n"
            f.write(line)
            
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
        ds = xr.open_dataset(example_file, engine='netcdf4')
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
    print(physical_devices)
    try:
        for kgpu in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[kgpu], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    tf.config.list_physical_devices('GPU')
    print("GPU setup done")

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
            ds = xr.open_dataset(file, engine='netcdf4')
            # subset ozone, ch4, n2o

            # ds_utls = ds[vars_mli_utls]\
            #             .isel(lev=slice(5,21)).rename({'lev':'lev2'})
            # combine ds and ds_utls
            ds = ds[vars_mli]

            # ds = ds.merge(ds_utls)
            
            # output read / preprocess #
            # read mlo (-> dso)
            dso = xr.open_dataset(file.replace('.mli.','.mlo.'), engine='netcdf4')
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

        corrected_initial_lambdas = {}
        for var in initial_lambdas:
            if initial_lambdas[var] ==0:
                corrected_initial_lambdas[var] = (-1e-10)
            else:
                corrected_initial_lambdas[var] = initial_lambdas[var]

        # Initialize trainable lambda parameters
        self.lambda_energy_param = self.add_weight(
            name='lambda_energy_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(corrected_initial_lambdas['energy']),
            trainable=(initial_lambdas['energy'] != 0) & ('energy' not in constant_lambdas),
        )
        self.lambda_mass_param = self.add_weight(
            name='lambda_mass_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(corrected_initial_lambdas['mass']),
            trainable=(initial_lambdas['mass'] != 0),
        )
        self.lambda_radiation_param = self.add_weight(
            name='lambda_radiation_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(corrected_initial_lambdas['radiation']),
            trainable=(initial_lambdas['radiation'] != 0),
        )
        self.lambda_humidity_param = self.add_weight(
            name='lambda_humidity_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(corrected_initial_lambdas['humidity']),
            trainable=(initial_lambdas['humidity'] != 0),
        )
        self.lambda_nonneg_param = self.add_weight(
            name='lambda_nonneg_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(corrected_initial_lambdas['nonneg']),
            trainable=(initial_lambdas['nonneg'] != 0),
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
                'energy': self.lambda_energy_param.numpy(),
                'mass': self.lambda_mass_param.numpy(),
                'radiation': self.lambda_radiation_param.numpy(),
                'humidity': self.lambda_humidity_param.numpy(),
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

# # Instantiate your base model
# base_model = build_model()

# # Wrap it with the custom model
# model = CustomModel(base_model)


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

def main(lambda_energy, 
        lambda_mass,
        lambda_radiation,
        lambda_humidity,
        lambda_nonneg,
        output_results_dirpath, data_subset_fraction, n_epochs, lr, batch_size=32):
    start_time = time.time()
    N_EPOCHS = n_epochs
    shuffle_buffer = 12 * 384  # ncol=384

    # append to results_{args.lambda_energy} to the output_results_dirpath
    output_results_dirpath = f"{output_results_dirpath}/results_{data_subset_fraction}"

    # Set up GPU memory growth
    setup_gpu()

    # Paths
    norm_path = f"{root_climsim_dirpath}/preprocessing/normalizations/"
    root_train_path = (
        f"{root_huggingface_data_dirpath}/datasets--LEAP--ClimSim_low-res/snapshots/"
        "bab82a2ebdc750a0134ddcd0d5813867b92eed2a/train/"
    )

    global vars_mlo, vars_mlo_0, vars_mli, vars_mlo_dims, vars_mli_dims, input_var_indices, output_var_indices, initial_loss_functions, lambdas_string_with_names, mlo_scale, mli_mean, mli_max, mli_min, constant_lambdas
    constant_lambdas = ["energy", "mass", "radiation", "humidity", "nonneg"]

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
    # vars_mli      = ['state_t','state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_v', 
    #                  'state_ps', 'pbuf_SOLIN','pbuf_LHFLX', 'pbuf_SHFLX',  'pbuf_TAUX', 'pbuf_TAUY', 'pbuf_COSZRS',
    #                  'cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_LWUP', 
    #                  'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_OCNFRAC', 'cam_in_SNOWHICE', 'cam_in_SNOWHLAND']
    vars_mli  = ['cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_LWUP', 'cam_in_OCNFRAC', 'cam_in_SNOWHICE', 'cam_in_SNOWHLAND', 'pbuf_COSZRS', 'pbuf_LHFLX', 'pbuf_SHFLX', 'pbuf_SOLIN', 'pbuf_TAUX', 'pbuf_TAUY', 'state_pmid', 'state_ps', 'state_q0001', 'state_q0002', 'state_q0003', 'state_t', 'state_u', 'state_v', 'pbuf_CH4', 'pbuf_N2O', 'pbuf_ozone']
    vars_mlo      = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v',
                     'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 
                     'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD', "state_t", "state_q0001"]
    # vars_mlo_0 is the same but without state_t, vars_mlo_0 corresponds with the scaleing factor
    vars_mlo_0   = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v',
                     'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 
                     'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

    # vars_mlo=list(mlo.data_vars.keys())[2:]
    # add the other variable shtat will be created in the generator
    # vars_mlo  = vars_mlo + ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v']
    print(f"vars_mlo: {vars_mlo}")
    vars_mlo_dims = [(mlo_scale[var].values.size) for var in vars_mlo_0]
    # vars_mlo_dims = vars_mlo_dims + [1, 1, 1, 1, 1, 1]
    assert len(vars_mlo_0) == len(vars_mlo_dims), f"vars_mlo and vars mlo_dims dont share the same length: {len(vars_mlo_0)} != {len(vars_mlo_dims)}"
    vars_mli = list(mli.data_vars.keys())[2:]
    print(f"vars_mli: {vars_mli}")
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

    initial_lambdas = {
        'energy': lambda_energy,
        'mass': lambda_mass,
        'radiation': lambda_radiation,
        'humidity': lambda_humidity,
        'nonneg': lambda_nonneg,
    }

    # Define model
    base_model = build_model()
    print(f"initial_lambdas: {initial_lambdas}")
    # exit()
    model = CustomModel(base_model, initial_lambdas)



    # Set up optimizer
    # optimizer = keras.optimizers.Adam(learning_rate=lr)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipvalue=1.0)
    # Optimizer for model weights
    model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipvalue=1.0)

    # Optimizer for lambda parameters with a higher learning rate
    lambda_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, clipvalue=1.0)

    


    # Ensure Log directory exists
    os.makedirs(output_results_dirpath, exist_ok=True)
    lambdas_string_with_names = "_".join([f"{name}_{value[1]}" for name, value in zip(initial_loss_functions.keys(), initial_loss_functions.values())])
    filepath_csv = f'{output_results_dirpath}/csv_logger_lambdsa_{lambdas_string_with_names}_datafrac_{data_subset_fraction}.csv'
    with open(filepath_csv, "w") as f:
        # headers
        line = "epoch,train_loss,train_mae,train_mse," + \
        "train_energy_loss,train_mass_loss,train_radiation_loss,"+ \
        "train_humidity_loss,train_nonneg_loss," + \
        "val_loss,val_mae,val_mse," + \
        "val_energy_loss,val_mass_loss,val_radiation_loss," + \
        "val_humidity_loss,val_nonneg_loss\n"
        f.write(line)

    # Train model using custom training loop
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
    filepath_csv, 
    output_results_dirpath,
    data_subset_fraction
    )

    print("DONE")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument(
        '--lambda_energy',
        type=float,
        default=0.1,
        help='Value of lambda_energy'
    )
    parser.add_argument(
        '--lambda_mass',
        type=float,
        default=0.1,
        help='Value of LAMBDA_MASS'
    )
    parser.add_argument(
        '--lambda_radiation',
        type=float,
        default=0.1,
        help='Value of LAMBDA_RADIATION'
    )
    parser.add_argument(
        '--lambda_humidity',
        type=float,
        default=0.1,
        help='Value of LAMBDA_HUMIDITY'
    )
    parser.add_argument(
        '--lambda_nonneg',
        type=float,
        default=0.1,
        help='Value of LAMBDA_NONNEG'
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
        args.lambda_energy,
        args.lambda_mass,
        args.lambda_radiation,
        args.lambda_humidity,
        args.lambda_nonneg,
        args.output_results_dirpath,
        args.data_subset_fraction,
        args.n_epochs,
        args.lr,
        args.batch_size
    )

# python combined_loss_model.py --lambda_energy=0.1 --lambda_mass=0.1 --lambda_radiation=0.1 --lambda_humidity=0.1 --lambda_nonneg=0.1 --output_results_dirpath=/home/alvarovh/code/cse598_climate_proj/results_new/ --data_subset_fraction=0.01 --n_epochs=1

#python combined_loss_model.py --lambda_energy=0.1 --lambda_mass=0.1 --lambda_radiation=0.1 --lambda_humidity=0.1 --lambda_nonneg=0.1 --output_results_dirpath=/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov24/trainable_lambdas --data_subset_fraction=0.1 --n_epochs=10

