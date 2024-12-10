#!/usr/bin/env python

import os
import sys
import glob
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import CustomModel, load_model
from preprocess_data import (
    create_dataset,
    calculate_dataset_size,
    prepare_test_files,
    prepare_validation_files,
    load_normalization_data,
)
from config import vars_mli, vars_mlo, norm_path, test_subset_dirpath, grid_path
import argparse
import xarray as xr

# seeds
np.random.seed(42)
tf.random.set_seed(42)

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

def from_output_scaled_to_original(scaled, var_name, mlo_scale):
    """
    Rescale output variables back to their original scale.
    """
    if var_name in ["state_t", "state_q0001"]:
        return scaled
    return scaled / mlo_scale[var_name].values
def evaluate_model(model, dataset, steps, output_dir, mlo_scale, model_id, test_or_val="test", file_basenames_to_save=None, lat_to_save=None, lon_to_save=None):
    if test_or_val == None:
        test_or_val = ""
    # Metrics
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    mse_metric = tf.keras.metrics.MeanSquaredError()
    variable_mae = {var: 0.0 for var in vars_mlo}
    variable_mse = {var: 0.0 for var in vars_mlo}

    # Initialize variables to accumulate metrics
    total_samples = 0
    start_time = time.time()

    # Lists to store predictions and true values
    all_y_pred = []
    all_y_true = []

    # Initialize lists for per-batch metrics
    per_batch_mae = []
    per_batch_mse = []

    # Initialize per-variable per-batch metrics file
    per_batch_variable_metrics_file = os.path.join(output_dir, f'{test_or_val}per_variable_pbtchmet_{model_id}.csv')
    f_per_var = open(per_batch_variable_metrics_file, 'w')
    f_per_var.write('batch,variable_name,mae,mse\n')

    for step, (x_batch, y_batch) in enumerate(dataset):
        if step >= steps:
            break
        y_pred = model(x_batch, training=False)

        # Append predictions and true values
        all_y_pred.append(y_pred.numpy())
        all_y_true.append(y_batch.numpy())

        # Update overall metrics
        mae_metric.update_state(y_batch, y_pred)
        mse_metric.update_state(y_batch, y_pred)

        # Compute batch MAE and MSE using TensorFlow operations
        batch_mae = tf.reduce_mean(tf.abs(y_batch - y_pred)).numpy()
        batch_mse = tf.reduce_mean(tf.square(y_batch - y_pred)).numpy()

        per_batch_mae.append(batch_mae)
        per_batch_mse.append(batch_mse)

        batch_size = x_batch.shape[0]
        total_samples += batch_size

        # Compute per-variable metrics for the current batch
        for i, var_name in enumerate(vars_mlo):
            # Compute per-variable MAE and MSE for the batch
            var_mae = tf.reduce_mean(tf.abs(y_batch[:, i] - y_pred[:, i])).numpy()
            var_mse = tf.reduce_mean(tf.square(y_batch[:, i] - y_pred[:, i])).numpy()

            # Accumulate over all batches for overall metrics
            variable_mae[var_name] += var_mae * batch_size
            variable_mse[var_name] += var_mse * batch_size

            # Write per-variable per-batch metrics to file
            f_per_var.write(f"{step+1},{var_name},{var_mae:.6f},{var_mse:.6f}\n")

    # Close the per-variable per-batch metrics file
    f_per_var.close()

    # Concatenate all predictions and true values
    all_y_pred = np.concatenate(all_y_pred, axis=0)
    all_y_true = np.concatenate(all_y_true, axis=0)

    # Rescale predictions and true values to original scale
    y_pred_rescaled = np.zeros_like(all_y_pred)
    y_true_rescaled = np.zeros_like(all_y_true)

    for i, var_name in enumerate(vars_mlo):
        y_pred_rescaled[:, i] = from_output_scaled_to_original(all_y_pred[:, i], var_name, mlo_scale)
        y_true_rescaled[:, i] = from_output_scaled_to_original(all_y_true[:, i], var_name, mlo_scale)

    # Save rescaled predictions and true values
    predictions_file = os.path.join(output_dir, f'{test_or_val}preds_{model_id}.npz')
    np.savez(predictions_file, y_pred=y_pred_rescaled, y_true=y_true_rescaled, variables=vars_mlo, file_basenames=file_basenames_to_save, latitudes=lat_to_save, longitudes=lon_to_save) 
    print(f"Rescaled predictions saved to {predictions_file}")
    print(f"confirming the shape of the saved file: {y_pred_rescaled.shape}, {y_true_rescaled.shape}, {len(vars_mlo)}", len(file_basenames_to_save), len(lat_to_save), len(lon_to_save)) 

    # Calculate average metrics
    avg_mae = mae_metric.result().numpy()
    avg_mse = mse_metric.result().numpy()

    global_mae = avg_mae
    global_mse = avg_mse

    for var_name in vars_mlo:
        variable_mae[var_name] /= total_samples
        variable_mse[var_name] /= total_samples

    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")
    print(f"Average MAE: {avg_mae:.6f}")
    print(f"Average MSE: {avg_mse:.6f}")
    # Print state_t mse:
    print(f"state_t MSE: {variable_mse['state_t']:.6f}")

    # Save metrics to a file
    metrics_file = os.path.join(output_dir, f'{test_or_val}met_{model_id}.csv')
    with open(metrics_file, 'w') as f:
        f.write("variable_name,mae,mse\n")
        for var_name in vars_mlo:
            f.write(f"{var_name},{variable_mae[var_name]:.6f},{variable_mse[var_name]:.6f}\n")
        f.write(f"Average,{avg_mae:.6f},{avg_mse:.6f}\n")

    print(f"Metrics saved to {metrics_file}")

    # Save per-batch metrics to a file
    per_batch_metrics_file = os.path.join(output_dir, f'{test_or_val}pbtchmet_{model_id}.csv')
    with open(per_batch_metrics_file, 'w') as f:
        f.write("batch,mae,mse\n")
        for i, (mae, mse) in enumerate(zip(per_batch_mae, per_batch_mse)):
            f.write(f"{i+1},{mae:.6f},{mse:.6f}\n")
    print(f"Per-batch metrics saved to {per_batch_metrics_file}")


def main(model_path, data_file=None, batch_size=32, output_dir='prediction_results', predict_validation=False):
    model_id = os.path.basename(model_path).replace('.keras', '').replace("best_model_lambdas_", "")

    # error if predict_validation is True and data_file is not None
    if predict_validation and data_file:
        raise ValueError("Cannot specify both predict_validation and data_file")

    setup_gpu()

    # Load the model
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Load normalization parameters
    mli_mean, mli_max, mli_min, mlo_scale = load_normalization_data(norm_path)

    # Load grid
    ds_grid = xr.open_dataset(grid_path, engine='netcdf4')
    latitudes = ds_grid['lat'].values.tolist()
    longitudes = ds_grid['lon'].values.tolist()

    all_file_lists = []
    all_file_lists_setnames = []
    # Prepare dataset
    if data_file:
        # Process a single file
        fl = [data_file]
        print(f"Processing single file: {data_file}")
        all_file_lists.append(fl)
        all_file_lists_setnames.append(None)
    else:
        # Load test dataset
        print("Loading test dataset...")
        test_file_list = prepare_test_files(data_subset_fraction=1.0)  # Use full test set
        all_file_lists.append(test_file_list)
        all_file_lists_setnames.append("test")
        if predict_validation:
            # Use the validation set instead of the test set
            print("Using the validation set instead of the test set")
            val_file_list = prepare_validation_files()
            all_file_lists.append(val_file_list)
            all_file_lists_setnames.append("val")

    shuffle_buffer = 12 * 384  # Adjust as needed
    for file_list, setname in zip(all_file_lists, all_file_lists_setnames):
        # Create dataset
        dataset = create_dataset(
            file_list, vars_mli, vars_mlo, mli_mean,
            mli_max, mli_min, mlo_scale, shuffle_buffer, batch_size,
            shuffle=False  # No need to shuffle during evaluation
        )
        # filenames to save
        
        # file_basenames_to_save = [[os.path.basename(f)] * 384 for f in file_list]
        # # flatten in one list
        # file_basenames_to_save = [item for sublist in file_basenames_to_save for item in sublist]
        # To repeat the filename 384 for filename 1, then for 2, instead of those two steps we can:
        file_basenames_to_save = [os.path.basename(f) for f in file_list for _ in range(384)]

        lat_to_save = latitudes * len(file_list)
        lon_to_save = longitudes * len(file_list)
        # print(f"Shapes of file_basenames_to_save, lat_to_save, lon_to_save: {len(file_list)}, {len(latitudes)}, {len(longitudes)}")
        # print(f"Lets inspect the first 10 elements of file_basenames_to_save: {file_basenames_to_save[:10]}")
        # print(f"Lets inspect the 384-350 elements of file_basenames_to_save: {file_basenames_to_save[383:394]}")
        # print(f'Lets inspect file_list: {file_list}')
        # exit()

        # Calculate steps
        total_samples = calculate_dataset_size(file_list, vars_mli)
        steps = int(np.ceil(total_samples / batch_size))
        print(f"Total samples: {total_samples}, Steps: {steps}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Evaluate the model and save predictions
        evaluate_model(model, dataset, steps, output_dir, mlo_scale, model_id, test_or_val= setname, file_basenames_to_save=file_basenames_to_save, lat_to_save=lat_to_save, lon_to_save=lon_to_save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the model and save predictions.')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model file (.keras)'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default=None,
        help='Path to a single data file to process (optional)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='prediction_results',
        help='Directory to save the prediction results and metrics'
    )
    parser.add_argument(
        '--predict_validation',
        action='store_true',
        help='Predict on the validation set'
    )
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        data_file=args.data_file,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        predict_validation=args.predict_validation
    )

# MASS
# python predict.py --predict_validation --model_path /nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov30/massmodel/results_0.01/best_model_lambdas_excluded_radiation_excluded_nonneg_constant_radiation_constant_nonneg_scaledfactorofmse_1.00_scaledfactorofmass_1.00_scaledfactorofradiation_1.00_scaledfactorofnonneg_1.00_datafrac_0.01_epoch_1.keras --output_dir /nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_subset/datafraction_0.01/prediction_results
# ZERO
# python predict.py --predict_validation --model_path /nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov30/zeromodel/results_0.01/best_model_lambdas_excluded_mass_excluded_radiation_excluded_nonneg_constant_mass_constant_radiation_constant_nonneg_scaledfactorofmse_1.00_scaledfactorofmass_1.00_scaledfactorofradiation_1.00_scaledfactorofnonneg_1.00_datafrac_0.01_epoch_1.keras --output_dir /nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_subset/datafraction_0.01/prediction_results

# MASS
# TEST
# Average MAE: 0.420916
# Average MSE: 1.186408
# state_t MSE: 14.396312
# VAL
# Average MAE: 0.414890
# Average MSE: 1.138606
# state_t MSE: 13.453640



# Average MAE: 0.321673
# Average MSE: 0.605441
# state_t MSE: 5.100840