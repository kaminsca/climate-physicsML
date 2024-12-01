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
    load_normalization_data,
)
from config import vars_mli, vars_mlo, norm_path, test_subset_dirpath
import argparse

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

def evaluate_model(model, dataset, steps, model_path, output_dir):
    # Metrics
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    mse_metric = tf.keras.metrics.MeanSquaredError()
    variable_mae = {var: 0.0 for var in vars_mlo}
    variable_mse = {var: 0.0 for var in vars_mlo}

    # Initialize variables to accumulate metrics
    total_samples = 0
    start_time = time.time()

    for step, (x_batch, y_batch) in enumerate(dataset):
        if step >= steps:
            break
        y_pred = model(x_batch, training=False)

        # Update overall metrics
        mae_metric.update_state(y_batch, y_pred)
        mse_metric.update_state(y_batch, y_pred)

        batch_size = x_batch.shape[0]
        total_samples += batch_size

        # Per-variable metrics
        for i, var_name in enumerate(vars_mlo):
            var_mae_metric = tf.keras.metrics.MeanAbsoluteError()
            var_mse_metric = tf.keras.metrics.MeanSquaredError()

            var_mae_metric.update_state(y_batch[:, i], y_pred[:, i])
            var_mse_metric.update_state(y_batch[:, i], y_pred[:, i])

            variable_mae[var_name] += var_mae_metric.result().numpy() * batch_size
            variable_mse[var_name] += var_mse_metric.result().numpy() * batch_size

    # Calculate average metrics
    avg_mae = mae_metric.result().numpy()
    avg_mse = mse_metric.result().numpy()

    for var_name in vars_mlo:
        variable_mae[var_name] /= total_samples
        variable_mse[var_name] /= total_samples

    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")
    print(f"Average MAE: {avg_mae:.6f}")
    print(f"Average MSE: {avg_mse:.6f}")

    # Save metrics to a file
    model_id = os.path.basename(model_path).replace('.keras', '')
    metrics_file = os.path.join(output_dir, f'general_metrics_{model_id}.txt')
    per_var_metrics_file = os.path.join(output_dir, f'per_variable_metrics_{model_id}.txt')

    with open(metrics_file, 'w') as f:
        f.write(f"mae,mse,model_basename\n")
        f.write(f"{avg_mae:.6f},{avg_mse:.6f},{os.path.basename(model_path)}\n")    
    with open(per_var_metrics_file, 'w') as f:
        f.write(f"var_name,mae,mse,model_basename\n")
        for var_name in vars_mlo:
            f.write(f"{var_name},{variable_mae[var_name]:.6f},{variable_mse[var_name]:.6f},{os.path.basename(model_path)}\n")
    print(f"Metrics saved to {metrics_file}")

def main(model_path, data_file=None, batch_size=32, output_dir='prediction_results'):
    setup_gpu()

    # Load the model
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Load normalization parameters
    mli_mean, mli_max, mli_min, mlo_scale = load_normalization_data(norm_path)

    # Prepare dataset
    if data_file:
        # Process a single file
        file_list = [data_file]
        print(f"Processing single file: {data_file}")
    else:
        # Load test dataset
        print("Loading test dataset...")
        file_list = prepare_test_files(data_subset_fraction=1.0)  # Use full test set

    # Create dataset
    shuffle_buffer = 12 * 384  # Adjust as needed
    dataset = create_dataset(
        file_list, vars_mli, vars_mlo, mli_mean,
        mli_max, mli_min, mlo_scale, shuffle_buffer, batch_size,
        shuffle=False  # No need to shuffle during evaluation
    )

    # Calculate steps
    total_samples = calculate_dataset_size(file_list, vars_mli)
    steps = int(np.ceil(total_samples / batch_size))
    print(f"Total samples: {total_samples}, Steps: {steps}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate the model
    evaluate_model(model, dataset, steps, model_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the model.')
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
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        data_file=args.data_file,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
# python predict.py --model_path /nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov30/zeromodel/results_0.01/best_model_lambdas_excluded_mass_excluded_radiation_excluded_nonneg_constant_mass_constant_radiation_constant_nonneg_scaledfactorofmse_1.00_scaledfactorofmass_1.00_scaledfactorofradiation_1.00_scaledfactorofnonneg_1.00_datafrac_0.01_epoch_1.keras --output_dir /nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_subset/datafraction_0.01/prediction_results