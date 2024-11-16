#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import random
import pickle
import os
import math
import tqdm
from tqdm import tqdm

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras

root_huggingface_data_dirpath = "/nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_all/"
root_climsim_dirpath = "/home/alvarovh/code/cse598_climate_proj/ClimSim/"

import tensorflow as tf
from tensorflow import keras
import numpy as np

def energy_loss(x, y_true, y_pred):
    r_out = x[:, 124]  # cam_in_lwup
    lh = x[:, 122]     # pbuf_lhflx
    sh = x[:, 123]     # pbuf_shflx
    r_in = tf.reduce_sum(y_true[:, 124:128], axis=1)
    loss_ec = r_in - (r_out + lh + sh)
    return tf.reduce_mean(tf.abs(loss_ec))

def combined_loss(x, y_true, y_pred, lambda_energy):
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(y_true, y_pred)
    weighted_energy_loss = lambda_energy * energy_loss(x, y_true, y_pred)
    return mse_loss + weighted_energy_loss

def train_model(model, dataset, val_dataset, lambda_energy, optimizer, epochs, steps_per_epoch, validation_steps, filepath_csv, output_results_dirpath, data_subset_fraction=1.0):
    import numpy as np  # Ensure numpy is imported

    # Ensure datasets are repeated
    dataset = dataset.repeat()
    val_dataset = val_dataset.repeat()

    best_val_mse = np.inf  # Initialize best validation MSE

    # Batch-level logging file paths
    train_batch_log_path = f"{output_results_dirpath}/batch_train_log_lambda_{lambda_energy}_datafrac_{data_subset_fraction}.csv"
    val_batch_log_path = f"{output_results_dirpath}/batch_val_log_lambda_{lambda_energy}_datafrac_{data_subset_fraction}.csv"

    # Create and initialize batch logging CSVs
    with open(train_batch_log_path, "w") as f:
        f.write("epoch,batch,loss,mae,mse,energy_loss\n")
    with open(val_batch_log_path, "w") as f:
        f.write("epoch,batch,loss,mae,mse,energy_loss\n")

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

        # Training loop with tqdm
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1} Training", unit="step") as pbar:
            for step, (x_batch_train, y_batch_train) in enumerate(dataset):
                if step >= steps_per_epoch:
                    break

                with tf.GradientTape() as tape:
                    y_pred = model(x_batch_train, training=True)
                    loss_value = combined_loss(x_batch_train, y_batch_train, y_pred, lambda_energy)

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Compute metrics for the batch
                loss_value_scalar = loss_value.numpy()
                energy_loss_value = tf.reduce_mean(energy_loss(x_batch_train, y_batch_train, y_pred)).numpy()
                train_mae.update_state(y_batch_train, y_pred)
                train_mse.update_state(y_batch_train, y_pred)

                # Log batch metrics
                with open(train_batch_log_path, "a") as f:
                    f.write(f"{epoch + 1},{step + 1},{loss_value_scalar},{train_mae.result().numpy()},{train_mse.result().numpy()},{energy_loss_value}\n")

                # Accumulate epoch-level metrics
                epoch_loss += loss_value_scalar
                train_energy_loss += energy_loss_value
                step_count += 1

                # Update tqdm progress bar
                pbar.set_postfix({
                    "Loss": f"{loss_value_scalar:.4f}",
                    "E. Loss": f"{energy_loss_value:.4f}",
                    "MAE": f"{train_mae.result().numpy():.4f}",
                    "MSE": f"{train_mse.result().numpy():.4f}",
                })
                pbar.update(1)

        # Compute averages for the epoch
        avg_epoch_loss = epoch_loss / step_count
        avg_epoch_mae = train_mae.result().numpy()
        avg_epoch_mse = train_mse.result().numpy()
        avg_epoch_energy_loss = train_energy_loss / step_count
        print(f"Average Training Loss for Epoch {epoch + 1}: {avg_epoch_loss:.4f}, "
              f"MAE: {avg_epoch_mae:.4f}, MSE: {avg_epoch_mse:.4f}, "
              f"E. Loss: {avg_epoch_energy_loss:.4f}")

        # Reset metrics for the next epoch
        train_mae.reset_state()
        train_mse.reset_state()

        # Validation loop
        val_loss = 0.0
        val_mae = tf.keras.metrics.MeanAbsoluteError()
        val_mse = tf.keras.metrics.MeanSquaredError()
        val_energy_loss = 0.0
        val_step_count = 0

        print("Validation")
        with tqdm(total=validation_steps, desc=f"Epoch {epoch+1} Validation", unit="step") as pbar:
            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                if step >= validation_steps:
                    break
                y_val_pred = model(x_batch_val, training=False)
                val_loss_value = combined_loss(x_batch_val, y_batch_val, y_val_pred, lambda_energy).numpy()
                val_energy_loss_value = tf.reduce_mean(energy_loss(x_batch_val, y_batch_val, y_val_pred)).numpy()
                val_mae.update_state(y_batch_val, y_val_pred)
                val_mse.update_state(y_batch_val, y_val_pred)

                # Log batch metrics for validation
                with open(val_batch_log_path, "a") as f:
                    f.write(f"{epoch + 1},{step + 1},{val_loss_value},{val_mae.result().numpy()},{val_mse.result().numpy()},{val_energy_loss_value}\n")

                # Accumulate validation metrics
                val_loss += val_loss_value
                val_energy_loss += val_energy_loss_value
                val_step_count += 1
                pbar.update(1)

        # Calculate average validation metrics
        avg_val_loss = val_loss / val_step_count
        avg_val_mae = val_mae.result().numpy()
        avg_val_mse = val_mse.result().numpy()
        avg_val_energy_loss = val_energy_loss / val_step_count

        print(f"Validation Loss for Epoch {epoch + 1}: {avg_val_loss:.4f}, "
              f"MAE: {avg_val_mae:.4f}, MSE: {avg_val_mse:.4f}, "
              f"E. Loss: {avg_val_energy_loss:.4f}")

        # Log epoch metrics to the epoch-level CSV
        with open(filepath_csv, "a") as f:
            f.write(f"{epoch + 1},{avg_epoch_loss},{avg_epoch_mae},{avg_epoch_mse},"
                    f"{avg_epoch_energy_loss},{avg_val_loss},{avg_val_mae},{avg_val_mse},"
                    f"{avg_val_energy_loss}\n")

        # Save model if validation MSE improved
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            model_save_path = os.path.join(output_results_dirpath, f"best_model_lambda_{lambda_energy}_datafrac_{data_subset_fraction}_epoch_{epoch + 1}.keras")
            model.save(model_save_path)
            print(f"Model saved at {model_save_path} with improved validation MSE: {best_val_mse:.4f}")
        else:
            print(f"No improvement in validation MSE: {avg_val_mse:.4f}")



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

def main(LAMBDA_ENERGY, output_results_dirpath, data_subset_fraction, n_epochs):
    N_EPOCHS = n_epochs
    shuffle_buffer = 12 * 384  # ncol=384
    batch_size = 32

    # append to results_{args.lambda_energy} to the output_results_dirpath
    output_results_dirpath = f"{output_results_dirpath}/results_{data_subset_fraction}"

    # Set up GPU memory growth
    setup_gpu()

    # Variable lists
    vars_mlo = [
        'ptend_t', 'ptend_q0001', 'cam_out_NETSW', 'cam_out_FLWDS',
        'cam_out_PRECSC', 'cam_out_PRECC', 'cam_out_SOLS',
        'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD'
    ]
    vars_mli = [
        'state_t', 'state_q0001', 'state_ps', 'pbuf_SOLIN',
        'pbuf_LHFLX', 'pbuf_SHFLX', 'cam_in_LWUP'
    ]

    # Paths
    norm_path = f"{root_climsim_dirpath}/preprocessing/normalizations/"
    root_train_path = (
        f"{root_huggingface_data_dirpath}/datasets--LEAP--ClimSim_low-res/snapshots/"
        "bab82a2ebdc750a0134ddcd0d5813867b92eed2a/train/"
    )

    # Load normalization datasets
    mli_mean = xr.open_dataset(norm_path + 'inputs/input_mean.nc')
    mli_max = xr.open_dataset(norm_path + 'inputs/input_max.nc')
    mli_min = xr.open_dataset(norm_path + 'inputs/input_min.nc')
    mlo_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc')

    # Load file list
    all_files = load_file_list(root_train_path)

    # Apply data subset fraction
    subset_size = int(len(all_files) * data_subset_fraction)
    all_files = all_files[:subset_size]

    random.shuffle(all_files)

    # Split files into training and validation sets
    val_proportion = 0.1
    split_index = int(len(all_files) * (1 - val_proportion))
    f_mli_train = all_files[:split_index]
    f_mli_val = all_files[split_index:]

    print(f'[TRAIN] Total # of input files after applying limit: {len(f_mli_train)}')

    # Create training and validation datasets
    tds = create_dataset(
        f_mli_train, vars_mli, vars_mlo, mli_mean,
        mli_max, mli_min, mlo_scale, shuffle_buffer, batch_size
    )
    tds_val = create_dataset(
        f_mli_val, vars_mli, vars_mlo, mli_mean,
        mli_max, mli_min, mlo_scale, shuffle_buffer, batch_size
    )

    print(f'[VAL] Total # of input files for validation (10% of training): {len(f_mli_val)}')

    # Estimate total samples
    # total_training_samples = estimate_total_samples(f_mli_train, vars_mli)
    # total_validation_samples = estimate_total_samples(f_mli_val, vars_mli)

    # steps_per_epoch = math.ceil(total_training_samples / batch_size)
    # validation_steps = math.ceil(total_validation_samples / batch_size)

    # Adjust `steps_per_epoch` based on the actual dataset size
    

    # Calculate dataset size
    total_training_samples = calculate_dataset_size(f_mli_train, vars_mli)
    steps_per_epoch = math.ceil(total_training_samples / batch_size)
    print(f"Steps per epoch: {steps_per_epoch}")

    total_validation_samples = calculate_dataset_size(f_mli_val, vars_mli)
    validation_steps = math.ceil(total_validation_samples / batch_size)
    print(f"Validation steps: {validation_steps}")


    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Define model
    model = build_model()

    # Set up optimizer
    optimizer = keras.optimizers.Adam()

    # Ensure Log directory exists
    os.makedirs(output_results_dirpath, exist_ok=True)

    filepath_csv = f'{output_results_dirpath}/csv_logger_lambda_{LAMBDA_ENERGY}_datafrac_{data_subset_fraction}.csv'
    with open(filepath_csv, "w") as f:
        # headers
        f.write("epoch,train_loss,train_mae,train_mse,train_energy_loss,"
                "val_loss,val_mae,val_mse,val_energy_loss\n")

    # Train model using custom training loop
    train_model(
    model, 
    tds, 
    tds_val, 
    LAMBDA_ENERGY, 
    optimizer, 
    N_EPOCHS, 
    steps_per_epoch, 
    validation_steps, 
    filepath_csv, 
    output_results_dirpath,
    data_subset_fraction
    )


    print("DONE")


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
            # Read mli
            ds = xr.open_dataset(file, engine='netcdf4')
            ds = ds[vars_mli]

            # Read mlo
            dso = xr.open_dataset(
                file.replace('.mli.', '.mlo.'), engine='netcdf4'
            )

            # Create mlo variables: ptend_t and ptend_q0001
            dso['ptend_t'] = (
                dso['state_t'] - ds['state_t']
            ) / 1200  # T tendency [K/s]
            dso['ptend_q0001'] = (
                dso['state_q0001'] - ds['state_q0001']
            ) / 1200  # Q tendency [kg/kg/s]
            dso = dso[vars_mlo]

            # Normalization, scaling
            ds_norm = (ds - mli_mean) / (mli_max - mli_min)
            dso_scaled = dso * mlo_scale

            # Stack
            ds_stack = ds_norm.stack({'batch': {'ncol'}})
            ds_stack = ds_stack.to_stacked_array(
                "mlvar", sample_dims=["batch"], name='mli'
            )
            dso_stack = dso_scaled.stack({'batch': {'ncol'}})
            dso_stack = dso_stack.to_stacked_array(
                "mlvar", sample_dims=["batch"], name='mlo'
            )
            # print(f"Shape of ds.values: {ds_stack.values.shape}")

            yield (ds_stack.values, dso_stack.values) #how muh is this yeilding?

    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, 125), (None, 128))
    )

def estimate_total_samples(filelist, variables, sample_size=10):
    sampled_files = random.sample(filelist, min(sample_size, len(filelist)))
    total_samples_sampled = 0
    for file in sampled_files:
        ds = xr.open_dataset(file, engine='netcdf4')
        total_samples_sampled += ds[variables[0]].shape[0]
    avg_samples_per_file = total_samples_sampled / len(sampled_files)
    estimated_total_samples = avg_samples_per_file * len(filelist)
    return int(estimated_total_samples)

def build_model():
    input_length = 2 * 60 + 5
    output_length_lin = 2 * 60
    output_length_relu = 8
    output_length = output_length_lin + output_length_relu
    n_nodes = 512

    input_layer = keras.layers.Input(shape=(input_length,), name='input')
    hidden_0 = keras.layers.Dense(n_nodes, activation='relu')(input_layer)
    hidden_1 = keras.layers.Dense(n_nodes, activation='relu')(hidden_0)
    output_pre = keras.layers.Dense(output_length, activation='elu')(hidden_1)
    output_lin = keras.layers.Dense(
        output_length_lin, activation='linear'
    )(output_pre)
    output_relu = keras.layers.Dense(
        output_length_relu, activation='relu'
    )(output_pre)
    output_layer = keras.layers.Concatenate()([output_lin, output_relu])

    model = keras.Model(input_layer, output_layer, name='Emulator')
    model.summary()
    return model

def define_callbacks(output_results_dirpath, lambda_energy, data_subset_fraction):
    # Create directory for logs specific to lambda_energy and data_subset_fraction
    log_dir = f'{output_results_dirpath}/logs_lambda_{lambda_energy}_datafrac_{data_subset_fraction}'
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard callback
    tboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
    )

    # Checkpoint callback - Save best model for the specific combination
    filepath_checkpoint = f'{output_results_dirpath}/best_model_lambda_{lambda_energy}_datafrac_{data_subset_fraction}.keras'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=filepath_checkpoint,
        save_weights_only=False,
        monitor='val_mae',  # Monitor validation MAE
        mode='min',  # Save model with minimum MAE
        save_best_only=True
    )

    # CSV logger callback - Save logs specific to this combination
    filepath_csv = f'{output_results_dirpath}/csv_logger_lambda_{lambda_energy}_datafrac_{data_subset_fraction}.csv'
    csv_callback = keras.callbacks.CSVLogger(
        filepath_csv, separator=",", append=True
    )

    return [tboard_callback, checkpoint_callback, csv_callback]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument(
        '--lambda_energy',
        type=float,
        default=0.1,
        help='Value of LAMBDA_ENERGY'
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
    args = parser.parse_args()

    if not args.output_results_dirpath:
        args.output_results_dirpath = (
            f"/home/alvarovh/code/cse598_climate_proj/results/"
        )

    main(
        args.lambda_energy,
        args.output_results_dirpath,
        args.data_subset_fraction,
        args.n_epochs
    )

