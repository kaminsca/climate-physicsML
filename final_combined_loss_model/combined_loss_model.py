#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import random
import pickle
import os
import math

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

root_huggingface_data_dirpath = "/nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_all/"
root_climsim_dirpath = "/home/alvarovh/code/cse598_climate_proj/ClimSim/"

def main(LAMBDA_ENERGY, output_results_dirpath, data_subset_fraction, n_epochs):
    N_EPOCHS = n_epochs
    shuffle_buffer = 12 * 384  # ncol=384
    batch_size = 1024

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

    # Define custom loss functions
    def energy_loss(y_true, y_pred):
        r_out = y_pred[:, 124]
        lh = y_pred[:, 122]
        sh = y_pred[:, 123]
        r_in = tf.reduce_sum(y_true[:, 124:128], axis=1)
        loss_ec = r_in - (r_out + lh + sh)
        return tf.reduce_mean(tf.abs(loss_ec))

    def combined_loss(y_true, y_pred):
        mse = tf.keras.losses.MeanSquaredError()
        mse_loss = mse(y_true, y_pred)
        weighted_energy_loss = LAMBDA_ENERGY * energy_loss(y_true, y_pred)
        return mse_loss + weighted_energy_loss

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
        mli_max, mli_min, mlo_scale, shuffle_buffer
    )
    tds_val = create_dataset(
        f_mli_val, vars_mli, vars_mlo, mli_mean,
        mli_max, mli_min, mlo_scale, shuffle_buffer
    )

    print(f'[VAL] Total # of input files for validation (10% of training): {len(f_mli_val)}')

    # Estimate total samples
    total_training_samples = estimate_total_samples(f_mli_train, vars_mli)
    total_validation_samples = estimate_total_samples(f_mli_val, vars_mli)

    steps_per_epoch = math.ceil(total_training_samples / batch_size)
    validation_steps = math.ceil(total_validation_samples / batch_size)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Define model
    model = build_model()

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=combined_loss,
        metrics=['mse', 'mae', 'accuracy', energy_loss]
    )

    # Define callbacks
    callbacks = define_callbacks(output_results_dirpath)

    # Train the model
    n = 0
    while n < N_EPOCHS:
        random.shuffle(f_mli_train)
        tds = create_dataset(
            f_mli_train, vars_mli, vars_mlo, mli_mean,
            mli_max, mli_min, mlo_scale, shuffle_buffer,
            shuffle=False
        )
        random.shuffle(f_mli_val)
        tds_val = create_dataset(
            f_mli_val, vars_mli, vars_mlo, mli_mean,
            mli_max, mli_min, mlo_scale, shuffle_buffer,
            shuffle=False
        )

        print(f'Epoch: {n + 1}')
        model.fit(
            tds,
            validation_data=tds_val,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        n += 1

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
    shuffle=True
):
    ds = load_nc_dir_with_generator(
        file_list, vars_mli, vars_mlo, mli_mean,
        mli_max, mli_min, mlo_scale
    )
    ds = ds.unbatch()
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=False)
    batch_size = 1024
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

            yield (ds_stack.values, dso_stack.values)

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

def define_callbacks(output_results_dirpath):
    # TensorBoard callback
    tboard_callback = keras.callbacks.TensorBoard(
        log_dir=f'{output_results_dirpath}/logs_tensorboard',
        histogram_freq=1,
    )
    # Checkpoint callback
    filepath_checkpoint = f'{output_results_dirpath}/best_model_proto_larger.keras'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=filepath_checkpoint,
        save_weights_only=False,
        monitor='val_mse',
        mode='min',
        save_best_only=True
    )
    # CSV logger callback
    filepath_csv = f'{output_results_dirpath}/csv_logger_larger.txt'
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
            f"/home/alvarovh/code/cse598_climate_proj/results_{args.lambda_energy}/"
        )

    main(
        args.lambda_energy,
        args.output_results_dirpath,
        args.data_subset_fraction,
        args.n_epochs
    )

