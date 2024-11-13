#!/usr/bin/env python
# coding: utf-8

N_EPOCHS = 1
shuffle_buffer = 12*384  # ncol=384
batch_size = 1024
import glob
import pickle

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import random

import tensorflow as tf
from tensorflow import keras

# Enable mixed precision for faster training on compatible GPUs
# Changed Line
tf.keras.mixed_precision.set_global_policy('mixed_float16')

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
    for kgpu in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[kgpu], True)
except:
    pass
tf.config.list_physical_devices('GPU')
print("done")

# in/out variable lists
vars_mli = ['state_t', 'state_q0001', 'state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t', 'ptend_q0001', 'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

norm_path = "/home/alvarovh/code/cse598_climate_proj/ClimSim/preprocessing/normalizations/"
root_train_path = "/nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_all/datasets--LEAP--ClimSim_low-res/snapshots/bab82a2ebdc750a0134ddcd0d5813867b92eed2a/train/"

mli_mean = xr.open_dataset(norm_path + 'inputs/input_mean.nc')
mli_max = xr.open_dataset(norm_path + 'inputs/input_max.nc')
mli_min = xr.open_dataset(norm_path + 'inputs/input_min.nc')
mlo_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc')


# Changed Function: Add parallel processing and caching to speed up loading
def load_nc_dir_with_generator(filelist: list):
    def gen():
        for file in filelist:
            ds = xr.open_dataset(file, engine='netcdf4')
            ds = ds[vars_mli]
            
            dso = xr.open_dataset(file.replace('.mli.', '.mlo.'), engine='netcdf4')
            dso['ptend_t'] = (dso['state_t'] - ds['state_t']) / 1200
            dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001']) / 1200
            dso = dso[vars_mlo]

            ds = (ds - mli_mean) / (mli_max - mli_min)
            dso = dso * mlo_scale

            ds = ds.stack({'batch': {'ncol'}})
            ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
            dso = dso.stack({'batch': {'ncol'}})
            dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')

            yield (ds.values, dso.values)

    # Add .cache(), .prefetch(), and parallel processing
    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float16, tf.float16),  # Changed from tf.float64 to tf.float16
        output_shapes=((None, 124), (None, 128))
    ).cache().prefetch(tf.data.AUTOTUNE)


import random

train_file_limit = int(420480 * 0.1)
val_proportion = 0.1

try:
    with open("file_list.pkl", "rb") as f:
        f_mli = pickle.load(f)
        print("Loaded file list from file_list.pkl.")
except FileNotFoundError:
    f_mli = glob.glob(root_train_path + '/*/*MMF.mli*.nc')
    with open("file_list.pkl", "wb") as f:
        pickle.dump(f_mli, f)
        print("File list generated and saved to file_list.pkl.")

print(f'f_mli initial length: {len(f_mli)}')

random.shuffle(f_mli)

f_mli = f_mli[:train_file_limit]
print(f'[TRAIN] Total # of input files after applying limit: {len(f_mli)}')

# Use AUTOTUNE and increased prefetch buffer size
# Changed Line
tds = load_nc_dir_with_generator(f_mli).shuffle(buffer_size=shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)

f_mli_val = f_mli[:int(train_file_limit * val_proportion)]
print(f'[VAL] Total # of input files for validation (10% of training): {len(f_mli_val)}')

# Changed Line: Increased prefetch buffer and added AUTOTUNE
tds_val = load_nc_dir_with_generator(f_mli_val).shuffle(buffer_size=shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)


input_length = 2*60 + 4
output_length_lin = 2*60
output_length_relu = 8
output_length = output_length_lin + output_length_relu
n_nodes = 512

input_layer = keras.layers.Input(shape=(input_length,), name='input')
hidden_0 = keras.layers.Dense(n_nodes, activation='relu')(input_layer)
hidden_1 = keras.layers.Dense(n_nodes, activation='relu')(hidden_0)
output_pre = keras.layers.Dense(output_length, activation='elu')(hidden_1)
output_lin = keras.layers.Dense(output_length_lin, activation='linear')(output_pre)
output_relu = keras.layers.Dense(output_length_relu, activation='relu')(output_pre)
output_layer = keras.layers.Concatenate()([output_lin, output_relu])

model = keras.Model(input_layer, output_layer, name='Emulator')
model.summary()

# compile
# Changed Line: Set mixed precision optimizer
model.compile(
    optimizer=tf.keras.mixed_precision.LossScaleOptimizer(keras.optimizers.Adam()),
    loss='mse',
    metrics=['mse', 'mae', 'accuracy']
)


tboard_callback = keras.callbacks.TensorBoard(log_dir='./logs_tensorboard', histogram_freq=1)
filepath_checkpoint = 'saved_model/best_model_proto_larger_faster.keras'
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=filepath_checkpoint,
    save_weights_only=False,
    monitor='val_mse',
    mode='min',
    save_best_only=True
)
filepath_csv = 'csv_logger_larger_faster.txt'
csv_callback = keras.callbacks.CSVLogger(filepath_csv, separator=",", append=True)
my_callbacks = [tboard_callback, checkpoint_callback, csv_callback]


import xarray as xr


def estimate_total_samples(filelist, variables, sample_size=10):
    sampled_files = random.sample(filelist, sample_size)
    total_samples_sampled = 0
    for file in sampled_files:
        ds = xr.open_dataset(file, engine='netcdf4')
        total_samples_sampled += ds[variables[0]].shape[0]
    avg_samples_per_file = total_samples_sampled / sample_size
    estimated_total_samples = avg_samples_per_file * len(filelist)
    return int(estimated_total_samples)


train_filelist = f_mli
val_filelist = f_mli_val

total_training_samples = estimate_total_samples(train_filelist, vars_mli)
total_validation_samples = estimate_total_samples(val_filelist, vars_mli)

import math

steps_per_epoch = math.ceil(total_training_samples / batch_size)
validation_steps = math.ceil(total_validation_samples / batch_size)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

n = 0
while n < N_EPOCHS:
    random.shuffle(f_mli)
    print(f'Epoch: {n+1}')
    model.fit(
        tds,
        validation_data=tds_val,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=my_callbacks
    )
    n += 1

