#!/usr/bin/env python
# coding: utf-8

# # Multi-Layer Perceptron (MLP) Example

# In[34]:

N_EPOCHS = 1
shuffle_buffer = 12*384 #ncol=384
#batch_size= 96 # 384/4
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

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
    for kgpu in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[kgpu], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
tf.config.list_physical_devices('GPU')
print("done")

#exit
# ## Build data pipeline

# ### input and output variable list
# - Note that ptend_t and ptend_q0001 are not in the output (mlo) netcdf files, but calculated real-time on a tf Dataset object.
# - Variable list: https://docs.google.com/spreadsheets/d/1ljRfHq6QB36u0TuoxQXcV4_DSQUR0X4UimZ4QHR8f9M/edit#gid=0

# In[35]:


# in/out variable lists
vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']


# ## tf Dataset pipeline
# - ref: https://www.noahbrenowitz.com/post/loading_netcdfs/
# - ref: https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

# In[36]:

norm_path = "/home/alvarovh/code/cse598_climate_proj/ClimSim/preprocessing/normalizations/"
root_train_path = "/nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_lowres_twomonths/datasets--LEAP--ClimSim_low-res/snapshots/bab82a2ebdc750a0134ddcd0d5813867b92eed2a/train"

root_train_path = "/nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_all/datasets--LEAP--ClimSim_low-res/snapshots/bab82a2ebdc750a0134ddcd0d5813867b92eed2a/train/"

# In[37]:


mli_mean = xr.open_dataset(norm_path + 'inputs/input_mean.nc')
mli_max = xr.open_dataset(norm_path + 'inputs/input_max.nc')
mli_min = xr.open_dataset(norm_path + 'inputs/input_min.nc')
mlo_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc')


# In[38]:


def load_nc_dir_with_generator(filelist:list):
    def gen():
        for file in filelist:
            
            # read mli
            ds = xr.open_dataset(file, engine='netcdf4')
            #print(f'available keys: {ds.keys()} for file {file}')
            with open('keys.txt', 'a') as f:
                f.write(str(file)+'\n')
                for key in list(ds.keys()):
                    f.write(f'{key}\n')
                f.write('#'*50+'\n')
            ds = ds[vars_mli]
            
            # read mlo
            dso = xr.open_dataset(file.replace('.mli.','.mlo.'), engine='netcdf4')
            
            # make mlo variales: ptend_t and ptend_q0001
            dso['ptend_t'] = (dso['state_t'] - ds['state_t'])/1200 # T tendency [K/s]
            dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001'])/1200 # Q tendency [kg/kg/s]
            dso = dso[vars_mlo]
            
            # normalizatoin, scaling
            ds = (ds-mli_mean)/(mli_max-mli_min)
            dso = dso*mlo_scale

            # stack
            #ds = ds.stack({'batch':{'sample','ncol'}})
            ds = ds.stack({'batch':{'ncol'}})
            ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
            #dso = dso.stack({'batch':{'sample','ncol'}})
            dso = dso.stack({'batch':{'ncol'}})
            dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')
            
            yield (ds.values, dso.values)

    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float64, tf.float64),
        output_shapes=((None,124),(None,128))
    )


# ## Instantiate tf.data.Dataset object here
# - Dataset file size and dimensions: https://docs.google.com/document/d/1HgfZZJM0SygjWvSAJ5kSfql9aXUFkvLybL36p-vmdZc/edit

# In[39]:


import random
import glob

# Parameters for debugging
train_file_limit = int(420480 * 0.1) #2000000  # Number of files to use for training
val_proportion = 0.1    # Percentage of training data used for validation

# Training files from year 0001, month 02
#f_mli = glob.glob(root_train_path + '/0002-01/E3SM-MMF.mli.0002-01-*.nc')
print("BEFORE GLOB")
#f_mli = glob.glob(root_train_path + '/*/*.nc')

# Check if the file list already exists
try:
    with open("file_list.pkl", "rb") as f:
        f_mli = pickle.load(f)
        print("Loaded file list from file_list.pkl.")
except FileNotFoundError:
    # If it doesn't exist, generate and save it
    f_mli = glob.glob(root_train_path + '/*/*MMF.mli*.nc')
    with open("file_list.pkl", "wb") as f:
        pickle.dump(f_mli, f)
        print("File list generated and saved to file_list.pkl.")

print("#"*50)
print(f'f_mli initial length: {len(f_mli)}')

random.shuffle(f_mli)

# Apply the limit to the training files
f_mli = f_mli[:train_file_limit]
print(f'[TRAIN] Total # of input files after applying limit: {len(f_mli)}')

# Load and prepare the training dataset
tds = load_nc_dir_with_generator(f_mli)
tds = tds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
tds = tds.prefetch(buffer_size=4)

# Validation files from year 0002, month 02
# Using 10% of the training files as validation
f_mli_val = f_mli[:int(train_file_limit * val_proportion)]
print(f'[VAL] Total # of input files for validation (10% of training): {len(f_mli_val)}')

# Load and prepare the validation dataset
tds_val = load_nc_dir_with_generator(f_mli_val)
tds_val = tds_val.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
tds_val = tds_val.prefetch(buffer_size=4)


# In[40]:


# shuffle_buffer=384*12

# # for training

# # # First 5 days of each month for the first 6 years
# # f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.000[123456]-*-0[12345]-*.nc')
# # f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0007-01-0[12345]-*.nc')
# # f_mli = [*f_mli1, *f_mli2]

# # every 10th sample
# f_mli1 = glob.glob(root_train_path + '/*/E3SM-MMF.mli.000[123456]-*-*-*.nc')
# f_mli2 = glob.glob(root_train_path + '/*/E3SM-MMF.mli.0007-01-*-*.nc')
# f_mli = sorted([*f_mli1, *f_mli2])
# random.shuffle(f_mli)
# f_mli = f_mli[::10]

# # # debugging
# # f_mli = f_mli[0:72*5]

# random.shuffle(f_mli)
# print(f'[TRAIN] Total # of input files: {len(f_mli)}')
# print(f'[TRAIN] Total # of columns (nfiles * ncols): {len(f_mli)*384}')
# tds = load_nc_dir_with_generator(f_mli)
# tds = tds.unbatch()
# tds = tds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
# tds = tds.prefetch(buffer_size=4) # in realtion to the batch size

# # for validation

# # # First 5 days of each month for the following 2 years
# # f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0007-0[23456789]-0[12345]-*.nc')
# # f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0007-1[012]-0[12345]-*.nc')
# # f_mli3 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.000[89]-*-0[12345]-*.nc')
# # f_mli_val = [*f_mli1, *f_mli2, *f_mli3]

# # every 10th sample
# f_mli1 = glob.glob(root_train_path + '/*/E3SM-MMF.mli.0007-0[23456789]-0[12345]-*.nc')
# f_mli2 = glob.glob(root_train_path + '/*/E3SM-MMF.mli.0007-1[012]-0[12345]-*.nc')
# f_mli3 = glob.glob(root_train_path + '/*/E3SM-MMF.mli.000[89]-*-0[12345]-*.nc')
# f_mli_val = sorted([*f_mli1, *f_mli2, *f_mli3])
# f_mli_val = f_mli_val[::10]

# # # debugging
# # f_mli_val = f_mli_val[0:72*5]

# random.shuffle(f_mli_val)
# print(f'[VAL] Total # of input files: {len(f_mli_val)}')
# print(f'[VAL] Total # of columns (nfiles * ncols): {len(f_mli_val)*384}')
# tds_val = load_nc_dir_with_generator(f_mli_val)
# tds_val = tds_val.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
# tds_val = tds_val.prefetch(buffer_size=4) # in realtion to the batch size

# #list(tds)
# # for count_batch in tds.repeat().batch(10).take(1):
# #     print(count_batch[0].numpy())
# #count_batch[0].shape


# ## ML training
# - While 4 GPUs are available on the node, using multi GPUs (with 'tf.distribute.MirroredStrategy()' strategy) does not speed up training process. It is possibly due to that the current Dataset pipeline is sequential.

# In[41]:


tf.config.list_physical_devices('GPU')


# In[42]:


# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():

# model params
input_length = 2*60 + 4
output_length_lin  = 2*60
output_length_relu = 8
output_length = output_length_lin + output_length_relu
n_nodes = 512

# constrcut a model
input_layer    = keras.layers.Input(shape=(input_length,), name='input')
hidden_0       = keras.layers.Dense(n_nodes, activation='relu')(input_layer)
hidden_1       = keras.layers.Dense(n_nodes, activation='relu')(hidden_0)
output_pre     = keras.layers.Dense(output_length, activation='elu')(hidden_1)
output_lin     = keras.layers.Dense(output_length_lin,activation='linear')(output_pre)
output_relu    = keras.layers.Dense(output_length_relu,activation='relu')(output_pre)
output_layer   = keras.layers.Concatenate()([output_lin, output_relu])

model = keras.Model(input_layer, output_layer, name='Emulator')
model.summary()

# compile
model.compile(optimizer=keras.optimizers.Adam(), #optimizer=keras.optimizers.Adam(learning_rate=clr),
              loss='mse',
              metrics=['mse','mae','accuracy'])


# In[43]:


# callbacks
# a. tensorboard
tboard_callback = keras.callbacks.TensorBoard(log_dir = './logs_tensorboard',
                                              histogram_freq = 1,)

# b. checkpoint
filepath_checkpoint = 'saved_model/best_model_proto_larger.keras'
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=filepath_checkpoint,
                                                            save_weights_only=False,
                                                            monitor='val_mse',
                                                            mode='min',
                                                            save_best_only=True)

# c. csv logger
filepath_csv = 'csv_logger_larger.txt'
csv_callback = keras.callbacks.CSVLogger(filepath_csv, separator=",", append=True)

my_callbacks= [tboard_callback, checkpoint_callback, csv_callback]

# !mkdir logs_tensorboard
# !mkdir saved_model


# In[44]:


# Manually shuffling the order of input files.
# "tds = tds.shuffle(buffer_size=<global>, reshuffle_each_iteration=True)" is possible,
# however, it is slow.
# So employing global shuffle (by file names) + local shuffle (using .shuffle).

# COUNT STEPS

import xarray as xr


def estimate_total_samples(filelist, variables, sample_size=10):
    # Randomly sample a subset of files
    sampled_files = random.sample(filelist, sample_size)
    
    # Calculate the total number of samples from the sampled files
    total_samples_sampled = 0
    for file in sampled_files:
        ds = xr.open_dataset(file, engine='netcdf4')
        total_samples_sampled += ds[variables[0]].shape[0]
    
    # Calculate the average number of samples per file
    avg_samples_per_file = total_samples_sampled / sample_size
    
    # Estimate total number of samples in the entire dataset
    estimated_total_samples = avg_samples_per_file * len(filelist)
    
    return int(estimated_total_samples)


train_filelist = f_mli  # Training file list
val_filelist = f_mli_val  # Validation file list

total_training_samples = estimate_total_samples(train_filelist, vars_mli)
total_validation_samples = estimate_total_samples(val_filelist, vars_mli)


import math



steps_per_epoch = math.ceil(total_training_samples / batch_size)
validation_steps = math.ceil(total_validation_samples / batch_size)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")



###################





n=0
while n < N_EPOCHS:
    random.shuffle(f_mli)
    tds = load_nc_dir_with_generator(f_mli) # global shuffle by file names
    tds = tds.unbatch()
 # local shuffle by elements    tds = tds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=False)
    tds = tds.batch(batch_size)
    tds = tds.prefetch(buffer_size=int(shuffle_buffer/384)) # in realtion to the batch size

    random.shuffle(f_mli_val)
    tds_val = load_nc_dir_with_generator(f_mli_val)
    tds_val = tds_val.unbatch()
    tds_val = tds_val.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=False)
    tds_val = tds_val.batch(batch_size)
    tds_val = tds_val.prefetch(buffer_size=int(shuffle_buffer/384))
    
    print(f'Epoch: {n+1}')
    model.fit(tds, 
              validation_data=tds_val,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              callbacks=my_callbacks)
    
    n+=1


# In[45]:

print("DONE")
# pwd


# # In[46]:


# # Here's the modified code to load data from a CSV file by specifying the file path

# import pandas as pd
# import matplotlib.pyplot as plt

# # Replace 'your_file_path.csv' with the path to your CSV file
# file_path = '/home/alvarovh/csv_logger.txt'

# # Load the data from the specified file path
# df = pd.read_csv(file_path)

# # Plotting accuracy and validation accuracy
# plt.figure(figsize=(10, 5))
# plt.plot(df.index, df['accuracy'], label='Training Accuracy')
# plt.plot(df.index, df['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plotting loss and validation loss
# plt.figure(figsize=(10, 5))
# plt.plot(df.index, df['loss'], label='Training Loss')
# plt.plot(df.index, df['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()


