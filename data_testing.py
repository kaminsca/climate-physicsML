# data from: https://huggingface.co/datasets/LEAP/subsampled_low_res/tree/main
import numpy as np
import pandas as pd
# train_input = np.load('./data/train_input.npy', mmap_mode='r')
# val_input = np.load('./data/val_input.npy', mmap_mode='r')
# val_target = np.load('./data/val_target.npy', mmap_mode='r')
# print(type(val_target))
# print("Shape:", val_target.shape)
# print(val_target[:5]) 

# data_input = pd.read_parquet('./data/val_input.parquet', engine='pyarrow')
# data_target = pd.read_parquet('./data/val_target.parquet', engine='pyarrow')
# scoring_target = pd.read_parquet('./data/scoring_target.parquet', engine='pyarrow')
# print(scoring_target.shape)
# print(scoring_target.head)
# print(data_target.columns.tolist())


# data from https://huggingface.co/datasets/LEAP/ClimSim_low-res/tree/main/train/0001-02
import netCDF4 as nc
file_path = './data/E3SM-MMF.mli.0001-02-01-00000.nc'
dataset = nc.Dataset(file_path, 'r')

print("Variables in dataset:")
for var_name in dataset.variables:
    print(var_name)
var_name = 'state_t'
data_var = dataset.variables[var_name]
print(data_var.shape)
print(data_var[:5])