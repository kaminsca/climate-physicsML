import os, glob, pickle
import pandas as pd
import numpy as np
import xarray as xr
import argparse
import tensorflow as tf
import random
import re
import shutil
from tqdm import tqdm
from config import climsim_downloaded_data_dirpath, norm_path, vars_mli, vars_mlo_0, vars_mlo, train_subset_dirpath, validation_subset_dirpath, test_subset_dirpath, data_fraction

# set seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def from_input_normalized_to_original(normalized, var, mli_mean, mli_max, mli_min, input_var_norm_epsilon=1e-5):
    '''
    Function to convert normalized input variables back to original scale
    
    Args:
    normalized: tf.Tensor, normalized input variable
    var: str, variable name
    input_var_norm_epsilon: float, epsilon value for normalization
    
    Returns:
    original: tf.Tensor, original scale input variable
        
    '''

    try:
        original = normalized * (mli_max[var] - mli_min[var] + input_var_norm_epsilon) + mli_mean[var]
    except Exception as e:
        print(f"ERROR Failed to get key {var}")
        with open("failed_keys.txt", "a") as f:
            f.write(f"Failed to get key {var}\n")
        
        raise e
        

    return original


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
            # read mli (-> ds)
            with xr.open_dataset(file, engine='netcdf4') as ds:
                with xr.open_dataset(file.replace('.mli.','.mlo.'), engine='netcdf4') as dso:
                    ds = ds[vars_mli]

                    # make mlo tendency variables ("ptend_xxxx"):
                    state_t_dso = dso['state_t']
                    state_q0001_dso = dso['state_q0001']
                    
                    for kvar in ['state_t','state_q0001','state_q0002', 'state_q0003', 'state_u', 'state_v']:
                        dso[kvar.replace('state','ptend')] = (dso[kvar] - ds[kvar])/1200 # timestep=1200[sec]
                    
                    # normalization, scaling
                    input_var_norm_epsilon = 1e-5
                    ds = (ds - mli_mean) / (mli_max - mli_min + input_var_norm_epsilon)
                    
                    # print if this was indifined:
                    dso = dso * mlo_scale

                    # get index 59 (surface) for variables that have more than 1 level
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

                    dso = dso[vars_mlo]

                    for var in vars_mli:
                        # print(var)
                        if len(ds[var].shape) == 2:
                            ds[var] = ds[var][index]
                            # print("changed")
                    ds=ds[vars_mli]

                    # flatten input variables to one array per grid location
                    ds = ds.stack({'batch':{'ncol'}})
                    ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
                    dso = dso.stack({'batch':{'ncol'}})
                    dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')

                    denominator = (mli_max - mli_min + input_var_norm_epsilon)
                    for var in vars_mli:
                        if np.any(denominator[var] == 0):
                            print(f"Zero range detected in variable {var}")


                    yield (ds.values, dso.values)

    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, len(vars_mli)), (None, len(vars_mlo)))
    )

def calculate_dataset_size(file_list, vars_mli):
    # Load one file to determine sample size
    try:
        example_file = file_list[0]
        with xr.open_dataset(example_file, engine='netcdf4') as ds:
            samples_per_file = ds[vars_mli[0]].shape[0]
        print(f"Samples per file: {samples_per_file}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return 0

    # Total dataset size
    total_samples = samples_per_file * len(file_list)
    print(f"Total number of samples in dataset: {total_samples}")
    return total_samples

def load_file_list(path):
    if os.path.exists("file_list.pkl"):
        with open("file_list.pkl", "rb") as f:
            all_files = pickle.load(f)
            print("Loaded file list from file_list.pkl.")
    else:
        all_files = glob.glob(os.path.join(path, '*/*MMF.mli*.nc'))
        with open("file_list.pkl", "wb") as f:
            pickle.dump(all_files, f)
            print("File list generated and saved to file_list.pkl.")
    return all_files

def prepare_validation_files(data_subset_fraction=1.0):
    """
    Prepare validation files: every 10th sample for the first 5 days of each month 
    for the following 2 years after the training set.
    """

    if os.path.exists(validation_subset_dirpath):
        # data fraction as string of 1 decimal
        validation_files = glob.glob(f"{validation_subset_dirpath}/*.mli.*.nc")
        print(f"Loaded validation files from {validation_subset_dirpath}")
    else:
        print(f"Validation files not found in {validation_subset_dirpath}. Generating new list...")

        file_list = load_file_list(climsim_downloaded_data_dirpath)
        
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
        print(f'[VAL] Total # of input files: {len(validation_files)}')
        
        # Shuffle and select every 10th file
        random.shuffle(validation_files)  # Global shuffle
        validation_files = validation_files[::10]  # Select every 10th file

        # subsample the data
        validation_files = validation_files[:int(len(validation_files) * data_subset_fraction)]

        # save the validation files
        os.makedirs(validation_subset_dirpath, exist_ok=True)
        for file in tqdm(validation_files):
            # copy the file structure to the validation_subset_dirpath
            shutil.copy(file, validation_subset_dirpath)
            # shutil.copy(file, validation_subset_dirpath)
            # now the .mlo. files
            shutil.copy(file.replace('.mli.','.mlo.'), validation_subset_dirpath)

        print(f"Validation files saved to {validation_subset_dirpath}")


    print(f'[VAL] Total # of input files AFTER SUBSET OF {data_subset_fraction}: {len(validation_files)}')
    return validation_files

def prepare_training_files(data_subset_fraction=1.0):
    """
    Prepare training files: every 10th sample for the first 5 days of each month 
    for the first 6 years, plus January of year 0007.
    """

    if os.path.exists(train_subset_dirpath):
        training_files = glob.glob(train_subset_dirpath + '/*.mli.*.nc')
        print(f"Loaded training files from {train_subset_dirpath}")
    else:
        file_list = load_file_list(climsim_downloaded_data_dirpath)

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

        # subsample the data
        training_files = training_files[:int(len(training_files) * data_subset_fraction)]

        # save the training files
        os.makedirs(train_subset_dirpath, exist_ok=True)

        for file in tqdm(training_files):
            # copy the file structure to the train_subset_dirpath
            shutil.copy(file, train_subset_dirpath)
            # now the .mlo. files
            shutil.copy(file.replace('.mli.','.mlo.'), train_subset_dirpath)
        print(f"Training files saved to {train_subset_dirpath}")

    print(f'[TRAIN] Total # of input files AFTER SUBSET OF {data_subset_fraction}: {len(training_files)}')
    return training_files


def prepare_test_files(data_subset_fraction=1.0):
    """
    Prepare test files: every first day of each month of the 8th year.
    """

    if os.path.exists(test_subset_dirpath):
        test_files = glob.glob(test_subset_dirpath + '/*.mli.*.nc')
        print(f"Loaded test files from {test_subset_dirpath}")
    else:
        print(f"Test files not found in {test_subset_dirpath}. Generating new list...")

        file_list = load_file_list(climsim_downloaded_data_dirpath)
        
        # regex patterns for matching desired files
        pattern = re.compile(r'E3SM-MMF\.mli\.0008-.*-01-00000\.nc')
        
        # Filter files matching the patterns
        f_mli = [f for f in file_list if pattern.search(f)]
        
        # Combine and sort test files
        test_files = sorted(f_mli)
        print(f'[TEST] Total # of input files: {len(test_files)}')

        # subsample the data
        test_files = test_files[:int(len(test_files) * data_subset_fraction)]

        # save the test files
        os.makedirs(test_subset_dirpath, exist_ok=True)
        for file in tqdm(test_files):
            # copy the file structure to the test_subset_dirpath
            shutil.copy(file, test_subset_dirpath)
            # now the .mlo. files
            shutil.copy(file.replace('.mli.','.mlo.'), test_subset_dirpath)

        print(f"Test files saved to {test_subset_dirpath}")

    print(f'[TEST] Total # of input files AFTER SUBSET OF {data_subset_fraction}: {len(test_files)}')
    return test_files

def load_normalization_data(norm_path):
    # # Load normalization datasets
    mli_mean = xr.open_dataset(norm_path + 'inputs/input_mean.nc')
    mli_max = xr.open_dataset(norm_path + 'inputs/input_max.nc')
    mli_min = xr.open_dataset(norm_path + 'inputs/input_min.nc')
    mlo_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc')

    # get only the 59 index (surface level) of the variables that have more than 1 level
    mli_mean = mli_mean.isel(lev=59)
    mli_max = mli_max.isel(lev=59)
    mli_min = mli_min.isel(lev=59)
    mlo_scale = mlo_scale.isel(lev=59)

    return mli_mean, mli_max, mli_min, mlo_scale
def main():

    # Prepare training and validation files
    training_files = prepare_training_files(data_fraction)
    validation_files = prepare_validation_files(data_fraction)
    test_files = prepare_test_files(data_subset_fraction=1.0) # always use the full test set

    # Print the number of files in each set
    print(f"Training files: {len(training_files)}")
    print(f"Validation files: {len(validation_files)}")
    print(f"Test files: {len(test_files)}")

if __name__ == "__main__":
    main()