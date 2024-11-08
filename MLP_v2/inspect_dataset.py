import xarray as xr
import glob, os
import random
from energy_loss import total_loss
import tensorflow as tf


CLIMSIM_PATH = '/home/clarkkam/AIFS/ClimSim/'
DATA_PATH = '/home/clarkkam/AIFS/ClimSim/climate-physicsML/data/climsim_lowres_0001-02/datasets--LEAP--ClimSim_low-res/snapshots/bab82a2ebdc750a0134ddcd0d5813867b92eed2a/train/0001-02/'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# in/out variable lists
vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

mli_mean = xr.open_dataset(CLIMSIM_PATH + '/preprocessing/normalizations/inputs/input_mean.nc')
mli_min = xr.open_dataset(CLIMSIM_PATH + '/preprocessing/normalizations/inputs/input_min.nc')
mli_max = xr.open_dataset(CLIMSIM_PATH + '/preprocessing/normalizations/inputs/input_max.nc')
mlo_scale = xr.open_dataset(CLIMSIM_PATH + '/preprocessing/normalizations/outputs/output_scale.nc')

def load_nc_dir_with_generator(filelist:list):
    def gen():
        for file in filelist:
            
            # read mli
            ds = xr.open_dataset(file, engine='netcdf4')
            ds = ds[vars_mli]
            print(f"Read mli from {file}. Dimensions: {ds.dims}")
            ds.info()
            
            # read mlo
            dso = xr.open_dataset(file.replace('.mli.','.mlo.'), engine='netcdf4')
            print(f"Read mlo from {file.replace('.mli.','.mlo.')}. Dimensions: {dso.dims}")
            dso.info()
            
            # make mlo variales: ptend_t and ptend_q0001
            dso['ptend_t'] = (dso['state_t'] - ds['state_t'])/1200 # T tendency [K/s]
            dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001'])/1200 # Q tendency [kg/kg/s]
            dso = dso[vars_mlo]
            print(f"Computed ptend_t and ptend_q0001. Dimensions: {dso.dims}")
            dso.info()

            # normalizatoin, scaling
            ds = (ds-mli_mean)/(mli_max-mli_min)
            dso = dso*mlo_scale
            print("Normalized and scaled datasets.")
            print(f"mli dimensions: {ds.dims}, mlo dimensions: {dso.dims}")

            # stack
            #ds = ds.stack({'batch':{'sample','ncol'}})
            ds = ds.stack({'batch':{'ncol'}})
            print(f"Stacked mli dataset. Dimensions: {ds.dims}")
            ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
            print(f"Converted mli dataset to stacked array. Shape: {ds.shape}")

            dso = dso.stack({'batch': {'ncol'}})
            print(f"Stacked mlo dataset. Dimensions: {dso.dims}")
            dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')
            print(f"Converted mlo dataset to stacked array. Shape: {dso.shape}")

            print("Shape of ds.values:", ds.values.shape)
            print("Shape of dso.values:", dso.values.shape)
            # print("Example of ds.values:", ds.values[:1])
            # print("Example of dso.values:", dso.values[:1])
            
            yield (ds.values, dso.values)

    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float64, tf.float64),
        output_shapes=((None,124),(None,128))
    )


shuffle_buffer=384*12
f_mli1 = glob.glob(DATA_PATH + 'E3SM-MMF.mli.0001-02-*-*.nc')
f_mli = sorted(f_mli1)
random.shuffle(f_mli)
f_mli = f_mli[::10]

random.shuffle(f_mli)
print(f'[TRAIN] Total # of input files: {len(f_mli)}')
print(f'[TRAIN] Total # of columns (nfiles * ncols): {len(f_mli)*384}')
tds = load_nc_dir_with_generator(f_mli)
tds = tds.unbatch()
tds = tds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
tds = tds.prefetch(buffer_size=4) # in realtion to the batch size

# iterator = iter(tds)
# # Inspect a single batch from the dataset
# x_batch, y_batch = next(iterator)
# # Print the shapes and some example data
# print("Shape of x_batch:", x_batch.shape)
# print("Shape of y_batch:", y_batch.shape)
# print("First example of x_batch:", x_batch[0])
# print("First example of y_batch:", y_batch[0])