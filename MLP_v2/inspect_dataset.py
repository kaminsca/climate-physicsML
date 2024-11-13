import xarray as xr
import glob, os
import random
from energy_loss import total_loss
import tensorflow as tf
import logging

# Configure logging
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLIMSIM_PATH = '/home/clarkkam/AIFS/ClimSim/'
DATA_PATH = '/home/clarkkam/AIFS/ClimSim/climate-physicsML/data/climsim_lowres_0001-02/datasets--LEAP--ClimSim_low-res/snapshots/bab82a2ebdc750a0134ddcd0d5813867b92eed2a/train/0001-02/'

# in/out variable lists
vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']
vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX', 'cam_in_LWUP'] 

mli_mean = xr.open_dataset(CLIMSIM_PATH + '/preprocessing/normalizations/inputs/input_mean.nc')
mli_min = xr.open_dataset(CLIMSIM_PATH + '/preprocessing/normalizations/inputs/input_min.nc')
mli_max = xr.open_dataset(CLIMSIM_PATH + '/preprocessing/normalizations/inputs/input_max.nc')
mlo_scale = xr.open_dataset(CLIMSIM_PATH + '/preprocessing/normalizations/outputs/output_scale.nc')

def load_nc_dir_with_generator(filelist:list):
    def gen():
        # lets just look at one file for now
        files = [filelist[0]]
        # switch to this to use all data
        # files = filelist
        for file in files:
            # read mli
            ds = xr.open_dataset(file, engine='netcdf4')
            inp_vars = list(ds.keys())
            unused_inp_vars = [x for x in inp_vars if x not in vars_mli]
            logger.debug("USED INPUT VARIABLES %s", vars_mli)
            logger.debug("UNUSED INPUT VARIABLES %s", unused_inp_vars)
            ds = ds[vars_mli]
            logger.debug(f"Read mli from {file}. Dimensions: {ds.dims}")
            # ds.info()
            
            # read mlo
            dso = xr.open_dataset(file.replace('.mli.','.mlo.'), engine='netcdf4')
            out_vars = list(dso.keys())
            unused_out_vars = [x for x in out_vars if x not in vars_mlo]
            logger.debug("USED OUTPUT VARIABLES %s", vars_mlo)
            logger.debug("UNUSED OUTPUT VARIABLES %s", unused_out_vars)
            # dso.info()
            
            # make mlo variales: ptend_t and ptend_q0001
            dso['ptend_t'] = (dso['state_t'] - ds['state_t'])/1200 # T tendency [K/s]
            dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001'])/1200 # Q tendency [kg/kg/s]
            dso = dso[vars_mlo]
            logger.debug(f"Read mlo from {file.replace('.mli.', '.mlo.')}. Dimensions: {dso.dims}")
            # dso.info()

            # normalizatoin, scaling
            ds = (ds-mli_mean)/(mli_max-mli_min)
            dso = dso*mlo_scale
            logger.debug("Normalized and scaled datasets.")
            # logger.debug(f"mli dimensions: {ds.dims}, mlo dimensions: {dso.dims}")

            # stack
            #ds = ds.stack({'batch':{'sample','ncol'}})
            ds = ds.stack({'batch':{'ncol'}})
            # logger.debug(f"Stacked mli dataset. Dimensions: {ds.dims}")
            ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
            logger.debug(f"Converted mli dataset to stacked array. Shape: {ds.shape}")
            logger.debug(f"0-59: state_t, 60-119: state_q0001, 120: state_ps, 121: pbuf_SOLIN, 122: pbuf_LHFLX, 123: pbuf_SHFLX, 124: cam_in_LWUP")

            dso = dso.stack({'batch': {'ncol'}})
            # logger.debug(f"Stacked mlo dataset. Dimensions: {dso.dims}")
            dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')
            logger.debug(f"Converted mlo dataset to stacked array. Shape: {dso.shape}")
            logger.debug(f"0-59: ptend_t, 60-119: ptend_q0001, 120: cam_out_NETSW, 121: cam_out_FLWDS, 122: cam_out_PRECSC, 123: cam_out_PRECC, 124: cam_out_SOLS, 125: cam_out_SOLL, 126: cam_out_SOLSD, 127: cam_out_SOLLD")


            logger.debug("Shape of ds.values: %s", ds.values.shape)
            logger.debug("Shape of dso.values: %s", dso.values.shape)
            # print("Example of ds.values:", ds.values[:1])
            # print("Example of dso.values:", dso.values[:1])
            
            yield (ds.values, dso.values)

    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float64, tf.float64),
        output_shapes=((None,125),(None,128))
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


def manual_count(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count
print(f"Length of the dataset: {manual_count(tds)}")

iterator = iter(tds)
# curious about cam_in_LWUP from input and cam_out_SOLS / SOLSD / etc from output:
x, y = next(iterator)
print(x[124])
print(y[124])
print(y[124:128])
count_non_zero = 0
try:
    while True:
        x, y = next(iterator)
        # Check if any element in y[124:128] is not 0
        if tf.reduce_any(y[124:128] != 0).numpy():
            count_non_zero += 1
            # print("y[124:128]:", y[124:128])
except StopIteration:
    pass
print(count_non_zero, 'non zero values in cam_out_SOLS / SOLSD / ...')

# test energy loss
iterator = iter(tds)
counter = 0
variable_contributions = {
    'r_out': 0,
    'lh': 0,
    'sh': 0,
    'r_in' : 0,
    'SOLS' : 0,
    'SOLL' : 0,
    'SOLSD' : 0,
    'SOLLD' : 0
}
try:
    while True:
        x, y = next(iterator)
        # r_out = upward longwave flux 
        # Surface latent heat flux (LH), and Surface sensible heat flux (SH)
        r_out = x[124]
        lh = x[122]
        sh = x[123]
        # r_in = visible direct flux + near-IR direct flux + visible diffuse flux + near-IR diffuse flux
        r_in = y[124] + y[125] + y[126] + y[127]

        variable_contributions['r_out'] += abs(r_out)
        variable_contributions['lh'] += abs(lh)
        variable_contributions['sh'] += abs(sh)
        variable_contributions['r_in'] += abs(r_in)
        variable_contributions['SOLS'] += abs(y[124])
        variable_contributions['SOLL'] += abs(y[125])
        variable_contributions['SOLSD'] += abs(y[126])
        variable_contributions['SOLLD'] += abs(y[127])

        # Lec = r_in − (r_out + lh + sh)
        loss_ec = r_in - (r_out + lh + sh)
        # print(f"EC Loss[{counter}]: {loss_ec}")
        # print(f"        r_in = {r_in}, r_out = {r_out}, lh = {lh}, sh = {sh}")
        counter += 1
except StopIteration:
    pass

average_contributions = {key: value / counter for key, value in variable_contributions.items()}
print("Average contributions to loss:")
for var, contrib in average_contributions.items():
    print(f"{var}: {contrib}")