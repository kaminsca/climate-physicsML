data_fraction = 0.01

vars_mli = ['cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_LWUP', 'cam_in_OCNFRAC', 'cam_in_SNOWHICE', 'cam_in_SNOWHLAND', 'pbuf_COSZRS', 'pbuf_LHFLX', 'pbuf_SHFLX', 'pbuf_SOLIN', 'pbuf_TAUX', 'pbuf_TAUY', 'state_pmid', 'state_ps', 'state_q0001', 'state_q0002', 'state_q0003', 'state_t', 'state_u', 'state_v', 'pbuf_CH4', 'pbuf_N2O', 'pbuf_ozone']
# vars_mlo_0 corresponds with the scaling factor mlo_scale
vars_mlo_0 = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003', 'ptend_u', 'ptend_v',
                    'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 
                    'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

vars_mlo = vars_mlo_0 + ['state_t', 'state_q0001']

# Path to the ClimSim directory (commit at the time of writing: 671db93cac4df30715628e4976f9b9f17a9b4ec6)
root_climsim_dirpath = "/home/alvarovh/code/cse598_climate_proj/ClimSim/"
root_huggingface_data_dirpath = "/nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_all/"

# The Path to the Subset of the Dataset Used for Training:
train_subset_dirpath = f"/nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_subset/datafraction_{data_fraction:.2f}/train/"
validation_subset_dirpath = f"/nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_subset/datafraction_{data_fraction:.2f}/validation/"
test_subset_dirpath = f"/nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_subset/datafraction_{data_fraction:.2f}/test/"

# Alternatively, Path to the Full Downloaded Dataset from Huggingface
climsim_downloaded_data_dirpath = (
    f"{root_huggingface_data_dirpath}/datasets--LEAP--ClimSim_low-res/snapshots/"
    "bab82a2ebdc750a0134ddcd0d5813867b92eed2a/train/"
)

norm_path = f"{root_climsim_dirpath}/preprocessing/normalizations/"
grid_path = f"{root_climsim_dirpath}/grid_info/ClimSim_low-res_grid-info.nc"
