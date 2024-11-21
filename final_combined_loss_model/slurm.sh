#!/bin/bash
#SBATCH --job-name=climsim
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --account=cse598s002f24_class
#SBATCH --partition=spgpu
#SBATCH --time=00-08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=32GB
#cse598s002f24_class
# Start time
start=$(date +%s)

# -------------------------

# Initialize Conda

eval "$(conda shell.bash hook)"

conda activate climsim_env

cd /home/alvarovh/code/cse598_climate_proj/climate-physicsML/final_combined_loss_model

# bash run_experiment_through_lambdas.sh

# N Epochs:3
# 30% data fraction
# Lambda: From 0.0 to 0.2 every step of 0.01
bash experiment_nov19.sh 0.3 0.0 0.2 0.01 3 # <data_subset_fraction> <lower_lambda> <upper_lambda> <lambda_step> <epochs>
# N Epochs:3
# 100% data fraction
# Lambdas: 0.0, 0.03, 0.3, 0.6, 1 (edited) 
# <data_subset_fraction> <lower_lambda> <upper_lambda> <lambda_step> <epochs>
bash experiment_nov19.sh 1.0 0.0 0.0 0.03 3 # Lambda = 0.0
bash experiment_nov19.sh 1.0 0.03 0.03 0.03 3 # Lambda = 0.03
bash experiment_nov19.sh 1.0 0.3 0.3 0.03 3 # Lambda = 0.3
bash experiment_nov19.sh 1.0 0.6 0.6 0.03 3 # Lambda = 0.6
bash experiment_nov19.sh 1.0 1.0 1.0 0.03 3 # Lambda = 1.0

# -------------------------
# End time
end=$(date +%s)
# Runtime
runtime=$((end-start))
echo "Runtime: $runtime"


# python combined_loss_model.py --lambda_energy=1.0 --data_subset_fraction=0.001 --n_epochs=10
