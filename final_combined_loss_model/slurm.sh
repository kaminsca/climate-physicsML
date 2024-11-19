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

bash run_experiment_through_lambdas.sh

# -------------------------
# End time
end=$(date +%s)
# Runtime
runtime=$((end-start))
echo "Runtime: $runtime"


# python combined_loss_model.py --lambda_energy=1.0 --data_subset_fraction=0.001 --n_epochs=10
