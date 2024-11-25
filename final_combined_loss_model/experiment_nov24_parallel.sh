#!/bin/bash

# ================================
# Experiment Configuration
# ================================

# Set experiment parameters
nov24_exp_dirpath="/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov24/initiallambdasscaled"
data_fraction=0.01
n_epochs=5

# Define the list of experiments
declare -a experiments=(
  # Baseline with no custom losses
  "zero_model mass_radiation_nonneg mass_radiation_nonneg 1 1 1 zero_model.log"

  # Constant Lambdas all set to 1
  "constant_radiation_sf1 mass_nonneg_radiation mass_nonneg 1 1 1 constant_radiation_sf1.log"
  "constant_nonneg_sf1 mass_nonneg_radiation mass_radiation 1 1 1 constant_nonneg_sf1.log"
  "constant_mass_sf1 mass_nonneg_radiation radiation_nonneg 1 1 1 constant_mass_sf1.log"

  # Radiation
  "constant_radiation_sf0.05 mass_nonneg_radiation mass_nonneg 0.05 1 1 constant_radiation_sf0.05.log"
  "constant_radiation_sf0.25 mass_nonneg_radiation mass_nonneg 0.25 1 1 constant_radiation_sf0.25.log"
  "constant_radiation_sf0.5 mass_nonneg_radiation mass_nonneg 0.5 1 1 constant_radiation_sf0.5.log"

  # Mass
  "constant_mass_sf0.05 mass_nonneg_radiation radiation_nonneg 1 0.05 1 constant_mass_sf0.05.log"
  "constant_mass_sf0.25 mass_nonneg_radiation radiation_nonneg 1 0.25 1 constant_mass_sf0.25.log"
  "constant_mass_sf0.5 mass_nonneg_radiation radiation_nonneg 1 0.5 1 constant_mass_sf0.5.log"

  # Non-Negativity
  "constant_nonneg_sf0.05 mass_nonneg_radiation mass_radiation 1 1 0.05 constant_nonneg_sf0.05.log"
  "constant_nonneg_sf0.25 mass_nonneg_radiation mass_radiation 1 1 0.25 constant_nonneg_sf0.25.log"
  "constant_nonneg_sf0.5 mass_nonneg_radiation mass_radiation 1 1 0.5 constant_nonneg_sf0.5.log"

  # All models with constant lambdas of 1
  "constant_lambdas_str mass_nonneg_radiation _ 1 1 1 constant_lambdas_str.log"

  #### Trainable Lambdas ####

  # Trainable Lambdas (we exclude and set constant the other two losses)
  
  # Radiation
  "trainable_lambdas_radiation mass_nonneg mass_nonneg 1 1 1 trainable_lambdas_radiation.log"
  # Mass
  "trainable_lambdas_mass nonneg_radiation nonneg_radiation 1 1 1 trainable_lambdas_mass.log"
  # Non-Negativity
  "trainable_lambdas_nonneg mass_radiation mass_radiation 1 1 1 trainable_lambdas_nonneg.log"

  # Train all lambdas
  "train_all_lambdas _ _ 1 1 1 train_all_lambdas.log"

)
# Explanation: "exp_name constant_lambdas_str exclude_these_losses_str relative_scale_lambda_radiation relative_scale_lambda_mass relative_scale_lambda_nonneg log_file"

# Set maximum number of concurrent jobs (number of concurrent jobs on GPU 0)
MAX_CONCURRENT_JOBS=1

# Initialize AVAILABLE_GPUS with multiple entries of GPU 0 to allow concurrent jobs
AVAILABLE_GPUS=(0)  # Seven slots for GPU 0

# Associative array to keep track of GPU assignments and their PIDs
declare -A GPU_ASSIGNMENTS=()

# ================================
# Signal Handling and Cleanup
# ================================

# Function to kill all background jobs
cleanup_and_exit() {
  echo "Termination signal received. Killing all background jobs..."
  for pid in "${!GPU_ASSIGNMENTS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Killing job with PID $pid on GPU ${GPU_ASSIGNMENTS[$pid]}"
      kill "$pid"
    fi
  done
  exit 1
}

# Trap SIGINT (Ctrl+C) and SIGTERM to execute cleanup_and_exit
trap 'cleanup_and_exit' SIGINT SIGTERM

# ================================
# Function to Launch a Job
# ================================

launch_job() {
  local gpu_id=$1
  local exp_name=$2
  local constant_lambdas_str=$3
  local exclude_these_losses_str=$4
  local relative_scale_lambda_radiation=$5
  local relative_scale_lambda_mass=$6
  local relative_scale_lambda_nonneg=$7
  local log_file=$8

  output_log_file="${nov24_exp_dirpath}/${log_file}"

  # Construct the command
  cmd="CUDA_VISIBLE_DEVICES=${gpu_id} TF_FORCE_GPU_ALLOW_GROWTH=true python combined_loss_model.py \
    --constant_lambdas_str=\"${constant_lambdas_str}\" \
    --exclude_these_losses_str=\"${exclude_these_losses_str}\" \
    --output_results_dirpath=\"${nov24_exp_dirpath}\" \
    --data_subset_fraction=${data_fraction} \
    --n_epochs=${n_epochs} \
    --relative_scale_lambda_radiation=${relative_scale_lambda_radiation} \
    --relative_scale_lambda_mass=${relative_scale_lambda_mass} \
    --relative_scale_lambda_nonneg=${relative_scale_lambda_nonneg} > \"${output_log_file}\" 2>&1 &"

  echo "Launching job on GPU ${gpu_id}: ${exp_name}"
  echo "${cmd}"

  # Run the command in the background
  eval "${cmd}"

  # Get the PID of the last background job
  pid=$!

  # Record the GPU assignment for this PID
  GPU_ASSIGNMENTS[$pid]=$gpu_id
}

# ================================
# Function to Check and Free GPU Slots
# ================================

check_jobs() {
  for pid in "${!GPU_ASSIGNMENTS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      # Job has completed
      echo "Job with PID $pid has completed."
      AVAILABLE_GPUS+=("${GPU_ASSIGNMENTS[$pid]}")  # Make the GPU slot available
      unset GPU_ASSIGNMENTS[$pid]
    fi
  done
}

# ================================
# Main Loop to Launch Experiments
# ================================

for exp in "${experiments[@]}"; do
  # Wait until at least one GPU slot is available
  while [ ${#AVAILABLE_GPUS[@]} -eq 0 ]; do
    sleep 1
    check_jobs
  done

  # Get an available GPU slot (GPU 0)
  gpu_id="${AVAILABLE_GPUS[0]}"
  AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")  # Remove the assigned GPU slot

  # Parse experiment parameters
  read -r exp_name constant_lambdas_str exclude_these_losses_str \
    relative_scale_lambda_radiation relative_scale_lambda_mass \
    relative_scale_lambda_nonneg log_file <<<"$exp"

  # Launch the job
  launch_job "$gpu_id" "$exp_name" "$constant_lambdas_str" "$exclude_these_losses_str" \
    "$relative_scale_lambda_radiation" "$relative_scale_lambda_mass" \
    "$relative_scale_lambda_nonneg" "$log_file"

  # Optional: Brief pause to prevent overwhelming the system
  sleep 1
done

# ================================
# Wait for All Background Jobs to Complete
# ================================

# Wait for all background jobs to finish
wait

echo "All jobs have completed."
