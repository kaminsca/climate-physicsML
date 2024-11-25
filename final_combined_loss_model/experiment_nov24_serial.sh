#!/bin/bash

# ================================
# Experiment Configuration
# ================================

# Set experiment parameters
nov24_exp_dirpath="/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov24/initiallambdasscaled"
data_fraction=0.01
n_epochs=5

# Define additional training parameters
relative_scale_lambda_mse=1.0  # Adjust as needed
lr=0.001                        # Learning rate
batch_size=32                   # Batch size

# Define the list of experiments
declare -a experiments=(
  # Baseline with no custom losses
  # "zero_model mass_radiation_nonneg mass_radiation_nonneg 1 1 1 zero_model.log"

  # Constant Lambdas all set to 1
  # "constant_radiation_sf1 mass_nonneg_radiation mass_nonneg 1 1 1 constant_radiation_sf1.log"
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

# ================================
# Command Logging Configuration
# ================================

commands_file="commands.txt"

# Clear the commands file if it exists
> "$commands_file"

# ================================
# Function to Construct and Log Commands
# ================================

construct_command() {
  local exp_name=$1
  local constant_lambdas_str=$2
  local exclude_these_losses_str=$3
  local relative_scale_lambda_radiation=$4
  local relative_scale_lambda_mass=$5
  local relative_scale_lambda_nonneg=$6
  local log_file=$7

  local output_log_file="${nov24_exp_dirpath}/${log_file}"

  # Construct the command
  cmd="CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python combined_loss_model.py \
    --constant_lambdas_str=\"${constant_lambdas_str}\" \
    --exclude_these_losses_str=\"${exclude_these_losses_str}\" \
    --relative_scale_lambda_mse=${relative_scale_lambda_mse} \
    --relative_scale_lambda_radiation=${relative_scale_lambda_radiation} \
    --relative_scale_lambda_mass=${relative_scale_lambda_mass} \
    --relative_scale_lambda_nonneg=${relative_scale_lambda_nonneg} \
    --output_results_dirpath=\"${nov24_exp_dirpath}\" \
    --data_subset_fraction=${data_fraction} \
    --n_epochs=${n_epochs} \
    --lr=${lr} \
    --batch_size=${batch_size} > \"${output_log_file}\" 2>&1"

  # Echo the command to commands.txt
  echo "${cmd}" >> "$commands_file"

  # Optionally, echo the command to the console
  echo "Added to commands.txt: ${cmd}"
}

# ================================
# Main Loop to Log All Commands
# ================================

echo "Constructing and logging all experiment commands to ${commands_file}..."

for exp in "${experiments[@]}"; do
  # Parse experiment parameters
  read -r exp_name constant_lambdas_str exclude_these_losses_str \
    relative_scale_lambda_radiation relative_scale_lambda_mass \
    relative_scale_lambda_nonneg log_file <<<"$exp"

  # Construct and log the command
  construct_command "$exp_name" "$constant_lambdas_str" "$exclude_these_losses_str" \
    "$relative_scale_lambda_radiation" "$relative_scale_lambda_mass" \
    "$relative_scale_lambda_nonneg" "$log_file"
done

echo "All commands have been written to ${commands_file}."
echo "Review the commands in ${commands_file} before execution."

# ================================
# Execute Commands Sequentially
# ================================

echo "Starting sequential execution of experiments..."

# Read and execute each command from commands.txt
while IFS= read -r command; do
  echo "----------------------------------------"
  echo "Executing: ${command}"
  echo "----------------------------------------"
  
  # Execute the command
  eval "${command}"
  
  # Check if the command was successful
  if [ $? -ne 0 ]; then
    echo "Error: Command failed - ${command}" >> "${nov24_exp_dirpath}/error_log.txt"
    echo "Skipping to the next command..."
    continue
  fi

done < "$commands_file"

echo "All experiments have been executed successfully."
