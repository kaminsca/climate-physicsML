#!/bin/bash

# Define the common parameters
DATA_SUBSET_FRACTION=0.1
N_EPOCHS=1

# Maximum number of concurrent jobs
MAX_CONCURRENT_JOBS=7
OUTPUT_GLOBAL_DIR="/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov18/DataFraction$DATA_SUBSET_FRACTION"

# Trap to kill all child processes on exit
trap 'cleanup_and_exit' SIGINT SIGTERM

cleanup_and_exit() {
  echo "Killing all running jobs..."
  pkill -P $$  # Kill all child processes spawned by this script
  exit 1
}

# Create an array to track available GPUs
AVAILABLE_GPUS=($(seq 0 $((MAX_CONCURRENT_JOBS - 1))))

# Function to launch a job
launch_job() {
  local gpu_id=$1
  local batch_size=$2
  local learning_rate=$3
  local lambda=$4

  output_dir="${OUTPUT_GLOBAL_DIR}/output_batch_size_${batch_size}_learning_rate_${learning_rate}_lambda_${lambda}"
  output_file="${output_dir}/output_batch_size_${batch_size}_learning_rate_${learning_rate}_lambda_${lambda}.out"

  # Skip if a .keras model file already exists in any subdirectory of output_dir
  if find "$output_dir" -type f -name "*.keras" 1> /dev/null 2>&1; then
    echo "Skipping: Model already exists in $output_dir or its subdirectories"
    return
  fi

  # Create the output directory if it doesn't exist
  mkdir -p "$output_dir"

  echo "Running on GPU $gpu_id:"
  # for the specific output file
  echo "CUDA_VISIBLE_DEVICES=$gpu_id python combined_loss_model.py --lambda_energy=$lambda --data_subset_fraction=$DATA_SUBSET_FRACTION --n_epochs=$N_EPOCHS --batch_size=$batch_size --lr=$learning_rate --output_results_dirpath=$output_dir > $output_file"
  # for the slurm out
  echo "CUDA_VISIBLE_DEVICES=$gpu_id python combined_loss_model.py --lambda_energy=$lambda --data_subset_fraction=$DATA_SUBSET_FRACTION --n_epochs=$N_EPOCHS --batch_size=$batch_size --lr=$learning_rate --output_results_dirpath=$output_dir 

  # Set CUDA_VISIBLE_DEVICES to the physical GPU and run the job
  CUDA_VISIBLE_DEVICES=$gpu_id python combined_loss_model.py \
    --lambda_energy="$lambda" \
    --data_subset_fraction="$DATA_SUBSET_FRACTION" \
    --n_epochs="$N_EPOCHS" \
    --batch_size="$batch_size" \
    --lr="$learning_rate" \
    --output_results_dirpath="$output_dir" > "$output_file" 2>&1 &
}


# Main loop
for batch_size in 32 64 128
do
  for learning_rate in 0.00001 0.00005 0.0001 0.001
  do
    for lambda in $(seq 0.0 0.1 1.0)
    do
      while [ ${#AVAILABLE_GPUS[@]} -eq 0 ]; do
        # Wait for a GPU to become available
        sleep 1

        # Check if any background jobs have finished
        for pid in $(jobs -p); do
          if ! kill -0 $pid 2>/dev/null; then
            # If the job is finished, free up the GPU
            AVAILABLE_GPUS+=(${GPU_ASSIGNMENTS[$pid]})
            unset GPU_ASSIGNMENTS[$pid]  # Remove the finished job from assignments
          fi
        done
      done

      # Pop an available GPU from the list
      gpu_id=${AVAILABLE_GPUS[0]}
      AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")

      # Launch the job
      launch_job $gpu_id $batch_size $learning_rate $lambda
      pid=$!
      GPU_ASSIGNMENTS[$pid]=$gpu_id  # Record the GPU assignment for this job
    done
  done
done

# Wait for any remaining background jobs to finish
wait
