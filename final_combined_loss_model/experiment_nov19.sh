#!/bin/bash

# Usage: ./train_models.sh <data_subset_fraction> <lower_lambda> <upper_lambda> <lambda_step> <epochs>

# Validate arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <data_subset_fraction> <lower_lambda> <upper_lambda> <lambda_step> <epochs>"
    exit 1
fi

# Assign command-line arguments to variables
DATA_SUBSET_FRACTION=$1
LOWER_LAMBDA=$2
UPPER_LAMBDA=$3
LAMBDA_STEP=$4
EPOCHS=$5

# Define the base output directory
BASE_OUTPUT_DIR="/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov19/results_59inindex_19novexperiment_datafraction${DATA_SUBSET_FRACTION}_epochs${EPOCHS}"

# Loop through lambda values
for i in $(seq $LOWER_LAMBDA $LAMBDA_STEP $UPPER_LAMBDA); do
    OUTPUT_LOG_PATH="${BASE_OUTPUT_DIR}/output_lambda_${i}_datafraction_${DATA_SUBSET_FRACTION}_epochs_${EPOCHS}.log"
    # Format lambda_energy to 2 decimal places
    LAMBDA=$(printf "%.2f" $i)
    # Create the output directory if it doesn't exist
    mkdir -p "$BASE_OUTPUT_DIR"


    # Start time
    START_TIME=$(date +%s)
    # Run the Python script with the current lambda value
    echo "Training model for lambda_energy=${LAMBDA}, data_subset_fraction=${DATA_SUBSET_FRACTION}, n_epochs=${EPOCHS}"
    python combined_loss_model.py \
        --lambda_energy=$LAMBDA \
        --data_subset_fraction=$DATA_SUBSET_FRACTION \
        --n_epochs=$EPOCHS \
        --output_results_dirpath "$BASE_OUTPUT_DIR" > "$OUTPUT_LOG_PATH" 2>&1
    
    # End time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Training completed for lambda_energy=${LAMBDA} in $DURATION seconds"
    else
        echo "Training failed for lambda_energy=${LAMBDA}" >&2
        exit 1
    fi
done

echo "All models trained successfully."
