#!/bin/bash

# Define paths
MODEL_DIR="/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov30_2230/results_0.01"
PREDICTION_DIR="/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov30_2230/predictions"

# Create the prediction directory if it doesn't exist
mkdir -p "$PREDICTION_DIR"

# Loop through all Keras models in the directory
for model_path in "$MODEL_DIR"/*keras; do
  echo "Evaluating model: $model_path"

  # Execute the predict.py script for each model
  python predict.py --predict_validation --model_path "$model_path" --output_dir "$PREDICTION_DIR"

  # Check if the command succeeded
  if [ $? -ne 0 ]; then
    echo "Error occurred while processing $model_path"
    exit 1
  fi
done

echo "All models have been evaluated successfully!"
