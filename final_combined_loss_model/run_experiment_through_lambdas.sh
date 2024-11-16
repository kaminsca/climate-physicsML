#!/bin/bash

# Define the common parameters
DATA_SUBSET_FRACTION=0.1
N_EPOCHS=10

# Loop through lambda values from 0.0 to 1.0 in increments of 0.1
for lambda in $(seq 0 0.1 1.0); do
  echo "Running the command:"
  echo "python combined_loss_model.py --lambda_energy=$lambda --data_subset_fraction=$DATA_SUBSET_FRACTION --n_epochs=$N_EPOCHS"
  python combined_loss_model.py --lambda_energy="$lambda" --data_subset_fraction="$DATA_SUBSET_FRACTION" --n_epochs="$N_EPOCHS"
# lets just print it
done
