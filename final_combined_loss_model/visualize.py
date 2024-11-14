import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the list of lambda values from 0 to 1 in steps of 0.1
lambdas = [round(x * 0.1, 1) for x in range(0, 11)]
lambdas = [round(x * 0.1, 1) for x in range(0, 5)]
# Base directory path for the results
base_dir = "/home/alvarovh/code/cse598_climate_proj/results_{lambda_value}/"

# Initialize a dictionary to store DataFrames for each lambda
lambda_dfs = {}

# Loop over each lambda value
for lambda_value in lambdas:
    # Format the directory path for the current lambda
    result_dir = base_dir.format(lambda_value=lambda_value)
    csv_file = os.path.join(result_dir, 'csv_logger_larger.txt')
    
    # Check if the CSV file exists
    if os.path.exists(csv_file):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        lambda_dfs[lambda_value] = df
    else:
        print(f"CSV file not found for lambda={lambda_value} at {csv_file}")

# Now, plot the metrics
for metric in ['accuracy', 'energy_loss', 'loss', 'mae', 'mse', 'val_accuracy', 'val_energy_loss', 'val_loss', 'val_mae', 'val_mse']:
    plt.figure(figsize=(10, 6))
    plt.title(f'{metric} vs Lambda')
    
    epochs_present = False  # Flag to check if any DataFrame has multiple epochs
    
    for lambda_value, df in lambda_dfs.items():
        if len(df) > 1:
            epochs_present = True
            plt.plot(df['epoch'], df[metric], label=f'Lambda {lambda_value}')
        elif len(df) == 1:
            plt.bar(lambda_value, df[metric].iloc[0], width=0.05, label=f'Lambda {lambda_value}')
        
    plt.xlabel('Epoch' if epochs_present else 'Lambda')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot
    plot_filename = f'{metric}_plot.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved plot: {plot_filename}")
