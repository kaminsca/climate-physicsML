import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the list of lambda values from 0 to 1 in steps of 0.1
lambdas = [round(x * 0.1, 1) for x in range(0, 7)]

# Base directory path for the results
base_dir = "/home/alvarovh/code/cse598_climate_proj/results_{lambda_value}/"

# Initialize a dictionary to store DataFrames for each lambda
lambda_dfs = {}

# Loop over each lambda value
for lambda_value in lambdas:
    # Format the directory path for the current lambda
    result_dir = base_dir.format(lambda_value=lambda_value)
    csv_file = os.path.join(result_dir, 'csv_logger_larger.txt')
    
    # Check if the CSV file exists and is not empty
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        lambda_dfs[lambda_value] = df
    else:
        print(f"CSV file missing or empty for lambda={lambda_value} at {csv_file}")

# Check if any DataFrame has multiple epochs
multiple_epochs = any(len(df) > 1 for df in lambda_dfs.values())

# Now, plot the metrics
for metric in ['accuracy', 'energy_loss', 'loss', 'mae', 'mse', 'val_accuracy', 'val_energy_loss', 'val_loss', 'val_mae', 'val_mse']:
    plt.figure(figsize=(10, 6))
    plt.title(f'{metric} vs {"Epoch" if multiple_epochs else "Lambda"}')
    
    has_data = False  # Track if there's data to plot

    if multiple_epochs:
        # If any DataFrame has multiple epochs, plot metric over epochs for each lambda
        for lambda_value, df in lambda_dfs.items():
            if metric in df.columns:
                plt.plot(df['epoch'], df[metric], label=f'Lambda {lambda_value}')
                has_data = True
        plt.xlabel('Epoch')
    else:
        # Single epoch case: plot metric against lambda values
        metric_values = [df[metric].iloc[0] for df in lambda_dfs.values() if metric in df.columns]
        lambdas_with_data = [lambda_val for lambda_val, df in lambda_dfs.items() if metric in df.columns]
        if metric_values:
            plt.plot(lambdas_with_data, metric_values, marker='o')
            has_data = True
        plt.xlabel('Lambda')
    
    plt.ylabel(metric)
    if has_data:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot
    plot_filename = f'{metric}_plot.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved plot: {plot_filename}")
