# Adopting Physical Laws in ClimSim

## Installation

## Dataset preparation
`python preprocess_data.py`

## Train

Specify what lambdas should be kept constant and what losses should be excluded through the string parameters `--constant_lambdas_str` and `--exclude_these_losses_str`, separating them with underscores. For instance, in this command we train a model with all lambdas kept constant, and excluding their extra losses:

`python train.py --constant_lambdas_str "mass_radiation_nonneg" --exclude_these_losses_str="mass_radiation_nonneg" --n_epochs 1 --lr 1e-4 --batch_size 32 --output_results_dirpath=/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov30/zeromodel`

## Predict

## Visualize

## Reproduce experiment and visualizations

In order to reproduce the experiments included in our final report, it is necessary to run `src/train_experiment.sh` and `src/predict_experiment.sh`

## Visualize the results of experiments through the ipynb:
`visualize_experiment_results.ipynb`