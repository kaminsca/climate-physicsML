# Adopting Physical Laws in ClimSim

## Installation
`conda env create --file src/environment.yaml`

## Dataset preparation
Set up paths in `src/config.py`, then run `python src/preprocess_data.py`

## Train

Specify which loss functions' lambdas (loss scaling factors) should be kept constant and what losses should be excluded through the string parameters `--constant_lambdas_str` and `--exclude_these_losses_str`, separating them with underscores. For instance, in this command we train a model with all lambdas kept constant, and excluding their extra losses:

`python src/train.py --constant_lambdas_str "mass_radiation_nonneg" --exclude_these_losses_str="mass_radiation_nonneg" --n_epochs 1 --lr 1e-4 --batch_size 32 --output_results_dirpath=/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov30/zeromodel`

## Predict

To run the prediction script, use `predict.py` indicating the model path, the output directory to save the prediction and metrics and indicating weather or not you want to also predict for the validation set by using the booelan flag `--predict_validation`: 
`python predict.py --predict_validation --model_path example.keras --output_dir ./out_dir_example`

## Visualize
The `plot_contour_map` function found in visualize.py allows you to plot the predicted variables on a global map by providing the latitudes, longitudes, and corresponding values of a variable. For example, you can use it as follows: ``plot_contour_map(lat, lon, values, cmap='plasma', vmin=0, vmax=100, clev=20, title='Global Temperature', save_path='temperature_map.png')``. You can customize the colormap using the `cmap` parameter, adjust the contour detail with `clev`, and define the value range with `vmin` and `vmax`. The function lets you add a title to the plot and save it to a specified file path using the `save_path` parameter. The plot is shown in the console if no file path is provided.

## Reproduce experiment and visualizations

In order to reproduce the experiments included in our final report, it is necessary to run `src/train_experiment.sh` and `src/predict_experiment.sh`.

## Visualize the results of experiments through the ipynb:
`visualize_experiment_results.ipynb`
