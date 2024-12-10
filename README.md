# Adopting Physical Laws in ClimSim

## Installation
`conda env create --file src/environment.yaml --name climsim_env`

## Dataset preparation
`python src/preprocess_data.py`

## Train

Specify what lambdas should be kept constant and what losses should be excluded through the string parameters `--constant_lambdas_str` and `--exclude_these_losses_str`, separating them with underscores. For instance, in this command we train a model with all lambdas kept constant, and excluding their extra losses:

`python src/train.py --constant_lambdas_str "mass_radiation_nonneg" --exclude_these_losses_str="mass_radiation_nonneg" --n_epochs 1 --lr 1e-4 --batch_size 32 --output_results_dirpath=/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov30/zeromodel`

## Predict

To run the prediction script, use `predict.py` indicating the model path, the output directory to save the prediction and metrics and indicating weather or not you want to also predict for the validation set by using the booelan flag `--predict_validation`: 
`python predict.py --predict_validation --model_path /nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov30/massmodel/results_0.01/best_model_lambdas_excluded_radiation_excluded_nonneg_constant_radiation_constant_nonneg_scaledfactorofmse_1.00_scaledfactorofmass_1.00_scaledfactorofradiation_1.00_scaledfactorofnonneg_1.00_datafrac_0.01_epoch_1.keras --output_dir /nfs/turbo/coe-mihalcea/alvarovh/climsim/climsim_subset/datafraction_0.01/prediction_results`

## Visualize
The `plot_contour_map` function found in visualize.py allows you to plot the predicted variables on a global map by providing the latitudes, longitudes, and corresponding values of the variable of interest. For example, you can use it as follows: ``plot_contour_map(lat, lon, values, cmap='plasma', vmin=0, vmax=100, clev=20, title='Global Temperature', save_path='temperature_map.png')``. You can customize the colormap using the `cmap` parameter, adjust the contour detail with `clev`, and define the value range with `vmin` and `vmax` which is essential to be able of comparing multiple plots together. The function lets you add a title to the plot and save it to a specified file path using the `save_path` parameter. The plot is displayed in the Python plotting console if no file path is provided.

## Reproduce experiment and visualizations

In order to reproduce the experiments included in our final report, it is necessary to run `src/train_experiment.sh` and `src/predict_experiment.sh`.

## Visualize the results of experiments through the ipynb:
`visualize_experiment_results.ipynb`