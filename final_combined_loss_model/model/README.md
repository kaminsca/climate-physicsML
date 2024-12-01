# Repeatability of experiments:

1. python preprocess_data.py 

2. To train:

- A single model:
python train.py --constant_lambdas_str "mass_radiation_nonneg" --exclude_these_losses_str="mass_radiation_nonneg" --relative_scale_lambda_mse 1.0 --relative_scale_lambda_mass 1.0 --relative_scale_lambda_radiation 1.0 --relative_scale_lambda_nonneg 1.0 --n_epochs 1 --lr 1e-4 --batch_size 32 --output_results_dirpath=/nfs/turbo/coe-mihalcea/alvarovh/large_data/cse598_project/experimental_results/Nov30/zeromodel

- The whole experiment:
bash train_experiment.sh

3. Test:

4. Figures and Tables:
