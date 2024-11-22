# Example:
# python combined_loss_model.py --lambda_energy=0.1 --lambda_mass=0.1 --lambda_radiation=0.1 --lambda_humidity=0.1 --lambda_nonneg=0.1 --output_results_dirpath=/home/alvarovh/code/cse598_climate_proj/results_new/ --data_subset_fraction=0.01 --n_epochs=1
# python combined_loss_model.py /home/alvarovh/code/cse598_climate_proj/results_new_losses//results_0.01/best_model_lambda_mse_1.0_energy_0.1_mass_0.0_radiation_0.0_humidity_0.0_nonneg_0.0_datafrac_0.01_epoch_1.keras


############################################################################################################
lambda_mse=$(printf "%.1f" 1.0)
lambda_energy=$(printf "%.1f" 0.0)
lambda_mass=$(printf "%.1f" 0.0)
lambda_radiation=$(printf "%.1f" 0.0)
lambda_humidity=$(printf "%.1f" 0.0)
lambda_nonneg=$(printf "%.1f" 0.0)
output_results_dirpath=/home/alvarovh/code/cse598_climate_proj/results_new_losses_trainable/
data_subset_fraction=0.01
n_epochs=1

keras_model_path=${output_results_dirpath}results_${data_subset_fraction}/best_model_lambdas_mse_${lambda_mse}_energy_${lambda_energy}_mass_${lambda_mass}_radiation_${lambda_radiation}_humidity_${lambda_humidity}_nonneg_${lambda_nonneg}_datafrac_${data_subset_fraction}_epoch_${n_epochs}.keras
output_figures_path=$output_results_dirpath/results_${data_subset_fraction}/visualizations_lambda_mse_${lambda_mse}_energy_${lambda_energy}_mass_${lambda_mass}_radiation_${lambda_radiation}_humidity_${lambda_humidity}_nonneg_${lambda_nonneg}_datafrac_${data_subset_fraction}_nepochs_${n_epochs}
out_log_path=$output_results_dirpath/results_${data_subset_fraction}/log_lambda_mse_${lambda_mse}_energy_${lambda_energy}_mass_${lambda_mass}_radiation_${lambda_radiation}_humidity_${lambda_humidity}_nonneg_${lambda_nonneg}_datafrac_${data_subset_fraction}_nepochs_${n_epochs}.log

echo "check this log: $out_log_path"
command="python combined_loss_model.py --lambda_energy=$lambda_energy --lambda_mass=$lambda_mass --lambda_radiation=$lambda_radiation --lambda_humidity=$lambda_humidity --lambda_nonneg=$lambda_nonneg --output_results_dirpath=$output_results_dirpath --data_subset_fraction=$data_subset_fraction --n_epochs=$n_epochs"
echo "Runnnig command: $command"
python combined_loss_model.py --lambda_energy=$lambda_energy --lambda_mass=$lambda_mass --lambda_radiation=$lambda_radiation --lambda_humidity=$lambda_humidity --lambda_nonneg=$lambda_nonneg --output_results_dirpath=$output_results_dirpath --data_subset_fraction=$data_subset_fraction --n_epochs=$n_epochs > $out_log_path 2>&1
python predict_and_visualize.py $keras_model_path $output_figures_path


############################################################################################################
lambda_mse=1.0
lambda_energy=0.1
lambda_mass=0.1
lambda_radiation=0.1
lambda_humidity=0.1
lambda_nonneg=0.1
output_results_dirpath=/home/alvarovh/code/cse598_climate_proj/results_new_losses_trainable/
data_subset_fraction=0.01
n_epochs=1

keras_model_path=${output_results_dirpath}results_${data_subset_fraction}/best_model_lambdas_mse_${lambda_mse}_energy_${lambda_energy}_mass_${lambda_mass}_radiation_${lambda_radiation}_humidity_${lambda_humidity}_nonneg_${lambda_nonneg}_datafrac_${data_subset_fraction}_epoch_${n_epochs}.keras
output_figures_path=$output_results_dirpath/results_${data_subset_fraction}/visualizations_lambda_mse_${lambda_mse}_energy_${lambda_energy}_mass_${lambda_mass}_radiation_${lambda_radiation}_humidity_${lambda_humidity}_nonneg_${lambda_nonneg}_datafrac_${data_subset_fraction}_nepochs_${n_epochs}
out_log_path=$output_results_dirpath/results_${data_subset_fraction}/log_lambda_mse_${lambda_mse}_energy_${lambda_energy}_mass_${lambda_mass}_radiation_${lambda_radiation}_humidity_${lambda_humidity}_nonneg_${lambda_nonneg}_datafrac_${data_subset_fraction}_nepochs_${n_epochs}.log

echo "check this log: $out_log_path"
command="python combined_loss_model.py --lambda_energy=$lambda_energy --lambda_mass=$lambda_mass --lambda_radiation=$lambda_radiation --lambda_humidity=$lambda_humidity --lambda_nonneg=$lambda_nonneg --output_results_dirpath=$output_results_dirpath --data_subset_fraction=$data_subset_fraction --n_epochs=$n_epochs"
echo "Runnnig command: $command"
python combined_loss_model.py --lambda_energy=$lambda_energy --lambda_mass=$lambda_mass --lambda_radiation=$lambda_radiation --lambda_humidity=$lambda_humidity --lambda_nonneg=$lambda_nonneg --output_results_dirpath=$output_results_dirpath --data_subset_fraction=$data_subset_fraction --n_epochs=$n_epochs > $out_log_path 2>&1
python predict_and_visualize.py $keras_model_path $output_figures_path

python predict_and_visualize.py /home/alvarovh/code/cse598_climate_proj/results_new_losses_trainable//results_0.01/best_model_lambdas_mse_1.0_energy_0_mass_0_radiation_0_humidity_0_nonneg_0_datafrac_0.01_epoch_1.keras /home/alvarovh/code/cse598_climate_proj/results_new_losses_trainable//results_0.01/
# python predict_and_visualize.py /home/alvarovh/code/cse598_climate_proj/results_new_losses_trainable//results_0.01/best_model_TRAINEDLAMBDAS_datafrac_0.01_epoch_1.keras /home/alvarovh/code/cse598_climate_proj/results_new_losses_trainable//results_0.01/best_model_TRAINEDLAMBDAS_FIGURES
# /home/alvarovh/code/cse598_climate_proj/results_new_losses//results_0.01/best_model_lambdas_mse_1.0_energy_0.0_mass_0.0_radiation_0.0_humidity_0.0_nonneg_0.0_datafrac_0.01_epoch_1.keras
# /home/alvarovh/code/cse598_climate_proj/results_new_losses//results_0.01/best_model_lambda_mse_0_energy_0_mass_0_radiation_0_humidity_0_nonneg_0_datafrac_0.01_epoch_1.keras