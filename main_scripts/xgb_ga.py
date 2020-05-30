'''
Hyperparameter optimization with Genetic Algorithm for ttH/HH analysis.
Call with 'python'

Usage: xgb_pso_hh.py
'''
from machineLearning.machineLearning import slurm_tools as st
from machineLearning.machineLearning import ga_main_tools as gmt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import xgb_tools as xt
import os
import numpy as np
np.random.seed(1)


def main():
    settings_dir = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/settings'
    )
    global_settings = ut.read_settings(settings_dir, 'global')
    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    ut.save_run_settings(output_dir)
    ut.save_info_settings(output_dir, global_settings)
    print("::::::: Reading parameters :::::::")
    param_file = os.path.join(
        settings_dir,
        'xgb_parameters.json'
    )
    parameters = ut.read_parameters(param_file)
    ga_settings = ut.read_settings(settings_dir, 'ga')
    settings = global_settings
    settings.update(ga_settings)
    print("\n============ Starting hyperparameter optimization ==========\n")
    best_hyperparameters = gmt.evolution(
        settings, parameters, xt.prepare_run_params, st.get_fitness_score)
    print("\n============ Saving results ================\n")
    best_parameters_path = os.path.join(
        output_dir, 'best_hyperparameters.json')
    ut.save_dict_to_json(best_hyperparameters, best_parameters_path)
    print("Results saved to " + str(output_dir))


if __name__ == '__main__':
    main()
