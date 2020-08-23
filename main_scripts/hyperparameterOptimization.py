'''
Hyperparameter optimization with Particle Swarm Optimization for HH/ttH analysis
Call with 'python'

Usage: hyperparameterOptimization.py
'''
from machineLearning.machineLearning import slurm_tools as st
from machineLearning.machineLearning import pso_tools as pt
from machineLearning.machineLearning import universal_tools as ut
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
    ut.save_info_dir(output_dir)
    print("::::::: Reading parameters :::::::")
    param_file = os.path.join(
        settings_dir,
        'xgb_parameters.json'
    )
    value_dicts = ut.read_parameters(param_file)
    pso_settings = ut.read_settings(settings_dir, 'pso')
    pso_settings.update(global_settings)
    print("\n============ Starting hyperparameter optimization ==========\n")
    best_hyperparameters = pt.particleSwarmOptimization(
        pso_settings, st.get_fitness_score, value_dicts)[0]
    print("\n============ Saving results ================\n")
    best_parameters_path = os.path.join(
        output_dir, 'best_hyperparameters.json')
    ut.save_dict_to_json(best_hyperparameters, best_parameters_path)
    print("Results saved to " + str(output_dir))


if __name__ == '__main__':
    main()
