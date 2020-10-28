'''
Hyperparameter optimization with Particle Swarm Optimization for HH/ttH analysis
Call with 'python'

Usage:
    hyperparameterOptimization.py [--continue=BOOL --opt_dir=STR]

Options:
    -c --continue=BOOL      Whether to continue from a previous optimization [default: 0]
    -o --opt_dir=STR        Directory of the previous iteration steps [default: None]
'''
import os
import numpy as np
import docopt
from machineLearning.machineLearning import slurm_tools as st
from machineLearning.machineLearning import pso_tools as pt
from machineLearning.machineLearning import universal_tools as ut
np.random.seed(1)


def main(to_continue, opt_dir):
    if not to_continue:
        settings_dir = os.path.join(
            os.path.expandvars('$CMSSW_BASE'),
            'src/machineLearning/machineLearning/settings'
        )
    else:
        settings_dir = os.path.join(opt_dir, 'run_settings')
    global_settings = ut.read_settings(settings_dir, 'global')
    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not to_continue:
        ut.save_run_settings(output_dir)
        ut.save_info_dir(output_dir)
    print("::::::: Reading parameters :::::::")
    param_file = os.path.join(
        settings_dir,
        'xgb_parameters.json'
    )
    hyperparameter_info = ut.read_json_cfg(param_file)
    pso_settings = ut.read_settings(settings_dir, 'pso')
    pso_settings.update(global_settings)
    print("\n============ Starting hyperparameter optimization ==========\n")
    swarm = pt.ParticleSwarm(pso_settings, st.get_fitness_score, hyperparameter_info)
    optimal_hyperparameters = swarm.particleSwarmOptimization()[0]
    print("\n============ Saving results ================\n")
    best_parameters_path = os.path.join(
        output_dir, 'best_hyperparameters.json')
    ut.save_dict_to_json(optimal_hyperparameters, best_parameters_path)
    print("Results saved to " + str(output_dir))


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        to_continue = bool(int(arguments['--continue']))
        opt_dir = arguments['--opt_dir']
        main(to_continue, opt_dir)
    except docopt.DocoptExit as e:
        print(e)
