'''
Call with 'python3'

Usage: slurm_nn_HH.py --parameter_file=PTH --output_dir=DIR

Options:
    -p --parameter_file=PTH      Path to parameters to be run
    --output_dir=DIR             Directory of the output
'''

from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import evaluation_tools as et
from machineLearning.machineLearning import nn_tools as nnt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import slurm_tools as st
from machineLearning.machineLearning import hh_aux_tools as hhat
from pathlib import Path
import os
import csv
import docopt
import json


def main(hyperparameter_file, output_dir):
    settings_dir = os.path.join(output_dir, 'run_settings')
    global_settings = ut.read_settings(settings_dir, 'global')
    nthread = global_settings['nthread']
    path = Path(hyperparameter_file)
    save_dir = str(path.parent)
    hyperparameters = ut.read_json_cfg(hyperparameter_file)
    addition = ut.create_infoPath_addition(global_settings)
    channel_dir = os.path.join(output_dir, 'run_info')
    info_dir = os.path.join(channel_dir, addition)
    preferences = hhat.get_hh_parameters(
        channel_dir,
        preferences['tauID_training'],
        info_dir
    )
    data = hhat.load_hh_data(preferences, global_settings)
    if bool(global_settings['use_kfold']):
        score = et.kfold_cv(
            nnt.model_evaluation_main,
            data,
            preferences['trainvars'],
            global_settings,
            hyperparameters
        )
    else:
        score, pred_train, pred_test = et.get_evaluation(
            nnt.model_evaluation_main,
            data,
            preferences['trainvars'],
            global_settings,
            hyperparameters
        )
        st.save_prediction_files(pred_train, pred_test, save_dir)
    score_path = os.path.join(save_dir, 'score.json')
    with open(score_path, 'w') as score_file:
        json.dump({global_settings['fitness_fn']: score}, score_file)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        parameter_file = arguments['--parameter_file']
        output_dir = arguments['--output_dir']
        main(parameter_file, output_dir)
    except docopt.DocoptExit as e:
        print(e)
