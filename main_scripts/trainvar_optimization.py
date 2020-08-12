'''
Call with 'python'

Usage:
    trainvar_optimization.py
    trainvar_optimization.py [--corr_threshold=FLOAT --min_nr_trainvars=INT --step_size=INT]

Options:
    -c --corr_threshold=FLOAT       Threshold from which trainvar is dropped [default: 0.8]
    -n --min_nr_trainvars=INT       Number trainvars to end up with [default: 10]
    -s --step_size=INT              Number of trainvars dropped per iteration [default: 5]

'''

from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_aux_tools as hhat
from machineLearning.machineLearning import xgb_tools as xt
import numpy as np
import xgboost as xgb
import docopt
from pathlib import Path
import shutil
import os
import json

def prepare_data():
    cmssw_base = os.path.expandvars('$CMSSW_BASE')
    settings_dir = os.path.join(
        cmssw_base,
        'src/machineLearning/machineLearning/settings'
    )
    global_settings = ut.read_settings(settings_dir, 'global')
    num_classes = global_settings['num_classes']
    nthread = global_settings['nthread']
    if 'nonres' in global_settings['bdtType']:
        mode = 'nonRes'
    else:
        mode = 'res'
    channel_dir = os.path.join(
        cmssw_base,
        'src/machineLearning/machineLearning/info',
        global_settings['process'],
        global_settings['channel'],
    )
    mode_dir = os.path.join(channel_dir, mode)
    trainvars_path = os.path.join(mode_dir, 'trainvars.json')
    all_trainvars_path = os.path.join(channel_dir, 'all_trainvars.json')
    shutil.copy(all_trainvars_path, trainvars_path)
    preferences = dlt.get_hh_parameters(
        global_settings['channel'],
        global_settings['tauID_training'],
        mode_dir
    )
    data = hhat.load_hh_data(preferences, global_settings)
    return data, preferences, global_settings


def optimization(
        data, hyperparameters, trainvars,
        global_settings, min_nr_trainvars, step_size
):
    while len(trainvars) > min_nr_trainvars:
        dtrain = create_dtrain(data, trainvars, global_settings['nthread'])
        model = xt.create_model(
            hyperparameters, dtrain,
            global_settings['nthread'],
        )
        trainvars = drop_worst_performing_ones(
            model, trainvars, step_size, min_nr_trainvars
        )
    return trainvars


def drop_worst_performing_ones(model, trainvars, step_size, min_nr_trainvars):
    if len(trainvars) < (min_nr_trainvars + step_size):
        step_size = len(trainvars) - min_nr_trainvars
    feature_importances = model.get_fscore()
    keys = np.array(feature_importances.keys())
    values = np.array(feature_importances.values())
    index = np.argpartition(values, step_size)[:step_size]
    n_worst_performing = keys[index]
    for element in n_worst_performing:
        trainvars.remove(element)
    return trainvars


def create_dtrain(data, trainvars, nthread):
    dtrain = xgb.DMatrix(
        data[trainvars],
        label=data['target'],
        nthread=nthread,
        feature_names=trainvars,
        weight=data['totalWeight']
    )
    return dtrain


def drop_highly_currelated_variables(data, trainvars_initial, corr_threshold):
    correlations = data[trainvars_initial].corr()
    trainvars = list(trainvars_initial)
    for trainvar in trainvars:
        trainvars_copy = list(trainvars)
        trainvars_copy.remove(trainvar)
        for item in trainvars_copy:
            corr_value = abs(correlations[trainvar][item])
            if corr_value > corr_threshold:
                trainvars.remove(item)
                print("Removing " + str(item) + ". Correlation with " \
                    + str(trainvar) + " is " + str(corr_value)
                )
    return trainvars


def load_trainvars(all_trainvars_path):
    trainvar_info = dlt.read_trainvar_info(all_trainvars_path)
    trainvars = list(trainvar_info.keys())
    return trainvars


def main(corr_threshold, min_nr_trainvars, step_size):
    data, preferences, global_settings = prepare_data()
    cmssw_base = os.path.expandvars('$CMSSW_BASE')
    hyperparameter_file = os.path.join(
        cmssw_base,
        'src/machineLearning/machineLearning/info/default_hyperparameters.json'
    )
    hyperparameters = ut.read_parameters(hyperparameter_file)[0]
    print("Optimizing training variables")
    trainvars = preferences['trainvars']
    trainvars = drop_highly_currelated_variables(
        data, trainvars, corr_threshold=corr_threshold)
    trainvars = optimization(
        data, hyperparameters, trainvars,
        global_settings, min_nr_trainvars=min_nr_trainvars, step_size=step_size)
    save_optimized_trainvars(trainvars, preferences, trainvars_path)


def save_optimized_trainvars(trainvars, preferences, outpath):
    with open(outpath, 'wt') as outfile:
        for trainvar in trainvars:
            trainvar_info = preferences['trainvar_info']
            trainvar_dict = {
                'key': trainvar,
                'true_int': trainvar_info[trainvar]
            }
            json.dump(trainvar_dict, outfile)
            outfile.write('\n')


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        corr_threshold = float(arguments['--corr_threshold'])
        min_nr_trainvars = int(arguments['--min_nr_trainvars'])
        step_size = int(arguments['--step_size'])
        main(corr_threshold, min_nr_trainvars, step_size)
    except docopt.DocoptExit as e:
        print(e)