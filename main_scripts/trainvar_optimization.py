'''
Call with 'python'

Usage:
    gen_mHH_profiling.py
    gen_mHH_profiling.py [--output_dir=DIR]

Options:
    -o --output_dir=DIR                     Fit the TProfile [default: 0]
    -i --create_info=BOOL             Create new histo_dict.json [default: 0]

'''

from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_aux_tools as hhat
import numpy as np
import xgboost
import docopt
from pathlib import Path

def prepare_data(output_dir):
    settings_dir = os.path.join(output_dir, 'run_settings')
    global_settings = ut.read_settings(settings_dir, 'global')
    num_classes = global_settings['num_classes']
    nthread = global_settings['nthread']
    path = Path(hyperparameter_file)
    save_dir = str(path.parent)
    hyperparameters = ut.read_parameters(hyperparameter_file)[0]
    channel_dir = os.path.join(output_dir, 'run_info')
    preferences = dlt.get_hh_parameters(
        global_settings['channel'],
        global_settings['tauID_training'],
        channel_dir
    )
    data = dlt.load_data(
        preferences,
        global_settings
    )
    if("nonres" not in global_settings['bdtType']):
        dlt.reweigh_dataframe(
            data,
            preferences['weight_dir'],
            preferences['trainvar_info'],
            ['gen_mHH'],
            preferences['masses']
        )
    elif 'nodeX' not in preferences['trainvars']:
        preferences['trainvars'].append('nodeX')
    hhat.normalize_hh_dataframe(data, preferences, global_settings)
    return data, preferences, global_settings


def optimization(data, hyperparameters, nthread, num_class):
    while len(trainvars) >= 10:
        dtrain = create_dtrain(data, trainvars, nthread)
        model = create_model(hyperparameters, dtrain, nthread, num_class)
        trainvars = drop_worst_performing_ones()


def create_dtrain(data, trainvars, nthread):
    dtrain = xgb.DMatrix(
        data[trainvars],
        label=data['Label'],
        nthread=nthread,
        feature_names=trainvars,
        weights=data['totalWeight']
    )
    return dtrain




def main(output_dir):
    print("Optimizing training variables")
    data, preferences, global_settings = prepare_data(output_dir)
    trainvars = optimization(
        data, hyperparameters, preferences, global_settings)
    save_optimized_trainvars()



if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        main(output_dir)
    except docopt.DocoptExit as e:
        print(e)