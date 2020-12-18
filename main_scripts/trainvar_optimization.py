"""
Call with 'python'

Usage:
    trainvar_optimization.py
    trainvar_optimization.py [--corr_threshold=FLOAT --min_nr_trainvars=INT --step_size=INT]

Options:
    -c --corr_threshold=FLOAT       Threshold from which trainvar is dropped [default: 0.8]
    -n --min_nr_trainvars=INT       Number trainvars to end up with [default: 10]
    -s --step_size=INT              Number of trainvars dropped per iteration [default: 5]
    -a --analysis=STR               Options: 'hh-bbWW', 'hh-multilepton' [default:hh-multilepton]
"""
import shutil
import os
import json
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_parameter_reader as hpr
from machineLearning.machineLearning import hh_tools as hht
from machineLearning.machineLearning import data_loader as dl
from machineLearning.machineLearning import xgb_tools as xt
import numpy as np
import docopt


def prepare_data():
    channel_dir, info_dir, global_settings = ut.find_settings()
    trainvars_path = os.path.join(info_dir, 'trainvars.json')
    all_trainvars_path = os.path.join(channel_dir, 'all_trainvars.json')
    shutil.copy(all_trainvars_path, trainvars_path)
    scenario = global_settings['scenario']
    reader = hpr.HHParameterReader(channel_dir, scenario)
    normalizer = hht.HHDataNormalizer
    loader = hht.HHDataLoader(
        normalizer,
        preferences,
        global_settings
    )
    data = loader.data
    return data, preferences, global_settings, trainvars_path



if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        corr_threshold = float(arguments['--corr_threshold'])
        min_nr_trainvars = int(arguments['--min_nr_trainvars'])
        step_size = int(arguments['--step_size'])
        main(corr_threshold, min_nr_trainvars, step_size)
    except docopt.DocoptExit as e:
        print(e)
