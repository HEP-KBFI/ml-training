"""
Call with 'python'

Usage:
    trainvar_optimization.py
    trainvar_optimization.py [--corr_threshold=FLOAT --min_nr_trainvars=INT --step_size=INT --analysis=STR]

Options:
    -c --corr_threshold=FLOAT       Threshold from which trainvar is dropped [default: 0.8]
    -n --min_nr_trainvars=INT       Number trainvars to end up with [default: 10]
    -s --step_size=INT              Number of trainvars dropped per iteration [default: 5]
    -a --analysis=STR               Options: 'hh-bbWW', 'hh-multilepton' [default: HHmultilepton]

"""
import os
import json
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_parameter_reader as hpr
from machineLearning.machineLearning import hh_tools as hht
from machineLearning.machineLearning import bbWW_tools as bbwwt
from machineLearning.machineLearning import trainvar_optimization_tools as tot
import docopt


def prepare_data(analysis):
    channel_dir, info_dir, global_settings = ut.find_settings()
    scenario = global_settings['scenario']
    reader = hpr.HHParameterReader(channel_dir, scenario)
    preferences = reader.parameters
    if analysis == 'HHmultilepton':
        normalizer = hht.HHDataNormalizer
        loader = hht.HHDataLoader(
            normalizer,
            preferences,
            global_settings
        )
    elif analysis == 'HHbbWW':
        normalizer = bbwwt.bbWWDataNormalizer
        loader = bbwwt.bbWWLoader(
            normalizer,
            preferences,
            global_settings
        )
    data = loader.data
    scenario = global_settings['scenario']
    scenario = scenario if 'nonres' in scenario else 'res/' + scenario
    hyperparameters_file = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/info/',
        global_settings['process'],
        global_settings['channel'],
        scenario,
        'hyperparameters.json'
    )
    with open(hyperparameters_file, 'rt') as in_file:
        preferences['hyperparameters'] = json.load(in_file)
    preferences['trainvars'] = preferences['all_trainvar_info'].keys()
    return data, preferences, global_settings


def main(corr_threshold, min_nr_trainvars, step_size, analysis):
    data, preferences, global_settings = prepare_data(analysis)
    if global_settings['ml_method'] == 'xgb':
        optimizer = tot.XGBTrainvarOptimizer(
            data, preferences, global_settings, preferences['hyperparameters'],
            corr_threshold, min_nr_trainvars, step_size, 'totalWeight'
        )
    elif global_settings['ml_method'] == 'nn':
        optimizer = tot.NNTrainvarOptimizer(
            data, preferences, global_settings, corr_threshold,
            min_nr_trainvars, step_size, 'totalWeight'
        )
    elif global_settings['ml_method'] == 'lbn':
        optimizer = tot.LBNTrainvarOptimizer(
            data, preferences, global_settings, particles, corr_threshold,
            min_nr_trainvars, step_size, 'totalWeight'
        )
    else:
        raise NotImplementedError('Given ml_method is not implemented')
    optimizer.optimization_collector()


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        corr_threshold = float(arguments['--corr_threshold'])
        min_nr_trainvars = int(arguments['--min_nr_trainvars'])
        step_size = int(arguments['--step_size'])
        analysis = arguments['--analysis']
        main(corr_threshold, min_nr_trainvars, step_size, analysis)
    except docopt.DocoptExit as e:
        print(e)
