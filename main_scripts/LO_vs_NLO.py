'''
Call with 'python'
Usage: 
    bbWW_bdtTraining.py
    bbWW_bdtTraining.py [--output_dir=DIR --settings_dir=DIR --hyperparameter_file=PTH --debug=BOOL --channel=STR --res_nonres=XGB --mode=STR --era=INT --BM=INT --type=STR]

Options:
    -o --output_dir=DIR             Directory of the output [default: None]
    -s --settings_dir=DIR           Directory of the settings [default: None]
    -h --hyperparameter_file=PTH    Path to the hyperparameters file [default: None]
    -d --debug=BOOL                 Whether to debug the event classification [default: 0]
    -channel --channel=INT          which channel to be considered [default: bb1l_bdt]
    -res_nonres --res_nonres=STR        res or nonres to be considered [default: nonres]
    -mode --mode=STR                resolved or boosted to be considered [default: resolved]
    -era --era=INT                  era to be processed [default: 2016]
    -BM --BM=STR                    BM point to be considered  [default: None]
    -t --type=STR                  process to be processed [default: LO]
'''
#-ml_method  --ml_method=XGB     name of ml_method  [default: xgb]                                             
 #   -scenario   --scenario=nonres   which scenario to be considered [default: nonres] 
import os
import docopt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_parameter_reader as hpr
from machineLearning.machineLearning import hh_tools as hht
from machineLearning.machineLearning import data_loader as dl
from machineLearning.machineLearning import xgb_tools as xt
from machineLearning.machineLearning import converter_tools as ct
from machineLearning.machineLearning import bbWW_tools as bbwwt
from sklearn.metrics import roc_curve
from machineLearning.machineLearning.visualization import hh_visualization_tools as hhvt
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pandas
import numpy as np
import json
import subprocess
from datetime import datetime
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

def main(output_dir, settings_dir, hyperparameter_file, debug, type):
    if settings_dir == 'None':
        settings_dir = os.path.join(
            os.path.expandvars('$CMSSW_BASE'),
            'src/machineLearning/machineLearning/settings'
        )
    global_settings = settings_dir+'/'+'global_%s_%s_%s_settings.json' %(channel, mode, res_nonres)
    command = 'rsync %s ~/machineLearning/CMSSW_11_2_0_pre1/src/machineLearning/machineLearning/settings/global_settings.json' %global_settings
    p = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    global_settings = ut.read_settings(settings_dir, 'global')
    if output_dir == 'None':
        output_dir = 'LO_vs_NLO'+'/'+global_settings['ml_method']+'/'+\
                     res_nonres + '/' + mode +'/' + era
        global_settings['output_dir'] = output_dir
    else:
        global_settings['output_dir'] = output_dir
    global_settings['output_dir'] = os.path.expandvars(
        global_settings['output_dir'])
    if not os.path.exists(global_settings['output_dir']):
        os.makedirs(global_settings['output_dir'])
    channel_dir, info_dir, _ = ut.find_settings()
    scenario = global_settings['scenario']
    reader = hpr.HHParameterReader(channel_dir, scenario)
    preferences = reader.parameters
    if not BM=='None': 
      preferences["nonResScenarios"]=[BM]
    print('BM point to be considered: ' + str(preferences["nonResScenarios"]))
    if not era=='0': preferences['included_eras'] = [era.replace('20', '')]
    print('era: ' + str(preferences['included_eras']))
    if hyperparameter_file == 'None':
        hyperparameter_file = os.path.join(info_dir, 'hyperparameters.json')
    hyperparameters = ut.read_json_cfg(hyperparameter_file)
    print('hyperparametrs ' + str(hyperparameters))
    evaluation_main(global_settings, preferences, hyperparameters, debug, type)


def split_data(global_settings, preferences, addition):
    print('============ Starting evaluation ============')
    if os.path.exists(preferences['data_csv']):
        data = pandas.read_csv(preferences['data_csv'])
    else:
        normalizer = bbwwt.bbWWDataNormalizer
        loader = bbwwt.bbWWLoader(
             normalizer,
             preferences,
             global_settings, True
         )
        data = loader.data
    hhvt.plot_trainvar_multi_distributions(
        data, preferences['trainvars'],
        global_settings['output_dir']
    )
    keysNotToSplit = []
    if '3l_1tau' in global_settings['channel']:
        keysNotToSplit = ['WZ', 'DY', 'TTTo']
        print('These keys are excluded from splitting: ', keysNotToSplit)
    evtNotToSplit = (data['key'].isin(keysNotToSplit))
    evtEven = (data['event'].values % 2 == 0)
    evtOdd = ~(data['event'].values % 2 == 0)
    even_data = data.loc[np.logical_or(evtEven, evtNotToSplit)]
    odd_data = data.loc[np.logical_or(evtOdd, evtNotToSplit)]
    return even_data, odd_data

def evaluation_main(global_settings, preferences, hyperparameters, debug, type):
    if type == 'LO':
        train, test = 'signal_HH', 'signal_ggf_nonresonant_cHHH1_hh'
    else:
        train, test = 'signal_ggf_nonresonant_cHHH1_hh', 'signal_HH'
    even_data_all, odd_data_all = split_data(global_settings, preferences, type)
    even_data = even_data_all.loc[even_data_all['process'] != test]
    even_model = model_creation(
        even_data, hyperparameters, preferences, global_settings, type+'_even_half'
    )
    odd_data = odd_data_all.loc[odd_data_all['process'] != test]
    odd_model = model_creation(
        odd_data, hyperparameters, preferences, global_settings, type+'_odd_half'
    )
    test_data = even_data_all.loc[even_data_all['process'] != train]
    odd_infos = list(performance_prediction(
        odd_model, test_data, odd_data, global_settings,
        'odd', preferences, debug
    ))
    test_data = odd_data_all.loc[odd_data_all['process'] != train]
    even_infos = list(performance_prediction(
        even_model, test_data, even_data, global_settings,
        'even', preferences, debug
    ))
    if type == 'LO':
        test = 'NLO'
    else:
        test = 'LO'
    plotROC(odd_infos, even_infos, global_settings, type, test)

    
def model_creation(
        data, hyperparameters, preferences, global_settings, addition
):
    data_dict = {'train': data, 'trainvars': preferences['trainvars']}
    model = xt.create_model(
        hyperparameters, data_dict, global_settings['nthread'],
        objective='auc', weight='totalWeight'
    )
    bst = model.get_booster()
    bst.feature_names = [f.encode('ascii') for f in bst.feature_names]
    hhvt.plot_feature_importances(model, global_settings, addition)
    return model


def performance_prediction(
        model, test_data, train_data, global_settings,
        addition, preferences, debug
):
    test_predicted_probabilities = model.predict_proba(
        test_data[preferences['trainvars']])[:,1]
    test_fpr, test_tpr, test_thresholds = roc_curve(
        test_data['target'].astype(int),
        test_predicted_probabilities,
        sample_weight=test_data['totalWeight'].astype(float)
    )
    train_predicted_probabilities = model.predict_proba(
        train_data[preferences['trainvars']])[:,1]
    train_fpr, train_tpr, train_thresholds = roc_curve(
        train_data['target'].astype(int),
        train_predicted_probabilities,
        sample_weight=train_data['totalWeight'].astype(float)
    )
    train_auc = auc(train_fpr, train_tpr, reorder=True)
    test_auc = auc(test_fpr, test_tpr, reorder=True)
    test_info = {
        'fpr': test_fpr,
        'tpr': test_tpr,
        'auc': test_auc,
        'type': 'test',
        'addition': addition,
        'prediction': test_predicted_probabilities
    }
    train_info = {
        'fpr': train_fpr,
        'tpr': train_tpr,
        'auc': train_auc,
        'type': 'train',
        'addition': addition,
        'prediction': train_predicted_probabilities
    }
    if debug:
        save_RLE_predictions(
            test_data,
            train_data,
            addition,
            test_predicted_probabilities,
            train_predicted_probabilities,
            global_settings['output_dir']
        )
    return train_info, test_info

def plotROC(odd_infos, even_infos, global_settings, train, test):
    output_dir = global_settings['output_dir']
    fig, ax = plt.subplots(figsize=(6, 6))
    linestyles = ['-', '--']
    for odd_info, linestyle in zip(odd_infos, linestyles):
        if odd_info['type'] == 'train':
            label = train + 'odd'
        else:
            label = test + 'even'
        ax.plot(
            odd_info['fpr'], odd_info['tpr'], ls=linestyle, color='g',
            label=label + '_' + odd_info['type'] + 'AUC = ' + str(
                round(odd_info['auc'], 4))
        )
    for even_info, linestyle in zip(even_infos, linestyles):
        if even_info['type'] == 'train':
            label = train + 'even'
        else:
            label = test + 'odd'
        ax.plot(
            even_info['fpr'], even_info['tpr'], ls=linestyle, color='r',
            label=label + '_' + even_info['type'] + 'AUC = ' + str(
                round(even_info['auc'], 4))
        )
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid()
    plot_out = os.path.join(output_dir, train+'_ROC_curve.png')
    plt.tight_layout()
    fig.savefig(plot_out, bbox_inches='tight')
    plt.close('all')

if __name__ == '__main__':
    startTime = datetime.now()
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        settings_dir = arguments['--settings_dir']
        hyperparameter_file = arguments['--hyperparameter_file']
        debug = bool(int(arguments['--debug']))
        channel = arguments['--channel']
        mode = arguments['--mode']
        res_nonres = arguments['--res_nonres']
        era = arguments['--era']
        BM = arguments['--BM']
        type_ = arguments['--type']
        main('None', settings_dir, hyperparameter_file, debug, type_)
    except docopt.DocoptExit as e:
        print(e)
    print(datetime.now() - startTime)
