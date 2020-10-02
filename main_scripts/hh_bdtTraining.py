'''
Call with 'python'

Usage: 
    bdtTraining.py
    bdtTraining.py [--output_dir=DIR --settings_dir=DIR --hyperparameter_file=PTH --debug=BOOL]

Options:
    -o --output_dir=DIR             Directory of the output [default: None]
    -s --settings_dir=DIR           Directory of the settings [default: None]
    -h --hyperparameter_file=PTH    Path to the hyperparameters file [default: None]
    -d --debug=BOOL                 Whether to debug the event classification [default: 0]
'''
import os
import docopt
from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_visualization_tools as hhvt
from machineLearning.machineLearning import hh_aux_tools as hhat
from machineLearning.machineLearning import xgb_tools as xt
from machineLearning.machineLearning import converter_tools as ct
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle


def main(output_dir, settings_dir, hyperparameter_file, debug):
    if output_dir == 'None':
        settings_dir = os.path.join(
            os.path.expandvars('$CMSSW_BASE'),
            'src/machineLearning/machineLearning/settings'
        )
    global_settings = ut.read_settings(settings_dir, 'global')
    if output_dir == 'None':
        output_dir = global_settings['output_dir']
    else:
        global_settings['output_dir'] = output_dir
    global_settings['output_dir'] = os.path.expandvars(
        global_settings['output_dir'])
    if not os.path.exists(global_settings['output_dir']):
        os.makedirs(global_settings['output_dir'])
    channel_dir, info_dir, _ = ut.find_settings()
    preferences = hhat.get_hh_parameters(
        channel_dir,
        global_settings['tauID_training'],
        info_dir
    )
    if hyperparameter_file == 'None':
        hyperparameter_file = os.path.join(info_dir, 'hyperparameters.json')
    hyperparameters = ut.read_json_cfg(hyperparameter_file)
    evaluation_main(global_settings, preferences, hyperparameters, debug)


def split_data(global_settings, preferences):
    print('============ Starting evaluation ============')
    data = hhat.load_hh_data(preferences, global_settings)
    hhvt.plot_correlations(data, preferences['trainvars'], global_settings)
    keysNotToSplit = []
    if ('3l_1tau' in global_settings['channel']):
        keysNotToSplit = ['WZTo', 'DY']
        print('These keys are excluded from splitting: ', keysNotToSplit)
    evtNotToSplit = (data['key'].isin(keysNotToSplit))
    evtEven = (data['event'].values % 2 == 0)
    evtOdd = ~(data['event'].values % 2 == 0)
    even_data = data.loc[np.logical_or(evtEven, evtNotToSplit)]
    odd_data = data.loc[np.logical_or(evtOdd, evtNotToSplit)]
    return even_data, odd_data


def evaluation_main(global_settings, preferences, hyperparameters, debug):
    even_data, odd_data = split_data(global_settings, preferences)
    even_model = model_creation(
        even_data, hyperparameters, preferences, global_settings, 'even_half'
    )
    odd_model = model_creation(
        odd_data, hyperparameters, preferences, global_settings, 'odd_half'
    )
    odd_infos = list(performance_prediction(
            odd_model, even_data, odd_data, global_settings,
            'odd', preferences, debug
    ))
    even_infos = list(performance_prediction(
            even_model, odd_data, even_data, global_settings,
            'even', preferences, debug
    ))
    hhvt.plotROC(odd_infos, even_infos, global_settings)
    hhvt.plot_sampleWise_bdtOutput(
        odd_model, even_data, preferences, global_settings
    )
    nodeWise_modelPredictions(
            odd_data, even_data, odd_model, even_model, preferences,
            global_settings
    )


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
    save_xmlFile(global_settings, model, addition)
    save_pklFile(global_settings, model, addition)
    hhvt.plot_feature_importances(model, global_settings, addition)
    return model


def save_xmlFile(global_settings, model, addition):
    if 'nonres' in global_settings['bdtType']:
        mode = 'nonres'
    else:
        mode = global_settings['spinCase']
    model_name = '_'.join([
        global_settings['channel'],
        addition,
        'model',
        mode
    ])
    xmlFile = os.path.join(global_settings['output_dir'], model_name + '.xml')
    bst = model.get_booster()
    features = bst.feature_names
    bdtModel = ct.BDTxgboost(model, features, ['Background', 'Signal'])
    bdtModel.to_tmva(xmlFile)
    print('.xml BDT model saved to ' + str(xmlFile))


def nodeWise_modelPredictions(
        odd_data, even_data,
        odd_model, even_model,
        preferences, global_settings,
        weight='totalWeight'
):
    output_dir = global_settings['output_dir']
    if 'nonres' in global_settings['bdtType']:
        nodes = preferences['nonResScenarios_test']
        mode = 'nodeXname'
    else:
        nodes = preferences['masses_test']
        mode = 'gen_mHH'
    nodeWise_performances = []
    roc_infos = []
    for node in nodes:
        split_odd_data = odd_data.loc[odd_data[mode] == node]
        split_odd_data_sig = split_odd_data.loc[split_odd_data['target'] == 1]
        split_odd_data_bkg = split_odd_data.loc[split_odd_data['target'] == 0]
        split_even_data = even_data.loc[even_data[mode] == node]
        split_even_data_sig = split_even_data.loc[split_even_data['target'] == 1]
        split_even_data_bkg = split_even_data.loc[split_even_data['target'] == 0]
        odd_info_sig = list(performance_prediction(
                odd_model, split_even_data_sig, split_odd_data_sig,
                global_settings, 'odd', preferences, False
        ))
        odd_info_bkg = list(performance_prediction(
                odd_model, split_even_data_bkg, split_odd_data_bkg,
                global_settings, 'odd', preferences, False
        ))
        odd_total_infos = list(performance_prediction(
                odd_model, split_even_data, split_odd_data, global_settings,
                'odd', preferences, False
        ))
        even_total_infos = list(performance_prediction(
                even_model, split_odd_data, split_even_data, global_settings,
                'even', preferences, False
        ))
        nodeWise_histo_dict = {
            'sig_test_w': split_even_data_sig[weight],
            'sig_train_w': split_odd_data_sig[weight],
            'bkg_test_w': split_even_data_bkg[weight],
            'bkg_train_w': split_odd_data_bkg[weight],
            'sig_test': odd_info_sig[1]['prediction'],
            'sig_train': odd_info_sig[0]['prediction'],
            'bkg_test': odd_info_bkg[1]['prediction'],
            'bkg_train': odd_info_bkg[0]['prediction'],
            'node': node
        }
        roc_info = {
            'even_auc_test': even_total_infos[1]['auc'],
            'odd_auc_test': odd_total_infos[1]['auc'],
            'even_auc_train': even_total_infos[0]['auc'],
            'odd_auc_train': odd_total_infos[0]['auc'],
            'even_fpr_test': even_total_infos[1]['fpr'],
            'odd_fpr_test': odd_total_infos[1]['fpr'],
            'even_fpr_train': even_total_infos[0]['fpr'],
            'odd_fpr_train': odd_total_infos[0]['fpr'],
            'even_tpr_test': even_total_infos[1]['tpr'],
            'odd_tpr_test': odd_total_infos[1]['tpr'],
            'even_tpr_train': even_total_infos[0]['tpr'],
            'odd_tpr_train': odd_total_infos[0]['tpr'],
            'node': node
        }
        nodeWise_performances.append(nodeWise_histo_dict)
        roc_infos.append(roc_info)
    hhvt.plot_nodeWise_performance(
        global_settings, nodeWise_performances, mode)
    hhvt.plot_nodeWise_roc(global_settings, roc_infos, mode)


def save_pklFile(global_settings, model, addition):
    output_dir = global_settings['output_dir']
    pklFile_path = os.path.join(output_dir, addition + '_model.pkl')
    with open(pklFile_path, 'wb') as pklFile:
        pickle.dump(model, pklFile)
    print('.pkl file saved to: ' + str(pklFile_path))


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


def save_RLE_predictions(
        test_data,
        train_data,
        addition,
        test_predicted,
        train_predicted,
        output_dir
):
    test_data = create_rle_str(test_data)
    test_rles = np.array(test_data['rle'])
    train_data = create_rle_str(train_data)
    train_rles = np.array(train_data['rle'])
    test_outfile = os.path.join(
        output_dir,
        '_'.join([addition, model, 'test']) + '.json')
    train_outfile = os.path.join(
        output_dir,
        '_'.join([addition, model, 'train']) + '.json')
    save_rle_dict(test_outfile, test_predicted, test_rles)
    save_rle_dict(train_outfile, train_predicted, train_rles)


def save_rle_dict(outfile, predictions, rles):
    with open(outfile, 'wt') as outFile:
        outDict = {}
        for pred, rle in zip(predictions, rles):
            outDict[rle] = pred
        json.dump(outDict, outFile)



def create_rle_str(data):
    data['rle'] = data.apply(lambda row: ':'.join([
        str(int(row.run)),
        str(int(row.luminosityBlock)),
        str(int(row.event))]),
        axis=1
    )
    return data



if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        settings_dir = arguments['--settings_dir']
        hyperparameter_file = arguments['--hyperparameter_file']
        debug = bool(int(arguments['--debug']))
        main(output_dir, settings_dir, hyperparameter_file, debug)
    except docopt.DocoptExit as e:
        print(e)
