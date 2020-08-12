'''
Call with 'python3'

Usage: 
    bdtTraining.py
    bdtTraining.py [--output_dir=DIR --settings_dir=DIR --hyperparameter_file=PTH]

Options:
    -o --output_dir=DIR             Directory of the output [default: None]
    -s --settings_dir=DIR           Directory of the settings [default: None]
    -h --hyperparameter_file=PTH    Path to the hyperparameters file [default: None]
'''
import os
import docopt
from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_aux_tools as hhat
from machineLearning.machineLearning import xgb_tools as xt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def main(output_dir, settings_dir, hyperparameter_file):
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
    channel_dir = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/info',
        global_settings['process'],
        global_settings['channel'],
        mode
    )
    preferences = dlt.get_hh_parameters(
        global_settings['channel'],
        global_settings['tauID_training'],
        channel_dir
    )
    if hyperparameter_file == 'None':
        hyperparameter_file = os.path.join(channel_dir, 'hyperparameters.json')
    hyperparameters = ut.read_parameters(hyperparas_file)[0]
    evaluation_main(global_settings, preferences, hyperparameters)


def split_data(global_settings, preferences):
    print('============ Starting evaluation ============')
    data = hhat.load_hh_data(preferences, global_settings)
    hhvt.plot_correlations(data, preferences['trainvars'])
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


def evaluation_main(global_settings, preferences, hyperparameters):
    even_data, odd_data = split_data(global_settings, preferences)
    even_model = model_creation(
        even_data, hyperparameters, preferences, global_settings, 'even_half'
    )
    odd_model = model_creation(
        odd_data, hyperparameters, preferences, global_settings, 'odd_half'
    )
    odd_infos = list(performance_prediction(
            odd_model, even_data, odd_data, global_settings,
            'odd', preferences
    ))
    even_infos = list(performance_prediction(
            even_model, odd_data, even_data, global_settings,
            'even', preferences
    ))
    hhvt.plotROC(odd_infos, even_infos, global_settings)
    hhvt.plot_sampleWise_bdtOutput(
        odd_model, even_data, preferences, global_settings
    )


def create_DMatrix(data, global_settings, preferences):
    trainvars = preferences['trainvars']
    dMatrix = xgb.DMatrix(
        data[trainvars],
        label=data['target'],
        nthread=global_settings['nthread'],
        feature_names=trainvars,
        weight=data['totalWeight']
    )
    return dMatrix


def model_creation(
        data, hyperparameters, preferences, global_settings, addition
):
    dtrain = create_DMatrix(data, global_settings, preferences)
    model = xt.create_model(
        hyperparameters, dtrain, global_settings['nthread']
    )
    save_pklFile(global_settings, model, addition)
    hhvt.plot_feature_importances(model, global_settings, addition)
    return model


def save_pklFile(global_settings, model, addition):
    output_dir = global_settings['output_dir']
    pklFile_path = os.path.join(output_dir, addition + '_model.pkl')
    with open(pklFile_path, 'wb') as pklFile:
        pickel.dump(model, pklFile)
    print('.pkl file saved to: ' + str(pklFile_path))


def performance_prediction(
        model, test_data, train_data, global_settings,
        addition, preferences
):
    dtest = create_DMatrix(test_data, global_settings, preferences)
    dtrain = create_DMatrix(train_data, global_settings, preferences)
    test_predicted_probabilities = model.predict(dtest)
    test_fpr, test_tpr, test_thresholds = roc_curve(
        test_data['target'].astype(int),
        predicted_probabilities,
        sampel_weight=test_data['totalWeight'].astype(float)
    )
    train_predicted_probabilities = model.predict(dtrain)
    train_fpr, train_tpr, train_thresholds = roc_curve(
        train_data['target'].astype(int),
        predicted_probabilities,
        sampel_weight=train_data['totalWeight'].astype(float)
    )
    train_auc = auc(train_fprt, train_tprt, reorder=True)
    test_auc = auc(test_fprt, test_tprt, reorder=True)
    test_info = {
        'fpr': test_fpr,
        'tpr': test_tpr,
        'auc': test_auc,
        'type': 'test',
        'addition': addition
    }
    train_info = {
        'fpr': train_fpr,
        'tpr': train_tpr,
        'auc': train_auc,
        'type': 'train',
        'addition': addition
    }
    return train_info, test_info









if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        settings_dir = arguments['--settings_dir']
        hyperparameter_file = arguments['--hyperparameter_file']
        main(output_dir, settings_dir, hyperparameter_file)
    except docopt.DocoptExit as e:
        print(e)
