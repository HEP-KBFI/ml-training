import os
from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_aux_tools as hhat
from machineLearning.machineLearning import nn_tools as nt
from machineLearning.machineLearning import multiclass_tools as mt
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
import numpy as np
import json

nn_hyperparameters = {
    'nr_hidden_layers': 3,
    'learning_rate': 0.005,
    'schedule_decay': 0.01,
    'visible_layer_dropout_rate': 0.9,
    'hidden_layer_dropout_rate': 0.65,
    'alpha': 5,
    'batch_size': 256,
    'epochs': 70
}


def main(output_dir):
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
    data_dict = create_data_dict(preferences, global_settings)
    model = create_model(
        nn_hyperparameters, preferences, global_settings, data_dict)
    train_info, test_info = evaluate_model(model, data_dict, global_settings)
    hhvt.plotROC(train_info, test_info, global_settings)


def create_data_dict(preferences, global_settings):
    data = dlt.load_data(
        preferences,
        global_settings,
        remove_neg_weights=True
    )
    data = mt.multiclass_encoding(data)
    train, test = train_test_split(
        data, test_size=0.2, random_state=1)
    data_dict = {
        'trainvars': preferences['trainvars'],
        'train': train,
        'test': test,
    }
    return data_dict


def create_model(nn_hyperparameters, preferences, global_settings, data_dict):
    trainvars = preferences['trainvars']
    nr_trainvars = len(trainvars)
    num_class = len(set(data_dict['train']['target']))
    number_samples = len(data_dict['train'])
    model_structure = nt.create_nn_model(
        nn_hyperparameters,
        nr_trainvars,
        num_class,
        number_samples,
        metrics=['accuracy'],
    )
    fitted_model = model_structure.fit(
        data_dict['train'][trainvars].values,
        data_dict['train']['target'],
        data_dict['train']['totalWeight'].values,
        validation_data=(
            data_dict['test'][trainvars],
            data_dict['test']['target'],
            data_dict['test']['totalWeight'].values
        )
    )
    return model_structure


def evaluate_model(model, data_dict, global_settings):
    trainvars = data_dict['trainvars']
    train_predicted_probabilities = model.predict_proba(
        data_dict['train'][trainvars].values)
    test_predicted_probabilities = model.predict_proba(
        data_dict['test'][trainvars].values)
    test_fpr, test_tpr, test_thresholds = mt.roc_curve(
        data_dict['test']['target'].astype(int),
        test_predicted_probabilities,
        data_dict['test']['totalWeight'].astype(float)
    )
    train_fpr, train_tpr, train_thresholds = mt.roc_curve(
        data_dict['train']['target'].astype(int),
        train_predicted_probabilities,
        data_dict['test']['totalWeight'].astype(float)
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
    return train_info, test_info




if __name__ == '__main__':
    main('None')