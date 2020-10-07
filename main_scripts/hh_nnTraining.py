import os
from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_aux_tools as hhat
from machineLearning.machineLearning import nn_tools as nt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import json


def main(output_dir, ):
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
    evaluate_model(model, data_dict, global_settings)


def create_data_dict(preferences, global_settings):
    data = dlt.load_data(
        preferences,
        global_settings,
        remove_neg_weights=True
    )
    train, test = train_test_split(
        prepared_data, test_size=0.2, random_state=1)
    data_dict = {
        'trainvars': trainvars,
        'train': train,
        'test': test,
    }
    return data_dict


def create_model(nn_hyperparameters, preferences, global_settings, data_dict):
    nr_trainvars = len(preferences['trainvars'])
    num_class = 3
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
        sample_weight=data_dict['train']["totalWeight"],
        # validation_data=(
        #     data_dict['test'][trainvars],
        #     data_dict['test']['target'],
        #     sample_weight=data_dict['test']["totalWeight"],
        # )
    )
    return fitted_model


def evaluate_model(model, data_dict, global_settings):
    pred_train = model.predict_proba(data_dict['train'].values)
    pred_test = model.predict_proba(data_dict['test'].values)


if __name__ == '__main__':
    main('None')