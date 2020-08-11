'''XGBoost tools for initializing the hyperparameters, creating the model and
other relevant ones.
'''
import numpy as np
import xgboost as xgb
import os
from machineLearning.machineLearning import evaluation_tools as et


def initialize_values(value_dicts):
    '''Initializes the parameters according to the value dict specifications

    Parameters:
    ----------
    value_dicts : list of dicts
        Specifications how each value should be initialized

    Returns:
    -------
    sample : list of dicts
        Hyperparameters of each particle
    '''
    sample = {}
    for xgb_params in value_dicts:
        if bool(xgb_params['true_int']):
            value = np.random.randint(
                low=xgb_params['range_start'],
                high=xgb_params['range_end']
            )
        else:
            value = np.random.uniform(
                low=xgb_params['range_start'],
                high=xgb_params['range_end']
            )
        if bool(xgb_params['exp']):
            value = np.exp(value)
        sample[str(xgb_params['p_name'])] = value
    return sample


def prepare_run_params(value_dicts, sample_size):
    ''' Creates parameter-sets for all particles (sample_size)

    Parameters:
    ----------
    value_dicts : list of dicts
        Specifications how each value should be initialized
    sample_size : int
        Number of particles to be created

    Returns:
    -------
    run_params : list of dicts
        List of parameter-sets for all particles
    '''
    run_params = []
    for i in range(sample_size):
        run_param = initialize_values(value_dicts)
        run_params.append(run_param)
    return run_params


def create_model(hyperparameters, dtrain, nthread):
    label = dtrain.get_label()
    weight = dtrain.get_weight()
    sum_wpos = sum(weight[i] for i in range(len(label)) if label[i] == 1.0)
    sum_wneg = sum(weight[i] for i in range(len(label)) if label[i] == 0.0)
    parameters = {
        'objective': 'binary:logistic',
        'scale_pos_weight': sum_wneg/sum_wpos,
        'eval_metric': 'auc',
        'silent': 1,
        'nthread': nthread
    }
    watchlist = [(dtrain,'train')]
    hyp_copy = hyperparameters.copy()
    num_boost_round = hyp_copy.pop('num_boost_round')
    parameters.update(hyp_copy)
    parameters = list(parameters.items())+[('eval_metric', 'ams@0.15')]
    model = xgb.train(
        parameters,
        dtrain,
        num_boost_round,
        watchlist,
        verbose_eval=False
    )
    return model


def create_xgb_data_dict(data_dict, nthread):
    '''Creates the data_dict for the XGBoost method

    Parameters:
    ----------
    data_dict : dict
        Contains some of the necessary information for the evaluation.
    nthread : int
        Number of threads to be used

    Returns:
    -------
    data_dict : dict
        Contains all the necessary information for the evaluation.
    '''
    dtrain = xgb.DMatrix(
        data_dict['traindataset'],
        label=data_dict['training_labels'],
        nthread=nthread,
        feature_names=data_dict['trainvars'],
        weight=data_dict['train_weights']
    )
    dtest = xgb.DMatrix(
        data_dict['testdataset'],
        label=data_dict['testing_labels'],
        nthread=nthread,
        feature_names=data_dict['trainvars'],
        weight=data_dict['test_weights']
    )
    data_dict['dtrain'] = dtrain
    data_dict['dtest'] = dtest
    return data_dict


def evaluate_model(data_dict, global_settings, model):
    '''Evaluates the model for the XGBoost method

    Parameters:
    ----------
    data_dict : dict
        Contains all the necessary information for the evaluation.
    global_settings : dict
        Preferences for the optimization
    model : XGBoost Booster?
        Model created by the xgboost.

    Returns:
    -------
    score : float
        The score calculated according to the fitness_fn
    '''
    pred_train = model.predict(data_dict['dtrain'])
    pred_test = model.predict(data_dict['dtest'])
    kappa = global_settings['kappa']
    if global_settings['fitness_fn'] == 'd_roc':
        score = et.calculate_d_roc(
            data_dict, pred_train, pred_test, kappa=kappa)
    elif global_settings['fitness_fn'] == 'd_ams':
        score = et.calculate_d_ams(
            pred_train, pred_test, data_dict, kappa=kappa)
    else:
        print('This fitness_fn is not implemented')
    return score, pred_train, pred_test


def model_evaluation_main(hyperparameters, data_dict, global_settings):
    ''' Collected functions for CGB model evaluation

    Parameters:
    ----------
    hyperparamters : dict
        hyperparameters for the model to be created
    data_dict : dict
        Contains all the necessary information for the evaluation.
    global_settings : dict
        Preferences for the optimization

    Returns:
    -------
    score : float
        The score calculated according to the fitness_fn
    '''
    data_dict = create_xgb_data_dict(
        data_dict, global_settings['nthread']
    )
    model = create_model(
        hyperparameters, data_dict['dtrain'],
        global_settings['nthread']
    )
    score, pred_train, pred_test = evaluate_model(
        data_dict, global_settings, model)
    return score, pred_train, pred_test
