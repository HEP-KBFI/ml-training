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


def create_model(hyperparameters, dtrain, nthread, num_class):
    ''' Creates the XGBoost model given the with the given hyperparameters and
    training dataset

    Parameters:
    ----------
    hyperparameters : dict
        Dictionary containing all the wanted hyperparameters. Must contain
        at least 'num_boost_round' parameter.
    dtrain : xgboost.core.DMatrix
        XGB Dmatrix containing the dataset and the labels.

    Returns:
    --------
    model : XGBoost Booster
        The trained model
    '''
    params = {
        'silent': 1,
        'objective': 'multi:softprob',
        'num_class': num_class,
        'nthread': nthread,
        'seed': 1,
    }
    parameters = hyperparameters.copy()
    num_boost_round = parameters.pop('num_boost_round')
    parameters.update(params)
    model = xgb.train(
        parameters,
        dtrain,
        num_boost_round=int(num_boost_round),
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
        feature_names=data_dict['trainvars']
    )
    dtest = xgb.DMatrix(
        data_dict['testdataset'],
        label=data_dict['testing_labels'],
        nthread=nthread,
        feature_names=data_dict['trainvars']
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
    if global_settings['fitness_fn'] == 'd_roc':
        score = et.calculate_d_roc(data_dict, pred_train, pred_test)
    elif global_settings['fitness_fn'] == 'd_ams':
        score = et.calculate_d_ams(pred_train, pred_test, data_dict)
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
        global_settings['nthread'], global_settings['num_classes']
    )
    score, pred_train, pred_test = evaluate_model(
        data_dict, global_settings, model)
    return score, pred_train, pred_test
