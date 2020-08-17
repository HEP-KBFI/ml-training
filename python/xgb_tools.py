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


def create_model(
        hyperparameters, data_dict, nthread,
        objective, weight
):
    train_data = data_dict['train']
    sum_wpos = sum(train_data.loc[train_data['target'] == 1, weight])
    sum_wneg = sum(train_data.loc[train_data['target'] == 0, weight])
    parameters = {
        'objective': 'binary:logistic',
        'scale_pos_weight': sum_wneg/sum_wpos,
        'eval_metric': objective,
        'silent': 1,
        'nthread': nthread
    }
    train_data = data_dict['train']
    trainvars = data_dict['trainvars']
    parameters.update(hyperparameters)
    classifier = xgb.XGBClassifier(**parameters)
    model = classifier.fit(
        train_data[trainvars],
        train_data['target'],
        sample_weight=train_data[weight]
    )
    return model


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
    trainvars = data_dict['trainvars']
    pred_train = model.predict_proba(data_dict['train'][trainvars])[:,1]
    pred_test = model.predict_proba(data_dict['test'][trainvars])[:,1]
    kappa = global_settings['kappa']
    if global_settings['fitness_fn'] == 'd_roc':
        score = et.calculate_d_roc(
            data_dict, pred_train, pred_test, kappa=kappa)
    elif global_settings['fitness_fn'] == 'd_ams':
        score = et.calculate_d_ams(
            pred_train, pred_test, data_dict, kappa=kappa)
    else:
        print('The' + str(global_settings['fitness_fn']) + \
            ' fitness_fn is not implemented'
        )
    return score, pred_train, pred_test


def model_evaluation_main(
        hyperparameters,
        data_dict,
        global_settings,
        objective='auc',
        weight='totalWeight'
):
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
    model = create_model(
        hyperparameters, data_dict, global_settings['nthread'], objective,
        weight
    )
    score, pred_train, pred_test = evaluate_model(
        data_dict, global_settings, model)
    return score, pred_train, pred_test
