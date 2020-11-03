'''Universal tools for file IO and other
'''
import shutil
import csv
import itertools
import json
import os
import numpy as np
import glob
from sklearn.metrics import confusion_matrix


def best_to_file(best_values, output_dir, assesment):
    '''Saves the best parameters and the scores to a file
    'best_parameters.json'

    Parameters:
    ----------
    best_values : dict
        Best parameters found during the evolutionary algorithm
    output_dir : str
        Directory where best parameters and assessment is saved
    assessment : dict
        Different scores for the best parameters found for both train and test
        dataset.
    '''
    output_path = os.path.join(output_dir, 'best_hyperparameters.json')
    with open(output_path, 'w') as file:
        json.dump(best_values, file)
        file.write('\n')
        json.dump(assesment, file)


def save_dict_to_json(dictionary, output_path):
    '''Saves the feature importances into a feature_importances.json file

    Parameters:
    ----------
    dictionary : dict
        Dicotionay to be saved
    output_dir : str
        Path to the output file

    Returns:
    -------
    Nothing
    '''
    with open(output_path, 'w') as out_file:
        json.dump(dictionary, out_file, indent=4)


def save_run_settings(output_dir):
    '''Saves the run settings for future reference

    Parameters:
    ----------
    output_dir : str
        Path to the output directory

    Returns:
    -------
    Nothing
    '''
    settings_dir = os.path.join(output_dir, 'run_settings')
    if not os.path.exists(settings_dir):
        os.makedirs(settings_dir)
    wild_card_path = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/settings/*'
    )
    for path in glob.glob(wild_card_path):
        shutil.copy(path, settings_dir)


def save_info_dir(output_dir):
    '''Saves the info dir for future reference

    Parameters:
    ----------
    output_dir : str
        Path to the output directory

    Returns:
    -------
    Nothing
    '''
    channel_dir, info_dir, global_settings = find_settings()
    run_info = os.path.join(output_dir, 'run_info')
    shutil.copytree(channel_dir, run_info)


def read_parameters(param_file):
    '''Read values form a '.json' file

    Parameters:
    ----------
    param_file : str
        Path to the '.json' file

    Returns:
    -------
    value_dicts : list containing dicts
        List of parameter dictionaries
    '''
    value_dicts = []
    with open(param_file, 'rt') as file:
        for line in file:
            json_dict = json.loads(line, object_hook=_decode_dict)
            value_dicts.append(json_dict)
    return value_dicts


def read_json_cfg(path):
    ''' Reads the json info from a given path

    Parameters:
    ----------
    path : str
        Path to the .json file

    Returns:
    --------
    info : dict
        The json dict that was loaded
    '''
    with open(path, 'rt') as jsonFile:
        info = json.load(jsonFile)
    return info


def read_settings(settings_dir, group):
    '''Function to read the global settings of the optimization

    Parameters:
    -----------
    group : str
        Group of settings wanted. Either: 'global', 'ga' or 'pso'

    Returns:
    --------
    settings_dict : dict
        Dictionary containing the settings for the optimization
    '''
    settings_path = os.path.join(
        settings_dir,
        group + '_settings.json')
    settings_dict = read_json_cfg(settings_path)
    return settings_dict


def _decode_list(data):
    '''Together with _decode_dict are meant to avoid unicode key and value
    pairs, due to the problem of ROOT not being able to load the parth when it
    is unicode. See https://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-from-json
    '''
    rv = []
    for item in data:
        if isinstance(item, unicode):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _decode_list(item)
        elif isinstance(item, dict):
            item = _decode_dict(item)
        rv.append(item)
    return rv


def _decode_dict(data):
    '''Together with _decode_list are meant to avoid unicode key and value
    pairs, due to the problem of ROOT not being able to load the parth when it
    is unicode. See https://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-from-json
    '''
    rv = {}
    for key, value in data.iteritems():
        if isinstance(key, unicode):
            key = key.encode('utf-8')
        if isinstance(value, unicode):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = _decode_list(value)
        elif isinstance(value, dict):
            value = _decode_dict(value)
        rv[key] = value
    return rv


def find_settings():
    ''' Gets info directory path and returns it together with the global
    settings

    Parameters:
    ----------
    None

    Returns:
    -------
    channel_dir : str
        Path to the global channel directory
    info_dir : str
        Path to the info directory of the specified channel
    global_settings: dict
        global settings for the training
    '''
    package_dir = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/'
    )
    settings_dir = os.path.join(package_dir, 'settings')
    global_settings = read_settings(settings_dir, 'global')
    channel = global_settings['channel']
    process = global_settings['process']
    mode = create_infoPath_addition(global_settings)
    channel_dir = os.path.join(package_dir, 'info', process, channel)
    info_dir =  os.path.join(channel_dir, mode)
    return channel_dir, info_dir, global_settings


def create_infoPath_addition(global_settings):
    if 'nonres' in global_settings['bdtType']:
        mode = 'nonRes'
    else:
        spinCase = global_settings['spinCase']
        mode = '/'.join(['res', spinCase])
    return mode
