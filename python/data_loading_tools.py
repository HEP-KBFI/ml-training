from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_data_tools as hhdt
from machineLearning.machineLearning import tth_aux_tools as tthat
import machineLearning.machineLearning as ml
import os
import json
from pathlib import Path
import os
import ROOT
import pandas
import glob
import numpy as np
from root_numpy import tree2array


def load_data(
        preferences,
        global_settings,
        remove_neg_weights=True
):
    '''
    Loads the data for all wanted eras

    Parameters:
    ----------
    preferences : dict
        dictionary conaining the preferences
    global_settings : dict
        global settings for the training
    [remove_neg_weights = True] : bool
        Whether to remove negative weight events from the data

    Returns:
    -------
    data : pandas DataFrame
        DataFrame containing the data from all the wanted eras
    '''
    eras = preferences['included_eras']
    total_data = pandas.DataFrame({})
    for era in eras:
        input_path_key = 'inputPath' + str(era)
        preferences['era_inputPath'] = preferences[input_path_key]
        preferences['era_keys'] = preferences['keys' + str(era)]
        data = load_data_from_one_era(
            global_settings,
            preferences,
            remove_neg_weights
        )
        data['era'] = era
        total_data = total_data.append(data)
    if global_settings['dataCuts'] != 0:
        data = data_cutting(data, global_settings)
    return total_data


def load_data_from_one_era(
        global_settings,
        preferences,
        remove_neg_weights,
):
    '''Loads the all the necessary data

    Parameters:
    ----------
    preferences : dict
        dictionary conaining the preferences
    global_settings : dict
        global settings for the training
    remove_neg_weights : bool
        Whether to remove negative weight events from the data

    Returns:
    --------
    data : pandas DataFrame
        All the loaded data so far.
    '''
    print_info(global_settings, preferences)
    my_cols_list = preferences['trainvars'] + ['process', 'key', 'target', 'totalWeight']
    if 'HH_nonres' in global_settings['bdtType']:
        my_cols_list += ['nodeX']
    data = pandas.DataFrame(columns=my_cols_list)
    for folder_name in preferences['era_keys']:
        data = data_main_loop(
            folder_name,
            global_settings,
            preferences,
            data
        )
        signal_background_calc(data, global_settings['bdtType'], folder_name)
    n = len(data)
    nS = len(data.ix[data.target.values == 1])
    nB = len(data.ix[data.target.values == 0])
    print('For ' + preferences['channelInTree'] + ':')
    print('\t Signal: ' + str(nS))
    print('\t Background: ' + str(nB))
    if remove_neg_weights:
        print('Removing events with negative weights')
        data = remove_negative_weight_events(data, weights='totalWeight')
    return data


def data_main_loop(
        folder_name,
        global_settings,
        preferences,
        data
):
    ''' Defines new variables based on the old ones that can be used in the
    training

    Parameters:
    ----------
    folder_name : str
        Folder from the keys where .root file is searched.
    preferences : dict
        dictionary conaining the preferences
    global_settings : dict
        global settings for the training
    data : str
        The loaded data so far

    Returns:
    -------
    data : pandas DataFrame
        All the loaded data so far.
    '''
    sample_name, target = find_sample_info(
        folder_name, global_settings['bdtType'], preferences['masses']
    )
    print(':::::::::::::::::')
    print('input_path:\t' + str(preferences['era_inputPath']))
    print('folder_name:\t' + str(folder_name))
    print('channelInTree:\t' + str(preferences['channelInTree']))
    input_tree = str(os.path.join(
        preferences['channelInTree'], 'sel/evtntuple', sample_name, 'evtTree'))
    paths = get_all_paths(
        preferences['era_inputPath'], folder_name, global_settings['bdtType']
    )
    for path in paths:
        if 'nonres' in global_settings['bdtType'] and 'nonresonant' in path:
            target = 1
            sample_name = 'HH_nonres_decay'
            input_tree = create_input_tree_path(
                path, preferences['channelInTree']
            )
        print('Loading from: ' + path)
        tree, tfile = read_root_tree(path, input_tree)
        data = load_data_from_tfile(
            tree,
            tfile,
            sample_name,
            folder_name,
            target,
            preferences,
            global_settings,
            data,
        )
    return data


def load_data_from_tfile(
        tree,
        tfile,
        sample_name,
        folder_name,
        target,
        preferences,
        global_settings,
        data,
):
    ''' Loads data from the ROOT TTree.

    Parameters:
    ----------
    tree : ROOT.TTree
        The tree from .root file
    tfile : ROOT.Tfile
        The loaded .root file
    sample_name : str
        For each folder seperate sample name as defined in samplename_info.json
    folder_name : str
        Folder from the keys where .root file is searched.
    target : int
        [0 or 1] Either signal (1) or background (0)
    preferences : dict
        dictionary conaining the preferences
    global_settings : dict
        global settings for the training
    data : pandas DataFrame
        All the loaded data will be appended there.

    Returns:
    -------
    data : pandas DataFrame
        All the loaded data so far.
    '''
    if tree is not None:
        try:
            chunk_arr = tree2array(tree)
            chunk_df = pandas.DataFrame(
                chunk_arr)
            tfile.Close()
        except Exception:
            print(
                'Error: Failed to load TTree in the file ' + str(tfile))
        else:
            data = define_new_variables(
                chunk_df,
                sample_name,
                folder_name,
                target,
                preferences,
                global_settings,
                data,
            )
    else:
        tfile.Close()
        print('Error: empty path')
    return data


def define_new_variables(
        chunk_df,
        sample_name,
        folder_name,
        target,
        preferences,
        global_settings,
        data
):
    ''' Defines new variables based on the old ones that can be used in the
    training

    Parameters:
    ----------
    chunk_df : pandas DataFrame
        DataFrame containing only columns that are specified in trainvars +
        evtWeight. The data is read from .root file.
    sample_name : str
        For each folder seperate sample name as defined in samplename_info.json
    folder_name : str
        Folder from the keys where .root file is searched.
    target : int
        [0 or 1] Either signal (1) or background (0)
    preferences : dict
        dictionary conaining the preferences
    global_settings : dict
        global settings for the training
    data : pandas DataFrame
        All the loaded data will be appended there.

    Returns:
    -------
    data : pandas DataFrame
        All the loaded data so far.
    '''
    chunk_df['process'] = sample_name
    chunk_df['key'] = folder_name
    chunk_df['target'] = int(target)
    chunk_df['totalWeight'] = chunk_df['evtWeight']
    if 'HH' in global_settings['bdtType']:
        data = hhdt.define_new_variables(
            chunk_df, sample_name, folder_name, target, preferences,
            global_settings, data
        )
    else:
        print('Data choosing not implemented for this bdtType')
    return data


def read_root_tree(path, input_tree):
    '''Reads the .root file with a given path and loads the given input tree

    Parameters:
    ----------
    path : str
        Path to the .root file
    input_tree : str
        Path to the correct tree

    Returns:
    -------
    tree : ROOT.TTree
        The tree from .root file
    tfile : ROOT.Tfile
        The loaded .root file
    '''
    try:
        tfile = ROOT.TFile(path)
    except Exception:
        print('Error: No ".root" file with the path ' + str(path))
    try:
        tree = tfile.Get(input_tree)
    except Exception:
        print('Error: Failed to read TTree ' + str(input_tree))
    return tree, tfile


def get_all_paths(input_path, folder_name, bdt_type):
    ''' Gets all the paths of the .root files which should be included
    in the data dataframe

    Parameters:
    ----------
    input_path : str
        Master-folder where the .root files are contained
    folder_name : str
        Name of the folder in the keys currently loading the data from
    bdt_type: str
        Type of the boosted decision tree (?)

    Returns:
    -------
    paths : list
        List of all the paths for the .root files containing the data to be
        included in the dataframe.
    '''
    if 'TTH' in bdt_type:
        paths = tthat.get_ntuple_paths(input_path, folder_name, bdt_type)
    elif 'HH' in bdt_type:
        paths = hhdt.get_ntuple_paths(input_path, folder_name, bdt_type)
    else:
        return ValueError('Unknown bdtType')
    return paths


def find_sample_info(folder_name, bdt_type, masses):
    samplename_info_path = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/info/samplename_info.json'
    )
    samplename_info = ut.read_json_cfg(samplename_info_path)
    if 'HH' in bdt_type:
        sample_name, target = hhdt.set_sample_info(
            folder_name, samplename_info, masses, bdt_type
        )
    elif 'TTH' in bdt_type:
        sample_name, target = tthat.set_sample_info(
            folder_name, samplename_info, masses, bdt_type
        )
    else:
        raise ValueError("Unknown bdtType")
    return sample_name, target


def signal_background_calc(data, bdt_type, folder_name):
    '''Calculates the signal and background

    Parameters:
    ----------
    data : pandas DataFrame
        data that is loaded
    bdt_type: str
        Type of the boosted decision tree (?)
    folder_name : str
        Name of the folder in the keys currently loading the data from

    Returns:
    -------
    Nothing
    '''
    try:
        nS = len(
            data.loc[(data.target.values == 1) & (data.key.values == folder_name)])
        nB = len(
            data.loc[(data.target.values == 0) & (data.key.values == folder_name)])
        nNW = len(
            data.loc[
                (data['totalWeight'].values < 0) & (
                    data.key.values == folder_name)])
        print('Signal: ' + str(nS))
        print('Background: ' + str(nB))
        print('Event weight: ' + str(data.loc[
                (data.key.values == folder_name)]['evtWeight'].sum()))
        print('Total data weight: ' + str(data.loc[
                (data.key.values == folder_name)]['totalWeight'].sum()))
        print('Events with negative weights: ' + str(nNW))
        print(':::::::::::::::::')
    except:
        if len(data) == 0:
            print('Error: No data (!!!)')


def load_era_keys(info):
    samples = info['keys'].keys()
    era_wise_keys = {'keys16': [], 'keys17': [], 'keys18': []}
    for sample in samples:
        included_eras = info['keys'][sample]
        if 16 in included_eras:
            era_wise_keys['keys16'].append(sample)
        if 17 in included_eras:
            era_wise_keys['keys17'].append(sample)
        if 18 in included_eras:
            era_wise_keys['keys18'].append(sample)
    return era_wise_keys


def find_input_paths(
        info_dict,
        tau_id_training,
):
    '''Finds era-wise inputPaths

    Parameters:
    ----------
    info_dict : dict
        Dict from info.json file
    tau_id_trainings : list of dicts
        info about all tau_id_training for inputPaths
    tau_id_training : str
        Value of tau_id_training node

    Returns:
    -------
    input_paths_dict : dict
        Dict conaining the inputPaths for all eras
    '''
    eras = info_dict['included_eras']
    input_paths_dict = {}
    tauID_training = info_dict['tauID_training'][tau_id_training]
    for era in eras:
        key = 'inputPath' + era
        input_paths_dict[key] = tauID_training[key]
    return input_paths_dict


def find_correct_dict(key, value, list_of_dicts):
    '''Finds the correct dictionary based on the requested key

    Parameters:
    ----------
    key : str
        Name of the key to find
    value: str
        Value the requested key should have
    list_of_dicts : list
        Contains dictionaries to be parsed

    Returns:
    -------
    requested_dict : dict
    '''
    new_dictionary = {}
    for dictionary in list_of_dicts:
        if dictionary[key] == value:
            new_dictionary = dictionary.copy()
            new_dictionary.pop(key)
    if new_dictionary == {}:
        print(
            'Given parameter for ' + str(key) + ' missing. Using the defaults')
        for dictionary in list_of_dicts:
            if dictionary[key] == 'default':
                new_dictionary = dictionary.copy()
                new_dictionary.pop(key)
    return new_dictionary


def read_list(path):
    ''' Creates a list from values on different rows in a file

    Parameters:
    ----------
    path : str
        Path to the file to be read

    Returns:
    -------
    parameters : list
        List of all the values read from the file with the given path
    '''
    parameters = []
    with open(path, 'r') as f:
        for line in f:
            parameters.append(line.strip('\n'))
    return parameters


def print_info(global_settings, preferences):
    '''Prints the data loading preferences info'''
    print('In data_manager')
    print(':::: Loading data ::::')
    print('inputPath: ' + str(preferences['era_inputPath']))
    print('channelInTree: ' + str(preferences['channelInTree']))
    print('-----------------------------------')
    print('variables:')
    print_columns(preferences['trainvars'])
    print('bdt_type: ' + str(global_settings['bdtType']))
    print('channel: ' + str(global_settings['channel']))
    print('keys: ')
    print_columns(preferences['era_keys'])
    print('masses: ' + str(preferences['masses']))
    print('mass_randomization: ' + str(global_settings['bkg_mass_rand']))


def print_columns(to_print):
    '''Prints the list into nice two columns'''
    to_print = sorted(to_print)
    if len(to_print) % 2 != 0:
        to_print.append(' ')
    split = int(len(to_print)/2)
    l1 = to_print[0:split]
    l2 = to_print[split:]
    print('-----------------------------------')
    for one, two in zip(l1, l2):
        print('{0:<45s} {1}'.format(one, two))
    print('-----------------------------------')


def create_input_tree_path(filename, channel_in_tree):
    '''Constructs the input tree path based on the filename and the
    channelInTree

    Parameters:
    -----------
    filename : str
        name (or path) of the file
    channel_in_tree : str
        Naame of the parent folder in the .root file

    Returns:
    -------
    input_tree : str
        Path in the .root file where data is located
    '''
    if '4v' in filename:
        addition = 'wwww'
    if '2v2t' in filename:
        addition = 'wwtt'
    if '4t' in filename:
        addition = 'tttt'
    if '2b2v_sl' in filename:
        addition = '2b2v_sl'
    elif '2b2v' in filename:
        addition = '2b2v'
    name = '_'.join(['signal_ggf_nonresonant_hh', addition])
    input_tree = os.path.join(
        channel_in_tree, 'sel/evtntuple', name, 'evtTree')
    return str(input_tree)


def remove_negative_weight_events(data, weights='totalWeight'):
    '''Removes negative weight events from the data dataframe

    Parameters:
    ----------
    data : pandas DataFrame
        The data that was loaded
    weights : str
        The name of the column in the dataframe to be used as the weight.

    Returns:
    -------
    new_data : pandas DataFrame
        Data that was loaded without negative weight events
    '''
    new_data = data.loc[data[weights] >= 0]
    return new_data


def read_trainvar_info(path):
    '''Reads the trainvar info

    Parameters:
    -----------
    path : str
        Path to the training file

    Returns:
    -------
    trainvar_info : dict
        Dictionary containing trainvar info (e.g is the trainvar supposed to
        be an integer or not)
    '''
    trainvar_info = {}
    info_dicts = ut.read_parameters(path)
    for single_dict in info_dicts:
        trainvar_info[single_dict['key']] = single_dict['true_int']
    return trainvar_info


def data_cutting(data, global_settings):
    package_dir = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/'
    )
    if 'nonres' in global_settings['bdtType']:
        addition = 'nonRes'
    else:
        addition = 'res/%s' %(global_settings['spinCase'])
    cut_file = os.path.join(
        package_dir, 'info', global_settings['process'],
        global_settings['channel'], addition, 'cuts.json'
    )
    if os.path.exists(cut_file):
        cut_dict = ut.read_json_cfg(cut_file)
        if cut_dict == {}:
            print('No cuts given in the cut file %s' %(cut_file))
        else:
            cut_keys = list(cut_dict.keys())
            for key in cut_keys:
                try:
                    min_value = cut_dict[key]['min']
                    data = data.loc[(data[key] >= min_value)]
                except KeyError:
                    print('Minimum condition for %s not implemented' %(key))
                try:
                    max_value = cut_dict[key]['max']
                    data = data.loc[(data[key] <= max_value)]
                except KeyError:
                    print('Maximum condition for %s not implemented' %(key))
    else:
        print('Cut file %s does not exist' %(cut_file))
    return data
