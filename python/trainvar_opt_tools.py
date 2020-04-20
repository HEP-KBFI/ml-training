'''
Global functions for finding the optimal trainingvariables
'''
import json
import ROOT
from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import universal_tools as ut
import matplotlib.pyplot as plt
import glob
import os
import pandas
import csv
import numpy as np


def write_new_trainvar_list(trainvars, out_dir):
    '''Writes new trainvars to be tested into a file

    Parameters:
    ----------
    trainvars : list
        List of training variables to be outputted into a file
    out_file : str
        Path to the file of the trainvars

    Returns:
    -------
    Nothing
    '''
    out_file = os.path.join(out_dir, 'optimization_trainvars.txt')
    with open(out_file, 'w') as file:
        for trainvar in trainvars[:-1]:
            file.write(str(trainvar) + '\n')
        file.write(str(trainvars[-1]))


def choose_trainvar(datacard_dir, channel, trainvar, bdt_type):
    '''Reads the training variables from the data folder from file
    'optimization_trainvars.txt'. Is used for the xgb_tth cf function.

    Parametrs:
    ---------
    datacard_dir : dummy argument
        Needed for compability with the other trainvars loading
    channel : dummy argument
        Needed for compability with the other trainvars loading
    trainvar : dummy argument
        Needed for compability with texpandvarshe other trainvars loading
    bdt_type : dummy argument
        Needed for compability with the other trainvars loading

    Returns:
    -------
    trainvars : list
        list of trainvars that are to be used in the optimization.
    '''
    global_settings = ut.read_settings('global')
    out_dir = os.path.expandvars(global_settings['output_dir'])
    trainvars_path = os.path.join(
        out_dir,
        'optimization_trainvars.txt'
    )
    try:
        trainvars = dlt.read_list(trainvars_path)
    except:
        print('Could not find trainvars')
        trainvars = ''
    return trainvars


def initialize_trainvars(channel='2l_2tau', process='HH', random_sample='TTZ'):
    '''Reads in all the possible trainvars for initial run

    Parameters:
    ----------
    channel : str
        Name of the channel where the .root file is taken from (e.g 2l_2tau)
    process : str
        Name of the process where the .root is loaded (e.g. ttH or HH)
    random_sample : str
        A random sample the .root file tha is to be loaded belongs to

    Returns:
    trainvars : list
        list of all possible trainvars that are to be used in the optimization
    '''
    info_folder = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/info')
    inputpath_info_path = os.path.join(
        info_folder, process, channel, 'tauID_training.json')
    info_dict = ut.read_parameters(inputpath_info_path)[1]
    path_to_files = info_dict['inputPath']
    wildcard_root_files = os.path.join(
        path_to_files, '*' + random_sample + '*', 'central', '*.root')
    single_root_file = glob.glob(wildcard_root_files)[0]
    channel_info_path = os.path.join(
        info_folder, process, channel, 'info.json')
    channel_info_dict = ut.read_multiline_json_to_dict(channel_info_path)
    channel_in_tree = channel_info_dict['channelInTree']
    samplename_info = os.path.join(info_folder, 'samplename_info.json')
    global_settings = ut.read_settings('global')
    samplename_info = ut.read_parameters(samplename_info)
    folder_name = random_sample
    sample_dict = dlt.find_sample(folder_name, samplename_info)
    if sample_dict == {}:
        sample_dict = dl.advanced_sample_name(
            global_settings['bdtType'], folder_name, [])
    sample_name = sample_dict['sampleName']
    input_tree = str(os.path.join(
        channel_in_tree, 'sel/evtntuple', sample_name, 'evtTree'))
    trainvars = access_ttree(single_root_file, input_tree)
    trainvars = data_related_trainvars(trainvars)
    return trainvars


def data_related_trainvars(trainvars):
    '''Drops non-data-related trainvars like 'gen' and 'Weight'

    Parameters:
    ----------
    trainvars : list
        Not cleaned trainvariable list

    Returns:
    -------
    true_trainvars : list
        Updated trainvar list, that contains only data-related trainvars
    '''
    excluded_trainvars_path = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/settings/excluded_trainvars.txt'
    )
    false_trainvars = dlt.read_list(excluded_trainvars_path)
    true_trainvars = []
    for trainvar in trainvars:
        do_not_include = 0
        for false_trainvar in false_trainvars:
            if false_trainvar in trainvar:
                do_not_include += 1
        if do_not_include == 0:
            true_trainvars.append(trainvar)
    return true_trainvars


def access_ttree(single_root_file, input_tree):
    '''Accesses the TTree and gets all trainvars from the branches

    Parameters:
    ----------
    single_root_file : str
        Path to the .root file the TTree is located in
    inputTree : str
        Path of the TTree

    Returns:
    -------
    trainvars : list
        List of all branch names in the .root file
    '''
    trainvars = []
    tfile = ROOT.TFile(single_root_file)
    ttree = tfile.Get(input_tree)
    branches = ttree.GetListOfBranches()
    for branch in branches:
        trainvars.append(branch.GetName())
    return trainvars


def drop_worst_parameters(named_feature_importances):
    '''Drops the worst performing training variable

    Parameters:
    ----------
    named_feature_importances : dict
        Contains the trainvar and the corresponding 'gain' score

    Returns:
    -------
    trainvars : list
        list of trainvars with the worst performing one removed
    '''
    worst_performing_value = 10000
    worst_performing_feature = ""
    for trainvar in named_feature_importances:
        value = named_feature_importances[trainvar]
        if value < worst_performing_value:
            worst_performing_feature = trainvar
            worst_performing_value = value
    trainvars = named_feature_importances.keys()
    index = trainvars.index(worst_performing_feature)
    del trainvars[index]
    return trainvars, worst_performing_feature
