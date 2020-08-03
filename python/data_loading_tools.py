from machineLearning.machineLearning import universal_tools as ut
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
        input_path_key = 'inputPath' + era
        input_path = preferences[input_path_key]
        data = load_data_from_one_era(
            input_path,
            preferences['channelInTree'],
            preferences['trainvars'],
            global_settings['bdtType'],
            global_settings['channel'],
            preferences['keys'],
            preferences['masses'],
            preferences['nonResScenarios'],
            global_settings['bkg_mass_rand']
        )
        data['era'] = era
        total_data = total_data.append(data)
    return total_data


def load_data_from_one_era(
        input_path,
        channel_in_tree,
        variables,
        bdt_type,
        channel,
        keys,
        masses=[],
        nonResScenarios=[],
        mass_randomization='default',
        remove_neg_weights=True,
):
    '''Loads the all the necessary data

    Parameters:
    ----------
    input_path : str
        folder where all the .root files are taken from
    channel_in_tree : str
        path where data is located in the .root file
    variables : list
        List of training variables to be used.
    bdt_type : str
        Boosted decision tree type
    channel : str
        What channel data is to be loaded
    keys : list
        Which keys to be included in the data
        (part included in the folder names)
    masses : list
        List of the masses to be used in data loading
    mass_randomization : str
        Which kind of mass randomization to use: "default" or "oversampling"

    Returns:
    --------
    data : pandas DataFrame
        All the loaded data so far.
    '''
    print_info(
        input_path, channel_in_tree, variables,
        bdt_type, channel, keys, masses, mass_randomization
    )
    my_cols_list = variables + ['process', 'key', 'target', 'totalWeight']
    if 'HH_nonres' in bdt_type:
        my_cols_list += ['nodeX']
    data = pandas.DataFrame(columns=my_cols_list)
    samplename_info_path = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/info/samplename_info.json'
    )
    samplename_info = ut.read_parameters(samplename_info_path)
    for folder_name in keys:
        data = data_main_loop(
            folder_name,
            samplename_info,
            channel_in_tree,
            masses,
            nonResScenarios,
            input_path,
            bdt_type,
            data,
            variables,
            mass_randomization,
        )
        signal_background_calc(data, bdt_type, folder_name)
    n = len(data)
    nS = len(data.ix[data.target.values == 1])
    nB = len(data.ix[data.target.values == 0])
    print('For ' + channel_in_tree + ':')
    print('\t Signal: ' + str(nS))
    print('\t Background: ' + str(nB))
    if remove_neg_weights:
        print('Removing events with negative weights')
        data = remove_negative_weight_events(data, weights='totalWeight')
    return data


def data_main_loop(
        folder_name,
        samplename_info,
        channel_in_tree,
        masses,
        nonResScenarios,
        input_path,
        bdt_type,
        data,
        variables,
        mass_randomization,
):
    ''' Defines new variables based on the old ones that can be used in the
    training

    Parameters:
    ----------
    folder_name : str
        Folder from the keys where .root file is searched.
    sample_name_info : dict
        dictionary containing info which samples are signal and which
        background.
    channel_in_tree : str
        path where data is located in the .root file
    masses : list
        List of the masses to be used in data loading
    input_path : str
        folder where all the .root files are taken from
    bdt_type : str
        Boosted decision tree type
    data : str
        The loaded data so far
    variables : list
        List of training variables to be used.
    mass_randomization : str
        Which kind of mass randomization to use: "default" or "oversampling"

    Returns:
    -------
    data : pandas DataFrame
        All the loaded data so far.
    '''
    sample_dict = find_sample(folder_name, samplename_info)
    if sample_dict == {}:
        sample_dict = advanced_sample_name(
            bdt_type, folder_name, masses
        )
    sample_name = sample_dict['sampleName']
    target = sample_dict['target']
    print(':::::::::::::::::')
    print('input_path:\t' + str(input_path))
    print('folder_name:\t' + str(folder_name))
    print('channelInTree:\t' + str(channel_in_tree))
    input_tree = str(os.path.join(
        channel_in_tree, 'sel/evtntuple', sample_name, 'evtTree'))
    paths = get_all_paths(input_path, folder_name, bdt_type)
    for path in paths:
        node_x = 'NonNode'
        if 'nonres' in bdt_type and 'nonresonant' in path:
            target = 1
            sample_name = 'HH_nonres_decay'
            input_tree = create_input_tree_path(path, channel_in_tree)
            node_x = get_node_nr(path)
        print('Loading from: ' + path)
        tree, tfile = read_root_tree(path, input_tree)
        data = load_data_from_tfile(
            tree,
            tfile,
            sample_name,
            folder_name,
            target,
            bdt_type,
            masses,
            mass_randomization,
            nonResScenarios,
            data,
            variables,
            node_x
        )
    return data


def load_data_from_tfile(
        tree,
        tfile,
        sample_name,
        folder_name,
        target,
        bdt_type,
        masses,
        mass_randomization,
        nonResScenarios,
        data,
        variables,
        node_x
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
    bdt_type : str
        Boosted decision tree type
    masses : list
        List of the masses to be used in data loading
    mass_randomization : str
        Which kind of mass randomization to use: "default" or "oversampling"
    data : pandas DataFrame
        All the loaded data will be appended there.
    variables : list
        List of training variables to be used.
    node_x : str
        Name of the node. Used with HH_nonres type bdtType

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
                bdt_type,
                masses,
                mass_randomization,
                nonResScenarios,
                data,
                variables,
                node_x
            )
    else:
        print('Error: empty path')
    return data


def define_new_variables(
        chunk_df,
        sample_name,
        folder_name,
        target,
        bdt_type,
        masses,
        mass_randomization,
        nonResScenarios,
        data,
        variables,
        node_x
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
    bdt_type : str
        Boosted decision tree type
    masses : list
        List of the masses to be used in data loading
    mass_randomization : str
        Which kind of mass randomization to use: "default" or "oversampling"
    data : pandas DataFrame
        All the loaded data will be appended there.
    node_x : str
        Name of the node. Used with HH_nonres type bdtType

    Returns:
    -------
    data : pandas DataFrame
        All the loaded data so far.
    '''
    chunk_df['process'] = sample_name
    chunk_df['key'] = folder_name
    chunk_df['target'] = target
    chunk_df['totalWeight'] = chunk_df['evtWeight']
    if "nonores" in bdt_type:
        chunk_df['nodeX'] = node_x
    if "HH_bb2l" in bdt_type:
        chunk_df["max_dR_b_lep"] = chunk_df[
            ["dR_b1lep1", "dR_b2lep1", "dR_b2lep1", "dR_b2lep2"]
        ].max(axis=1)
        chunk_df["max_lep_pt"] = chunk_df[["lep1_pt", "lep2_pt"]].max(axis=1)
    if "HH_bb1l" in bdt_type:
        chunk_df["max_dR_b_lep"] = chunk_df[
            ["dR_b1lep", "dR_b2lep"]].max(axis=1)
        chunk_df["max_bjet_pt"] = chunk_df[
            ["bjet1_pt", "bjet2_pt"]].max(axis=1)
    if (("HH_0l_2tau" in bdt_type) or ("HH_2l_2tau" in bdt_type)):
        chunk_df["tau1_eta"] = abs(chunk_df["tau1_eta"])
        chunk_df["tau2_eta"] = abs(chunk_df["tau2_eta"])
        chunk_df["max_tau_eta"] = chunk_df[
            ["tau1_eta", "tau2_eta"]].max(axis=1)
    if "HH_2l_2tau" in bdt_type:
        chunk_df["min_dr_lep_tau"] = chunk_df[
            ["dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2"]
        ].min(axis=1)
        chunk_df["max_dr_lep_tau"] = chunk_df[
            ["dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2"]
        ].max(axis=1)
        chunk_df["lep1_eta"] = abs(chunk_df["lep1_eta"])
        chunk_df["lep2_eta"] = abs(chunk_df["lep2_eta"])
        chunk_df["max_lep_eta"] = chunk_df[
            ["lep1_eta", "lep2_eta"]].max(axis=1)
        chunk_df["sum_lep_charge"] = sum(
            [chunk_df["lep1_charge"], chunk_df["lep2_charge"]]
        )
    HHRes = ('HH' in bdt_type) and 'nonres' not in bdt_type
    if HHRes and "gen_mHH" in variables:
        if target == 1:
            for mass in masses:
                if str(mass) in folder_name:
                    chunk_df["gen_mHH"] = float(mass)
        elif target == 0:
            if mass_randomization == "default":
                chunk_df["gen_mHH"] = float(np.random.choice(
                    masses, size=len(chunk_df)))
            elif mass_randomization == "oversampling":
                for mass in masses:
                    chunk_df["gen_mHH"] = float(mass)
                    data = data.append(chunk_df, ignore_index=True, sort=False)
            else:
                raise ValueError(
                    'Cannot use ', mass_randomization, "as mass_randomization")
        else:
            raise ValueError('Cannot use ', target, 'as target')
    elif "nonres" in bdt_type:
        for i in range(len(nonResScenarios)):
            chunk_df_node = chunk_df.copy()
            scenario = nonResScenarios[i]
            chunk_df_node['nodeX'] = i
            chunk_df_node['nodeXname'] = scenario
            if target == 1:
                if scenario is not "SM":
                    nodeWeight = chunk_df_node['weight_' + scenario]
                    nodeWeight /= chunk_df_node['weight_SM']
                    chunk_df_node['totalWeight'] *= nodeWeight
            data = data.append(chunk_df_node, ignore_index=True, sort=False)
        return data
    case1 = mass_randomization != "oversampling"
    case2 = mass_randomization == "oversampling" and target == 1
    if case1 or case2:
        data = data.append(chunk_df, ignore_index=True, sort=False)
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
        if folder_name == 'ttHToNonbb':
            wild_card_path = os.path.join(
                input_path, folder_name + '_M125_powheg',
                folder_name + '*.root'
            )
            paths = glob.glob(wild_card_path)
        elif ('TTW' in folder_name) or ('TTZ' in folder_name):
            wild_card_path = os.path.join(
                input_path, folder_name + '_LO*', folder_name + '*.root')
            paths = glob.glob(wild_card_path)
        else:
            if 'ttH' in folder_name:
                wild_card_path = os.path.join(
                    input_path,
                    folder_name + '*Nonbb*',
                    'central',
                    folder_name + '*.root'
                )
                paths = glob.glob(wild_card_path)
            wild_card_path = os.path.join(
                input_path, folder_name + "*", folder_name + '*.root')
            paths = glob.glob(wild_card_path)
            if len(paths) == 0:
                wild_card_path = os.path.join(
                    input_path, folder_name + '*', '*.root')
                paths = glob.glob(wild_card_path)
    else:
        paths = []
        sample_categories = [{}]
        if 'HH' in bdt_type:  # hardcoded path
            catfile = os.path.join(
                os.path.expandvars('$CMSSW_BASE'),
                'src/machineLearning/machineLearning/info',
                'HH',
                'sample_categories.json'
            )
            sample_categories = ut.read_parameters(catfile)
        if (folder_name in sample_categories[0].keys()):
            for fname in sample_categories[0][folder_name]:
                wild_card_path = os.path.join(
                    input_path, fname + '*', 'central', '*.root')
                addpaths = glob.glob(wild_card_path)
                if len(addpaths) == 0:
                    wild_card_path = os.path.join(
                        input_path, fname + '*', '*.root')
                    addpaths = glob.glob(wild_card_path)
                paths.extend(addpaths)
            paths = list(dict.fromkeys(paths))
        else:
            wild_card_path = os.path.join(
                input_path, folder_name + '*', 'central', '*.root')
            paths = glob.glob(wild_card_path)
            if len(paths) == 0:
                wild_card_path = os.path.join(
                    input_path, folder_name + '*', '*.root')
                paths = glob.glob(wild_card_path)
        paths = [path for path in paths if 'hadd' not in path]
    return paths


def advanced_sample_name(bdt_type, folder_name, masses):
    ''' If easily classified samples are not enough to get 'target' and
    'sample_name'

    Parameters:
    ----------
    bdt_type: str
        Type of the boosted decision tree (?)
    folder_name : str
        Name of the folder in the keys currently loading the data from
    masses : list
        list of int. Masses that are used

    Returns:
    -------
    sample_dict : dict
        Dictionary containing 'sampleName' and 'target'
    '''
    if 'evtLevelSUM_HH_bb2l' in bdt_type or 'evtLevelSUM_HH_bb1l' in bdt_type:
        if 'signal_ggf' in folder_name:
            if 'evtLevelSUM_HH_bb2l_res' in bdt_type:
                sample_name = 'signal_ggf_spin0'
            else:
                sample_name = 'signal_ggf_nonresonant_node'
            for mass in masses:
                if mass == 20:
                    sample_name = sample_name + '_' + 'sm' + '_'
                    break
                elif '_' + str(mass) + '_' in folder_name:
                    sample_name = sample_name + '_' + str(mass) + '_'
                    break
            if '_2b2v' in folder_name:
                sample_name = sample_name + 'hh_bbvv'
            target = 1
    elif 'HH' in bdt_type:
        isSpin0 = 'signal_ggf_spin0' in folder_name
        isSpin2 = 'signal_ggf_spin2' in folder_name
        if isSpin0 or isSpin2:
            if 'signal_ggf_spin0' in folder_name:
                sample_name = 'signal_ggf_spin0_'
            elif 'signal_ggf_spin2' in folder_name:
                sample_name = 'signal_ggf_spin2_'
            for mass in masses:
                if str(mass) in folder_name:
                    sample_name = sample_name + str(mass)
            if '_4t' in folder_name:
                sample_name = sample_name + '_hh_tttt'
            if '_4v' in folder_name:
                sample_name = sample_name + '_hh_wwww'
            if '_2v2t' in folder_name:
                sample_name = sample_name + '_hh_wwtt'
            target = 1
    if 'ttH' in folder_name:
        if 'HH' in bdt_type:
            target = 0
            sample_name = 'TTH'
        else:
            target = 1
            sample_name = 'ttH'  # changed from 'signal'
    if 'nonres' in bdt_type:
        target = 1
        sample_name = "signal_ggf_nonresonant"
        if '_4t' in folder_name:
            sample_name = sample_name + '_hh_tttt'
        if '_4v' in folder_name:
            sample_name = sample_name + '_hh_wwww'
        if '_2v2t' in folder_name:
            sample_name = sample_name + '_hh_wwtt'
        #  sample_name = 'HH_nonres_decay'
    sample_dict = {
        'sampleName': sample_name,
        'target': target
    }
    return sample_dict


def find_sample(folder_name, samplename_info):
    '''Finds which sample corresponds to the given folder name

    Parameters:
    ----------
    folder_name : str
        Name of the folder where data would be loaded
    samplename_info : dict
        Info regarding what sample name and target each folder has

    Returns:
    -------
    sample_dict : dict
        Dictionary containing the info of the sample for the folder.
    '''
    sample_dict = {}
    for sample in samplename_info:
        if sample['type'] in folder_name:
            sample_dict = sample
    return sample_dict


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
    if len(data) == 0:
        print('Error: No data (!!!)')
    if 'evtLevelSUM_HH_bb2l' in bdt_type and folder_name == 'TTTo2L2Nu':
        data.drop(data.tail(6000000).index, inplace=True)
    elif 'evtLevelSUM_HH_bb1l' in bdt_type:
        if folder_name == 'TTToSemiLeptonic_PSweights':
            data.drop(data.tail(24565062).index, inplace=True)
        if folder_name == 'TTTo2L2Nu_PSweights':
            data.drop(data.tail(11089852).index, inplace=True)  # 12089852
        if folder_name.find('signal') != -1:
            if folder_name.find('900') == -1 and folder_name.find('1000') == -1:
                data.drop(data.tail(15000).index, inplace=True)
            if bdt_type.find('nonres') != -1:
                data.drop(data.tail(20000).index, inplace=True)
        elif folder_name == 'W':
            data.drop(data.tail(2933623).index, inplace=True)
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
    print('events with -ve weights: ' + str(nNW))
    print(':::::::::::::::::')


def reweigh_dataframe(
        data,
        weight_files_dir,
        trainvar_info,
        cancelled_trainvars,
        masses,
        skip_int_vars=True
):
    '''Reweighs the dataframe

    Parameters:
    ----------
    data : pandas Dataframe
        Data to be reweighed
    weighed_files_dir : str
        Path to the directory where the reweighing files are
    trainvar_info : dict
        Dictionary containing trainvar info (e.g is the trainvar supposed to
        be an integer or not)
    cancelled_trainvars :list
        list of trainvars not to include
    masses : list
        list of masses

    Returns:
    -------
    Nothing
    '''
    trainvars = list(trainvar_info.keys())
    for trainvar in trainvars:
        if trainvar in cancelled_trainvars:
            continue
        filename = '_'.join(['TProfile_signal_fit_func', trainvar]) + '.root'
        file_path = os.path.join(weight_files_dir, filename)
        tfile = ROOT.TFile.Open(file_path)
        fit_function_name = '_'.join(['fitFunction', trainvar])
        function = tfile.Get(fit_function_name)
        if bool(trainvar_info[trainvar]) and skip_int_vars:
            data[trainvar] = data[trainvar].astype(int)
            continue
        for mass in masses:
            data.loc[
                data['gen_mHH'] == mass, [trainvar]] /= function.Eval(mass)
        tfile.Close()


def get_hh_parameters(
        channel,
        tau_id_training,
        channel_dir
):
    '''Reads the parameters for HH data loading

    Parameters:
    ----------
    channel : str
        The name of the channel (e.g. 2l_2tau)
    tau_id_training : str
        Tau ID for training
    channel_dir : str
        Path to the "info" firectory of the current run

    Returns:
    --------
    parameters : dict
        The necessary info for loading data
    '''
    info_path = os.path.join(channel_dir, 'info.json')
    keys_path = os.path.join(channel_dir, 'keys.txt')
    tau_id_application_path = os.path.join(
        channel_dir, 'tauID_application.json')
    tau_id_training_path = os.path.join(channel_dir, 'tauID_training.json')
    trainvars_path = os.path.join(channel_dir, 'trainvars.json')
    info_dict = ut.read_multiline_json_to_dict(info_path)
    tau_id_trainings = ut.read_parameters(tau_id_training_path)
    tau_id_applications = ut.read_parameters(tau_id_application_path)
    parameters = find_correct_dict(
        'tauID_application',
        info_dict['default_tauID_application'],
        tau_id_applications
    )
    parameters.update(find_input_paths(
        info_dict, tau_id_trainings, tau_id_training))
    parameters['keys'] = read_list(keys_path)
    trainvar_info = read_trainvar_info(trainvars_path)
    parameters['trainvars'] = list(trainvar_info.keys())
    parameters['trainvar_info'] = trainvar_info
    parameters.update(info_dict)
    return parameters


def find_input_paths(
        info_dict,
        tau_id_trainings,
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
    correct_dict = find_correct_dict(
        'tauID_training', tau_id_training, tau_id_trainings)
    for era in eras:
        key = 'inputPath' + era
        input_paths_dict[key] = correct_dict[key]
    return input_paths_dict


def get_tth_parameters(channel, bdt_type, channel_dir):
    '''Reads the parameters for the tth channel

    Parameters:
    ----------
    channel : str
        Name of the channel for which the parameters will be loaded
    bdt_type : str
        Name of the bdtType
    channel_dir : str
        Path to the "info" directory for the run

    Returns:
    -------
    parameters : dict
        Necessary info for loading and weighing the data
    '''
    parameters = {}
    keys_path = os.path.join(channel_dir, 'keys.txt')
    info_path = os.path.join(channel_dir, 'info.json')
    datacard_info_path = os.path.join(channel_dir, 'datacard_info.json')
    trainvar_path = os.path.join(channel_dir, 'trainvars.txt')
    htt_var_path = os.path.join(channel_dir, 'HTT_var.txt')
    dict_list = ut.read_parameters(datacard_info_path)
    multidict = {}
    if dict_list != []:
        for dictionary in dict_list:
            if bdt_type in dictionary['bdtType']:
                if multidict == {}:
                    multidict = dictionary
                else:
                    print(
                        '''Warning: Multiple choices with the
                        given bdtType. Using %s as bdtType'''
                        % (multidict['bdtType']))
        parameters.update(multidict)
    parameters['HTT_var'] = read_list(htt_var_path)
    parameters['trainvars'] = read_list(trainvar_path)
    info_dict = ut.read_multiline_json_to_dict(info_path)
    if os.path.exists(keys_path):
        parameters['keys'] = read_list(keys_path)
    else:
        print('Error: File %s does not exist. No keys found' % (keys_path))
    parameters.update(info_dict)
    return parameters


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


def print_info(
        input_path,
        channel_in_tree,
        variables,
        bdt_type,
        channel,
        keys,
        masses,
        mass_randomization
):
    '''Prints the data loading preferences info'''
    print('In data_manager')
    print(':::: Loading data ::::')
    print('inputPath: ' + str(input_path))
    print('channelInTree: ' + str(channel_in_tree))
    print('-----------------------------------')
    print('variables:')
    print_columns(variables)
    print('bdt_type: ' + str(bdt_type))
    print('channel: ' + str(channel))
    print('keys: ')
    print_columns(keys)
    print('masses: ' + str(masses))
    print('mass_randomization: ' + str(mass_randomization))


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


def get_node_nr(path):
    '''Extracts the node number from the path. For 'hh_nonres type.

    Parameters:
    ----------
    path : str
        Path to the file

    Returns:
    -------
    node_name : str
        Name for the node (under nodeX in the pandas dataframe)
    '''
    filename = os.path.basename(path)
    filename_elements = filename.split('_')
    try:
        index = filename_elements.index('node')
        node_name = '_'.join(
            [filename_elements[index], filename_elements[index + 1]]
        )
    except ValueError:
        node_name = filename
    return node_name


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
