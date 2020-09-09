''' Auxiliary tools for ttH analysis
'''


def normalize_tth_dataframe(
        data,
        preferences,
        global_settings,
        weight='totalWeight',
        target='target'
):
    '''Normalizes the weights for the HH data dataframe

    Parameters:
    ----------
    data : pandas Dataframe
        Dataframe containing all the data needed for the training.
    preferences : dict
        Preferences for the data choice and data manipulation
    global_settings : dict
        Preferences for the data, model creation and optimization
    [weight='totalWeight'] : str
        Type of weight to be normalized

    Returns:
    -------
    Nothing
    '''
    bdt_type = global_settings['bdtType']
    if 'evtLevelSUM_TTH' in bdt_type:
        bkg_weight_factor = 100000 / data.loc[data[target] == 0][weights].sum()
        sig_weight_factor = 100000 / data.loc[data[target] == 1][weights].sum()
        data.loc[data[target] == 0, [weights]] *= bkg_weight_factor
        data.loc[data[target] == 1, [weights]] *= sig_weight_factor
    if 'oversampling' in  bkg_mass_rand:
         data.loc[(data['target']==1),[weight]] *= 1./float(
            len(preferences['masses']))
         data.loc[(data['target']==0),[weight]] *= 1./float(
            len(preferences['masses']))


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
    info_dict = ut.read_json_cfg(info_path)
    if os.path.exists(keys_path):
        parameters['keys'] = read_list(keys_path)
    else:
        print('Error: File %s does not exist. No keys found' % (keys_path))
    parameters.update(info_dict)
    return parameters


def set_sample_info(folder_name, samplename_info, bdt_type):
    sample_name = None
    target = None
    if 'signal' in folder_name:
        sample_name, target = set_signal_sample_info(
            bdt_type, folder_name)
    else:
        sample_name, target = set_background_sample_info(
            folder_name, samplename_info)
    return sample_name, target


def set_signal_sample_info(bdt_type, folder_name):
    if 'ttH' in folder_name:
        target = 1
        sample_name = 'ttH'
    return sample_name, target


def set_background_sample_info(folder_name, samplename_info):
    sample_name, target = dlt.set_background_sample_info(
        folder_name, samplename_info)
    return sample_name, target


def get_ntuple_paths(input_path, folder_name, bdt_type):
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
    return paths