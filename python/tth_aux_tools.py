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
