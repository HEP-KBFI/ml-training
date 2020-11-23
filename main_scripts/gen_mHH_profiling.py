'''
Call with 'python'

Usage:
    gen_mHH_profiling.py
    gen_mHH_profiling.py [--fit=BOOL --create_info=BOOL --create_profile=BOOL --weight_dir=DIR --masses_type=STR]

Options:
    -f --fit=BOOL                     Fit the TProfile [default: 0]
    -i --create_info=BOOL             Create new histo_dict.json [default: 0]
    -p --create_profile=BOOL          Creates the TProfile without the fit. [default: 0]
    -w --weight_dir=DIR               Directory where the weights will be saved [default: $HOME/gen_mHH_weight_dir]
    -m --masses_type=STR              'low', 'high' or 'all' [default: all]
'''
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_parameter_reader as hpr
from machineLearning.machineLearning import hh_tools as hht
from machineLearning.machineLearning import data_loader as dl
from machineLearning.machineLearning import hh_parameter_reader as hpr
from machineLearning.machineLearning import data_loading_tools as dlt
from ROOT import TCanvas, TProfile, TF1
from ROOT import TFitResultPtr
import os
import json
import ROOT
import numpy as np
import docopt
import glob
import subprocess


def get_all_trainvars(info_dir):
    ''' Reads the trainvars from the provided info dir

    Parameters:
    ----------
    info_dir : str
        Path to the info_directory of the required channel

    Returns:
    -------
    trainvars : list
        List of the training variables that will be used to create the
        TProfiles.
    '''
    trainvar_path = os.path.join(info_dir, 'trainvars.json')
    trainvar_dicts = ut.read_parameters(trainvar_path)
    trainvars = [trainvar_dict['key'] for trainvar_dict in trainvar_dicts]
    return trainvars


def create_histo_info(trainvars):
    '''Creates the histogram info for each trainvar

    Parameters:
    ----------
    trainvars : list
        List of the training variables that will be used to create the 
        TProfiles

    Returns:
    -------
    histo_infos : list of dicts
        List of dictionaries containing the info for each trainvar TProfile
        creation
    '''
    template = {
        'Variable': '',
        'nbins': 55,
        'min': 0.0,
        'max': 1100.0,
        'fitFunc_AllMassTraining': 'pol1',
        'fitFunc_LowMassTraining': 'pol1',
        'fitFunc_HighMassTraining': 'pol1'
    }
    histo_infos = []
    for trainvar in trainvars:
        histo_info = template.copy()
        histo_info['Variable'] = trainvar
        histo_infos.append(histo_info)
    return histo_infos


def create_histo_dict(info_dir):
    ''' Creates the histo_dict.json. WARNING: Will overwrite the existing one
    in the info_dir

    Parameters:
    ----------
    info_dir : str
        Path to the info_directory of the required channel

    Returns:
    -------
    Nothing
    '''
    histo_dict_path = os.path.join(info_dir, 'histo_dict.json')
    trainvars = get_all_trainvars(info_dir)
    if os.path.exists(histo_dict_path):
        histo_infos = update_histo_dict(trainvars, histo_dict_path)
    else:
        histo_infos = create_histo_info(trainvars)
    with open(histo_dict_path, 'wt') as out_file:
        for histo_info in histo_infos:
            json.dump(histo_info, out_file)
            out_file.write('\n')


def update_histo_dict(trainvars, histo_dict_path):
    '''Updates the current histo_dict.json according to the new trainvars

    Parameters:
    ----------
    trainvars : list
        List of trainvars to be present in the histo_dict.json
    histo_dict_path : str
        Path where the histo_dict.json is located

    Returns:
    -------
    histo_infos : list of dicts
        Updates info about each trainvar to be saved into histo_dict.json
    '''
    old_trainvars = read_trainvars_from_histo_dict(histo_dict_path)
    missing_trainvars = list(set(trainvars) - set(old_trainvars))
    redundant_trainvars = list(set(old_trainvars) - set(trainvars))
    histo_infos = create_renewed_histo_dict(
        missing_trainvars, redundant_trainvars, histo_dict_path
    )
    return histo_infos


def create_renewed_histo_dict(
        missing_trainvars,
        redundant_trainvars,
        histo_dict_path
):
    ''' Creates renewed list of histogram infos.

    Parameters:
    -----------
    missing_trainvars : list
        List of new trainvars not present in the old histo_dict.json
    redundant_trainvars : list
        List of trainvars in the old histo_dict.json not present in the new
        trainvars.json
    histo_dict_path : str
        Path where the histo_dict.json is located

    Returns:
    -------
    new_histo_infos : list of dicts
        List of histo_infos to be saved into the renewed histo_dict.json
    '''
    old_histo_dicts = ut.read_parameters(histo_dict_path)
    new_histo_infos = []
    template = {
        'Variable': '',
        'nbins': 55,
        'min': 0.0,
        'max': 1100.0,
        'fitFunc_AllMassTraining': 'pol1',
        'fitFunc_LowMassTraining': 'pol1',
        'fitFunc_HighMassTraining': 'pol1'
    }
    for old_histo_dict in old_histo_dicts:
        if old_histo_dict['Variable'] in redundant_trainvars:
            continue
        else:
            new_histo_infos.append(old_histo_dict)
    for missing_trainvar in missing_trainvars:
        histo_info = template.copy()
        histo_info['Variable'] = missing_trainvar
        new_histo_infos.append(histo_info)
    return new_histo_infos



def read_trainvars_from_histo_dict(histo_dict_path):
    '''Reads the trainvars for which there is histogram info set previously

    Parameters:
    -----------
    histo_dict_path : str
        Path where the histo_dict.json is located

    Returns:
    -------
    old_trainvars : list
        List of trainvars read from the histo_dict.json file
    '''
    histo_dicts = ut.read_parameters(histo_dict_path)
    old_trainvars = [histo_dict['Variable'] for histo_dict in histo_dicts]
    return old_trainvars


####################################################################


def create_TProfiles(
        info_dir, weight_dir, data,
        masses_type, global_settings, label
):
    ''' Creates the TProfiles (without the fit) for all the trainvars vs
    gen_mHH

    Parameters:
    -----------
    info_dir : str
        Path to the info_directory of the required channel
    weight_dir : str
        Path to the directory where the TProfiles will be saved
    data : pandas DataFrame
        Data to be used for creating the TProfiles
    masses_type : str
        Which masses type to use. 'low', 'high' or 'all'
    global_settings : dict
        Global settings (channel, bdtType etc.)
    label : str
        An optional string to be added to the end of the file name. For
        example 'before weighing' and 'after weighing'

    Returns:
    --------
    Nothing
    '''
    trainvars = list(get_all_trainvars(info_dir))
    if 'gen_mHH' in trainvars:
        trainvars.remove('gen_mHH')
    for dtype in [0, 1]:
        type_data = data.loc[data['target'] == dtype]
        single_dtype_TProfile(
            type_data, trainvars, dtype, label, info_dir, global_settings,
            weight_dir)


def single_dtype_TProfile(
        type_data, trainvars, dtype,
        label, info_dir, global_settings,
        weight_dir
):
    '''

    Parameters:
    -----------
    type_data : int
        data for either signal or background only
    trainvars : list
        List of the training variables that will be used to create the 
        TProfiles
    dtype : str
        Either 0 or 1. Will be added used to create the
        filename.
    label : str
        An optional string to be added to the end of the file name. For
        example 'before weighing' and 'after weighing'
    info_dir : str
        Path to the info_directory of the required channel
    global_settings : dict
        Global settings (channel, bdtType etc.)
    weight_dir : str
        Path to the directory where the TProfiles will be saved

    Returns:
    --------
    Nothing
    '''
    for trainvar in trainvars:
        print('Variable Name: ' + str(trainvar))
        filename = choose_file_name(weight_dir, dtype, label, trainvar)
        plotting_main(
            type_data, trainvar, filename,
            masses_type, info_dir, global_settings
        )


def choose_file_name(weight_dir, dtype, label, trainvar):
    '''

    Parameters:
    -----------
    weight_dir : str
        Path to the directory where the TProfiles will be saved
    dtype : str
        Either 0 or 1. Will be added used to create the
        filename.
    label : str
        An optional string to be added to the end of the file name. For
        example 'before weighing' and 'after weighing'
    trainvar : str
        Name of the training variable for the TProfile file.

    Returns:
    --------
    out_file : str
        Path of the file to where TProfile will be saved.
    '''
    if dtype == 1:
        pre_str = 'TProfile_signal'
    else:
        pre_str = 'TProfile'
    filename = '_'.join([pre_str, str(trainvar), label])
    out_file = os.path.join(weight_dir, filename + '.root')
    return out_file


##################################################################


def do_fit(weight_dir, info_dir, data, masses_type):
    ''' Fits the Data with a given order of polynomial

    Parameters:
    -----------
    weight_dir : str
        Path to the directory where the TProfiles will be saved
    info_dir : str
        Path to the info_directory of the required channel
    data : pandas DataFrame
        Data to be used for creating the TProfiles & fits
    masses_type : str
        Which masses type to use. 'low', 'high' or 'all'

    Returns:
    --------
    Nothing
    '''
    trainvars = list(get_all_trainvars(info_dir))
    if 'gen_mHH' in trainvars:
        trainvars.remove('gen_mHH')
    masses = find_masses(masses_type)
    histo_dicts_json = os.path.join(info_dir, 'histo_dict.json')
    histo_dicts = ut.read_parameters(histo_dicts_json)
    for trainvar in trainvars:
        histo_dict = dlt.find_correct_dict(
            'Variable', str(trainvar), histo_dicts)
        fit_poly_order = get_fit_function(histo_dict, masses_type)
        canvas, profile = plotting_init(data, trainvar, histo_dict, masses)
        print('Variable Name: ' + str(trainvar))
        filename = '_'.join(['TProfile_signal_fit_func', str(trainvar)])
        out_file = os.path.join(weight_dir, filename + '.root')
        fit_function = 'fitFunction_' + str(trainvar)
        masses = find_masses(masses_type)
        mass_min = min(masses)
        mass_max = max(masses)
        print('Fitfunction: ' + fit_function)
        print('Range: ' + '[' + str(mass_min) + ',' + str(mass_max) + ']')
        function_TF1 = TF1(
            fit_function, fit_poly_order, float(mass_min), float(mass_max)
        )
        result_ptr = TFitResultPtr()
        result_ptr = profile.Fit(function_TF1, 'SF') # Fit with Minuit
        function_TF1.Draw('same')
        canvas.Modified()
        canvas.Update()
        canvas.SaveAs(out_file)
        tfile = ROOT.TFile(out_file, "RECREATE")
        function_TF1.Write()
        tfile.Close()


def get_fit_function(histo_dict, masses_type):
    ''' Reads the polynomial order to be used for the fit for a given trainvar
    histo_dict

    Parameters:
    -----------
    histo_dict : dict
        Dictionary containing the info for plotting for a given trainvar
    masses_type : str
        'low', 'high' or 'all'. Used to find which key to use from the
        histo_dict.

    Returns:
    --------
    poly_order : int
        Order of the polynomial to be used in the fit
    '''
    key = 'fitFunc_' + masses_type.capitalize() + 'MassTraining'
    poly_order = histo_dict[key]
    return poly_order


def find_masses(masses_type):
    ''' Finds the masses to be used in the fit

    Parameters:
    -----------
    info_dir : str
        Path to the info_directory of the required channel
    global_settings : dict
        Global settings (channel, bdtType etc.)
    masses_type : str
        Which masses type to use. 'low', 'high' or 'all'

    Returns:
    --------
    masses : list
        List of masses to be used.
    '''
    channel_dir, info_dir, global_settings = ut.find_settings()
    scenario = global_settings['scenario']
    reader = hpr.HHParameterReader(channel_dir, scenario)
    preferences = reader.parameters
    if masses_type == 'all':
        masses = preferences['masses']
    else:
        masses = preferences['masses_' + masses_type]
    return masses


def plotting_init(data, trainvar, histo_dict, masses, weights='totalWeight'):
    ''' Initializes the plotting

    Parameters:
    -----------
    data : pandas DataFrame
        Data to be used for creating the TProfiles
    trainvar : str
        Name of the training variable.
    histo_dict : dict
        Dictionary containing the info for plotting for a given trainvar
    masses : list
        List of masses to be used
    [weights='totalWeight'] : str
        What column to be used for weight in the data.

    Returns:
    --------
    canvas : ROOT.TCanvas instance
        canvas to be plotted on
    profile : ROOT.TProfile instance
        profile for the fitting
    '''
    canvas = TCanvas('canvas', 'TProfile plot', 200, 10, 700, 500)
    canvas.GetFrame().SetBorderSize(6)
    canvas.GetFrame().SetBorderMode(-1)
    gen_mHH_values = np.array(data['gen_mHH'].values, dtype=np.float)
    trainvar_values = np.array(data[trainvar].values, dtype=np.float)
    weights = np.array(data[weights].values, dtype=np.float)
    sanity_check = len(gen_mHH_values) == len(trainvar_values) == len(weights)
    assert sanity_check
    title = 'Profile of '+ str(trainvar) +' vs gen_mHH'
    num_bins = (len(masses) - 1)
    xlow = masses[0]
    xhigh = (masses[(len(masses) - 1)] + 100.0)
    ylow = histo_dict["min"]
    yhigh = histo_dict["max"]
    profile = TProfile(
        'profile', title, num_bins,
        xlow, xhigh, ylow, yhigh
    )
    mass_bins = np.array(masses, dtype=float)
    profile.SetBins((len(mass_bins) - 1), mass_bins)
    profile.GetXaxis().SetTitle("gen_mHH (GeV)")
    profile.GetYaxis().SetTitle(str(trainvar))
    for x, y, w in zip(gen_mHH_values, trainvar_values, weights):
        profile.Fill(x, y, w)
    profile.Draw()
    canvas.Modified()
    canvas.Update()
    return canvas, profile


def plotting_main(
        data,
        trainvar,
        filename,
        masses_type,
        info_dir,
        global_settings
):
    ''' Main function for plotting.

    Parameters:
    -----------
    data : pandas DataFrame
        Data to be used for creating the TProfiles
    trainvar : str
        Name of the training variable.
    filename : str
        Path to the file where the TProfile will be saved
    masses_type : str
        Type of the masses to be used. 'low', 'high' or 'all'
    info_dir : str
        Path to the info_directory of the required channel
    global_settings : dict
        Global settings (channel, bdtType etc.)

    Returns:
    --------
    Nothing
    '''
    masses = find_masses(masses_type)
    histo_dicts_json = os.path.join(info_dir, 'histo_dict.json')
    histo_dicts = ut.read_parameters(histo_dicts_json)
    histo_dict = dlt.find_correct_dict(
        'Variable', str(trainvar), histo_dicts)
    canvas, profile = plotting_init(data, trainvar, histo_dict, masses)
    canvas.SaveAs(filename)


def create_all_fitFunc_file(weight_dir, global_settings):
    wild_card_path = os.path.join(weight_dir, '*signal_fit_func*')
    all_single_files = glob.glob(wild_card_path)
    all_paths_str = ' '.join(all_single_files)
    if 'nonres' in global_settings['scenario']:
        scenario = 'nonres'
    else:
        scenario = global_settings['scenario']
    res_fileName = '_'.join([
        global_settings['channel'],
        'TProfile_signal_fit_func',
        scenario
        ])
    resulting_file = os.path.join(weight_dir, res_fileName + '.root')
    subprocess.call('hadd ' + resulting_file + ' ' + all_paths_str, shell=True)


def main(fit, create_info, weight_dir, masses_type, create_profile):
    ''' Main function for operating the fitting, plotting and creation of
    histo_dict

    Parameters:
    -----------
    fit : bool
        Whether to do a fit
    create_info : bool
        Wheter to create histo_dict from scratch
    weight_dir : str
        Path to the directory where the TProfile files will be saved
    masses_type : str
        Type of the masses to be used. 'low', 'high' or 'all'
    create_profile : bool
        Whether to create the TProfiles.

    Returns:
    --------
    Nothing
    '''
    channel_dir, info_dir, global_settings = ut.find_settings()
    if 'nonres' in global_settings['scenario']:
        scenario = 'nonres'
    else:
        scenario = global_settings['scenario']
    reader = hpr.HHParameterReader(channel_dir, scenario)
    normalizer = hht.HHDataNormalizer
    data_helper = hht.HHDataHelper
    preferences = reader.parameters
    if create_info:
        create_histo_dict(info_dir)
    if create_profile or fit:
        loader = dl.DataLoader(
            data_helper,
            normalizer,
            global_settings,
            preferences,
            normalize=False
        )
        data = loader.data
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        if fit:
            do_fit(weight_dir, info_dir, data, masses_type)
            resulting_hadd_file = os.path.join(weight_dir, 'all_fitFunc.root')
            print(
                'Creating a single fit file with "hadd" to: ' + str(
                    resulting_hadd_file
                )
            )
            create_all_fitFunc_file(weight_dir, global_settings)
        if create_profile:
            create_TProfiles(
                info_dir, weight_dir, data,
                masses_type, global_settings, label='raw'
            )
            try:
                data = loader.prepare_data(data)
                create_TProfiles(
                    info_dir, weight_dir, data,
                    masses_type, global_settings, label='reweighed'
                )
            except ReferenceError:
                print('No fit for variables found')
                print('Please fit the variables for plots after reweighing')


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        fit = bool(int(arguments['--fit']))
        create_info =  bool(int(arguments['--create_info']))
        weight_dir = os.path.expandvars(arguments['--weight_dir'])
        masses_type = arguments['--masses_type']
        create_profile = bool(int(arguments['--create_profile']))
        main(fit, create_info, weight_dir, masses_type, create_profile)
    except docopt.DocoptExit as e:
        print(e)
