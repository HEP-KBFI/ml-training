'''
Call with 'python3'

Usage: slurm_xgb_hh.py --parameter_file=PTH --output_dir=DIR

Options:
    -p --parameter_file=PTH      Path to parameters to be run
    --output_dir=DIR             Directory of the output
'''

from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import evaluation_tools as et
from machineLearning.machineLearning import xgb_tools as xt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import slurm_tools as st
from pathlib import Path
import os
import csv
import docopt
import json


def main(hyperparameter_file, output_dir):
    settings_dir = os.path.join(output_dir, 'run_settings')
    global_settings = ut.read_settings(settings_dir, 'global')
    num_classes = global_settings['num_classes']
    nthread = global_settings['nthread']
    path = Path(hyperparameter_file)
    save_dir = str(path.parent)
    hyperparameters = ut.read_parameters(hyperparameter_file)[0]
    preferences = dlt.get_parameters(
        global_settings['process'],
        global_settings['channel'],
        global_settings['bkg_mass_rand'],
        global_settings['tauID_training']
    )
    data = dlt.load_data(
        preferences['inputPath'],
        preferences['channelInTree'],
        preferences['trainvars'],
        global_settings['bdtType'],
        global_settings['channel'],
        preferences['keys'],
        preferences['masses'],
        global_settings['bkg_mass_rand'],
    )
    dlt.reweigh_dataframe(
        data,
        preferences['weight_dir'],
        preferences['trainvars'],
        ['gen_mHH'],
        preferences['masses']
    )
    dlt.normalize_hh_dataframe(data, preferences)
    if bool(global_settings['use_kfold']):
        score = et.kfold_cv(
            xt.model_evaluation_main,
            data,
            preferences['trainvars'],
            global_settings,
            hyperparameters
        )
    else:
        score, pred_train, pred_test = et.get_evaluation(
            xt.model_evaluation_main,
            data,
            preferences['trainvars'],
            global_settings,
            hyperparameters
        )
        st.save_prediction_files(pred_train, pred_test, save_dir)
    score_path = os.path.join(save_dir, 'score.json')
    with open(score_path, 'w') as score_file:
        json.dump({global_settings['fitness_fn']: score}, score_file)


def normalize_hh_dataframe(data, preferences, weight='totalWeight'):
    '''Normalizes the weights for the HH data dataframe

    Parameters:
    ----------
    data : pandas Dataframe
        Dataframe containing all the data needed for the training.
    preferences : dict
        Preferences for the data choice and data manipulation
    [weight='totalWeight'] : str
        Type of weight to be normalized

    Returns:
    -------
    Nothing
    '''
    ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu']
    weight = 'totalWeight'
    condition_sig = data['target'] == 1
    condition_bkg = data['target'] == 0
    if 'SUM_HH' in bdt_type:
        ttbar_weights = data.loc[data['key'].isin(ttbar_samples), [weight]]
        ttbar_factor = preferences['TTdatacard']/ttbar_weights.sum()
        data.loc[data['key'].isin(ttbar_samples), [weight]] *= ttbar_factor
        dy_weights = data.loc[data['key'] == 'DY', [weight]]
        dy_factor = preferences['DYdatacard']/dy_weights.sum()
        data.loc[data['key'] == 'DY', [weight]] *= dy_factor
        if "evtLevelSUM_HH_bb1l_res" in bdt_type:
            w_weights = data.loc[data['key'] == 'W', [weight]]
            w_factor = preferences['Wdatacard']/w_weights.sum()
            data.loc[data['key'] == 'W', [weight]] *= w_factor
        if "evtLevelSUM_HH_2l_2tau_res" in bdt_type:
            ttz_weights = data.loc[data['key'] == 'TTZJets', [weight]]
            ttz_factor = preferences['TTZdatacard']/ttz_weights.sum()
            data.loc[data['key'] == 'TTZJets', [weight]] *= ttz_factor
            ttw_weights = data.loc[data['key'] == 'TTWJets', [weight]]
            ttw_factor = preferences['TTWdatacard']/ttw_weights.sum()
            data.loc[data['key'] == 'TTWJets', [weight]] *= ttw_factor
            zz_weights = data.loc[data['key'] == 'ZZ', [weight]]
            zz_factor = preferences['ZZdatacard']/zz_weights.sum()
            data.loc[data['key'] == 'ZZ', [weight]] *= zz_factor
            wz_weights = data.loc[data['key'] == 'WZ', [weight]]
            wz_factor = preferences['WZdatacard']/wz_weights.sum()
            data.loc[data['key'] == 'WZ', [weight]] *= wz_factor
            ww_weights = data.loc[data['key'] == 'WW', [weight]]
            ww_factor = preferences['WWdatacard']/ww_weights.sum()
            data.loc[data['key'] == 'WW', [weight]] *= ww_factor
        for mass in range(len(preferences['masses'])):
            condition_mass = data['gen_mHH'].astype(int) == int(
                preferences['masses'][mass])
            mass_sig_weight = data.loc[
                condition_sig & condition_mass, [weight]]
            sig_mass_factor = 100000./mass_sig_weight.sum()
            data.loc[
                condition_sig & condition_mass, [weight]] *= sig_mass_factor
            mass_bkg_weight = data.loc[
                condition_bkg & condition_mass, [weight]]
            bkg_mass_factor = 100000./mass_bkg_weight.sum()
            data.loc[
                condition_bkg & condition_mass, [weight]] *= bkg_mass_factor
    else:
        sig_factor = 100000./data.loc[condition_sig, [weight]].sum()
        data.loc[condition_sig, [weight]] *= sig_factor
        bkg_factor = 100000./data.loc[condition_bkg, [weight]].sum()
        data.loc[condition_bkg, [weight]] *= bkg_factor


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        parameter_file = arguments['--parameter_file']
        output_dir = arguments['--output_dir']
        main(parameter_file, output_dir)
    except docopt.DocoptExit as e:
        print(e)