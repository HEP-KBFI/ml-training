import os
import glob
import numpy as np
from machineLearning.machineLearning import universal_tools as ut


def set_signal_sample_info(folder_name):
    target = 1
    if 'nonresonant' in folder_name:
        sample_name = 'signal_ggf_nonresonant_hh'
    else:
        sample_name = folder_name.replace('_' + folder_name.split('_')[-1], '')
    return sample_name, target


def set_background_sample_info(folder_name, path):
    target = 0
    background_catfile = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/info',
        'HH',
        'background_categories.json'
    )
    background_categories = ut.read_json_cfg(background_catfile)
    for category in background_categories:
        possible_samples = background_categories[category]
        for sample in possible_samples.keys():
            if sample in path:
                sample_name = possible_samples[sample]
                return sample_name, target


def set_sample_info(folder_name, path):
    sample_name = None
    target = None
    if 'signal' in folder_name:
        sample_name, target = set_signal_sample_info(folder_name)
    else:
        sample_name, target = set_background_sample_info(folder_name, path)
    return sample_name, target


def get_ntuple_paths(input_path, folder_name, file_type='hadd*'):
    paths = []
    background_catfile = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/info',
        'HH',
        'background_categories.json'
    )
    background_categories = ut.read_json_cfg(background_catfile)
    if 'signal' not in folder_name and folder_name in background_categories.keys():
        bkg_elements = background_categories[folder_name]
        for bkg_element in bkg_elements:
            bkg_element_paths = find_paths_both_conventions(
                input_path, bkg_element, paths, file_type=file_type)
            paths.extend(bkg_element_paths)
    else:
        paths = find_paths_both_conventions(
            input_path, folder_name, paths, file_type=file_type)
    return paths


def find_paths_both_conventions(input_path, folder_name, paths, file_type='hadd*'):
    wild_card_path = os.path.join(
        input_path, folder_name + '*', 'central', file_type + '.root')
    paths = glob.glob(wild_card_path)
    if len(paths) == 0:
        wild_card_path = os.path.join(
            input_path, folder_name + '*', file_type + '.root')
        paths = glob.glob(wild_card_path)
    return paths


def define_new_variables(
        chunk_df,
        sample_name,
        folder_name,
        target,
        preferences,
        global_settings,
        data
):
    if 'nonres' not in global_settings['bdtType']:
        if target == 1:
            for mass in preferences['masses']:
                if str(mass) in folder_name:
                    chunk_df["gen_mHH"] = float(mass)
            data = data.append(chunk_df, ignore_index=True, sort=False)
        elif target == 0:
            if global_settings['bkg_mass_rand'] == "default":
                chunk_df["gen_mHH"] = float(np.random.choice(
                    preferences['masses'], size=len(chunk_df)))
                data = data.append(chunk_df, ignore_index=True, sort=False)
            elif global_settings['bkg_mass_rand'] == "oversampling":
                for mass in preferences['masses']:
                    chunk_df["gen_mHH"] = float(mass)
                    data = data.append(chunk_df, ignore_index=True, sort=False)
            else:
                raise ValueError(
                    'Cannot use ' + global_settings['bkg_mass_rand'] + \
                    " as mass_randomization"
                )
        else:
            raise ValueError('Cannot use ' + str(target) + ' as target')
    else:
        for i in range(len(preferences['nonResScenarios'])):
            chunk_df_node = chunk_df.copy()
            scenario = preferences['nonResScenarios'][i]
            chunk_df_node['nodeX'] = i
            for idx, node in enumerate(preferences['nonResScenarios']):
                chunk_df_node[node] = 1 if idx == i else 0
            chunk_df_node['nodeXname'] = scenario
            if target == 1:
                if scenario is not "SM":
                    nodeWeight = chunk_df_node['Weight_' + scenario]
                    nodeWeight /= chunk_df_node['Weight_SM']
                    chunk_df_node['totalWeight'] *= nodeWeight
            data = data.append(chunk_df_node, ignore_index=True, sort=False)
    return data
