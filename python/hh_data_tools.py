import os
import glob
import numpy as np
from machineLearning.machineLearning import universal_tools as ut


def set_signal_sample_info(bdt_type, folder_name, masses):
    target = 1
    name_map = {
        '4t': 'tttt',
        '2v2t': 'wwtt',
        '4v': 'wwww',
        '2b2v': 'bbvv'
    }
    for key in name_map:
        if key in folder_name:
            sample_name = folder_name.replace(key, name_map[key])
            if 'node' in sample_name:
                sample_name = sample_name.replace('_node', '')
                node_type = sample_name.split('_')[3]
                sample_name = sample_name.replace(str(node_type) + '_', '')
                sample_name = sample_name.replace('_PSWeights', '')
            if sample_name.find('_dipoleRecoilOff') != -1:
                sample_name = sample_name.replace('_dipoleRecoilOff', '')
            sample_name = sample_name.replace('hh_bbvv_sl', 'hh')
            sample_name = sample_name.replace('hh_bbvv', 'hh')
    return sample_name, target


def set_background_sample_info(folder_name, samplename_info):
    sample_name, target = set_background_sample_info_d(
        folder_name, samplename_info)
    if 'ttH' in folder_name:
        target = 0
        sample_name = 'TTH'
    return sample_name, target


def set_sample_info(folder_name, samplename_info, masses, bdt_type):
    sample_name = None
    target = None
    if 'signal' in folder_name:
        sample_name, target = set_signal_sample_info(
            bdt_type, folder_name, masses)
    else:
        sample_name, target = set_background_sample_info(
            folder_name, samplename_info)
    return sample_name, target


def get_ntuple_paths(input_path, folder_name, bdt_type, file_type='hadd*'):
    paths = []
    catfile = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/info',
        'HH',
        'sample_categories.json'
    )
    sample_categories = ut.read_json_cfg(catfile)
    if (folder_name in sample_categories.keys()):
        for fname in sample_categories[folder_name]:
            wild_card_path = os.path.join(
                input_path, fname + '*', 'central', file_type + 'Tight.root')
            print '1stwild=', wild_card_path
            addpaths = glob.glob(wild_card_path)
            if len(addpaths) == 0:
                wild_card_path = os.path.join(
                    input_path, fname + '*', file_type + 'Tight.root')
                addpaths = glob.glob(wild_card_path)
            paths.extend(addpaths)
        paths = list(dict.fromkeys(paths))
    if len(paths) == 0:
        if 'signal' in folder_name:
            wild_card_path = os.path.join(
                input_path, folder_name, 'central', file_type + 'Tight.root')
            paths = glob.glob(wild_card_path)
            if len(paths) == 0:
                wild_card_path = os.path.join(
                    input_path, folder_name, file_type + 'Tight.root')
                paths = glob.glob(wild_card_path)
        else:
            wild_card_path = os.path.join(
                input_path, folder_name + '*', 'central', file_type + 'Tight.root')
            print 'wild=', wild_card_path
            paths = glob.glob(wild_card_path)
            if len(paths) == 0:
                wild_card_path = os.path.join(
                    input_path, folder_name + '*', file_type + 'Tight.root')
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
                chunk_df["gen_mHH"] = np.random.choice(
                    preferences['masses'], size=len(chunk_df))
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
        if target == 1 and 'ggf' in folder_name:
            for i in range(len(preferences['nonResScenarios'])):
               chunk_df_node = chunk_df.copy()
               scenario = preferences['nonResScenarios'][i]
               chunk_df_node['nodeX'] = i
               for idx, node in enumerate(preferences['nonResScenarios']):
                 chunk_df_node[node] = 1 if idx == i else 0
               chunk_df_node['nodeXname'] = scenario
               if scenario is not "SM":
                   nodeWeight = chunk_df_node['Weight_' + scenario]
                   nodeWeight /= chunk_df_node['Weight_SM']
                   chunk_df_node['totalWeight'] *= nodeWeight
               data = data.append(chunk_df_node, ignore_index=True, sort=False)
        else :
            chunk_df_node = chunk_df.copy()
            chunk_df_node['nodeXname'] = np.random.choice(preferences['nonResScenarios'], size=len(chunk_df_node))
            for idx, node in enumerate(preferences['nonResScenarios']):
              if len(chunk_df_node.loc[chunk_df_node['nodeXname'] == node]):
                  chunk_df_node.loc[chunk_df_node['nodeXname'] == node, node] = 1
                  chunk_df_node.loc[chunk_df_node['nodeXname'] != node, node] = 0
                  chunk_df_node.loc[chunk_df_node['nodeXname'] == node, 'nodeX'] = idx
            data = data.append(chunk_df_node, ignore_index=True, sort=False)
        '''for i in range(len(preferences['nonResScenarios'])):
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
            data = data.append(chunk_df_node, ignore_index=True, sort=False)'''
    return data


def set_background_sample_info_d(folder_name, samplename_info):
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
    sample_name = None
    target = None
    for sample in samplename_info.keys():
        if sample in folder_name:
            sample_dict = samplename_info[sample]
            sample_name = sample_dict['sampleName']
            target = sample_dict['target']
    return sample_name, target
