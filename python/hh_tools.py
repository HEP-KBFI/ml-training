import ROOT
import glob
import numpy as np
import os
from machineLearning.machineLearning import universal_tools as ut

# needs to be able to load files also when hyperopt ??


class HHDataNormalizer:
    def __init__(self, data, preferences, global_settings):
        print("Using HHDataNormalizer")
        self.data = data
        self.preferences = preferences
        self.global_settings = global_settings
        self.weight = 'totalWeight'
        self.condition_sig = data['target'] == 1
        self.condition_bkg = data['target'] == 0

    def normalization_step1(self):
        if 'nonres' in self.global_settings['scenario']:
            self.data.loc[(self.data['target'] == 1), [self.weight]] *= 1./float(
               len(self.preferences['nonResScenarios']))
            self.data.loc[(self.data['target'] == 0), [self.weight]] *= 1./float(
               len(self.preferences['nonResScenarios']))
        elif 'oversampling' in self.global_settings['bkg_mass_rand']:
            self.data.loc[(self.data['target'] == 1), [self.weight]] *= 1./float(
                len(self.preferences['masses']))
            self.data.loc[(self.data['target'] == 0), [self.weight]] *= 1./float(
                len(self.preferences['masses']))

    def flatten_distributions(self):
        if 'SUM_HH' in self.global_settings['bdtType']:
            if 'nonres' in self.global_settings['scenario']:
                self.flatten_nonres_distributions()
            else:
                self.flatten_resonant_distributions()
        else:
            sig_factor = 100000./self.data.loc[
                self.condition_sig, [self.weight]
            ].sum()
            self.data.loc[self.condition_sig, [self.weight]] *= sig_factor
            bkg_factor = 100000./self.data.loc[
                self.condition_bkg, [self.weight]
            ].sum()
            self.data.loc[self.condition_bkg, [self.weight]] *= bkg_factor

    def flatten_nonres_distributions(self):
        for node in set(self.data['nodeXname'].astype(str)):
            condition_node = self.data['nodeXname'].astype(str) == str(node)
            node_sig_weight = self.data.loc[
                self.condition_sig & condition_node, [self.weight]]
            sig_node_factor = 100000./node_sig_weight.sum()
            self.data.loc[
                self.condition_sig & condition_node,
                [self.weight]] *= sig_node_factor
            node_bkg_weight = self.data.loc[
                self.condition_bkg & condition_node, [self.weight]]
            bkg_node_factor = 100000./node_bkg_weight.sum()
            self.data.loc[
                self.condition_bkg & condition_node,
                [self.weight]] *= bkg_node_factor

    def flatten_resonant_distributions(self):
        for mass in set(self.data['gen_mHH']):
            condition_mass = self.data['gen_mHH'].astype(int) == int(mass)
            mass_sig_weight = self.data.loc[
                self.condition_sig & condition_mass, [self.weight]]
            sig_mass_factor = 100000./mass_sig_weight.sum()
            self.data.loc[
                self.condition_sig & condition_mass,
                [self.weight]] *= sig_mass_factor
            mass_bkg_weight = self.data.loc[
                self.condition_bkg & condition_mass, [self.weight]]
            bkg_mass_factor = 100000./mass_bkg_weight.sum()
            self.data.loc[
                self.condition_bkg & condition_mass,
                [self.weight]] *= bkg_mass_factor

    def data_normalization(self):
        self.normalization_step1()
        self.flatten_distributions()
        return self.data


class HHDataHelper:
    def __init__(self, data_normalizer, preferences, global_settings):
        print("Using HHDataHelper")
        self.preferences = preferences
        self.global_settings = global_settings
        self.weight = 'totalWeight'
        self.nr_events_per_file = -1
        self.data_normalizer = data_normalizer
        self.set_extra_df_columns()
        self.cancelled_trainvars = ['gen_mHH']

    def set_extra_df_columns(self):
        self.extra_df_columns = []
        if 'nonres' in self.global_settings['scenario']:
            self.extra_df_columns.append('nodeX')

    def create_to_be_dropped_list(self, sample_name):
        self.to_be_dropped = []
        self.to_be_loaded = []
        if 'nonres' in self.global_settings['scenario']:
            self.nonres_weights = [
                str('Weight_') + scenario for scenario in self.preferences['nonResScenarios']
            ]
            if 'Base' in self.preferences['nonResScenarios']:
                self.to_be_dropped.append('Weight_Base')
            if 'nonres' in sample_name:
                self.to_be_loaded.extend(self.nonres_weights)
                if 'SM' not in self.preferences['nonResScenarios']:
                    self.to_be_loaded.append('Weight_SM')
            self.to_be_dropped.extend(list(self.preferences['nonResScenarios']))
            self.to_be_dropped.extend(['nodeX'])
        else:
            self.to_be_dropped.extend(['gen_mHH'])

    def get_ntuple_paths(self, input_path, folder_name, file_type='hadd*'):
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
                bkg_element_paths = self.find_paths_both_conventions(
                    input_path, bkg_element, paths, file_type=file_type)
                print('--------------')
                print(bkg_element)
                print(bkg_element_paths)
                paths.extend(bkg_element_paths)
        else:
            paths = self.find_paths_both_conventions(
                input_path, folder_name + '*', paths, file_type=file_type)
        return paths

    def find_paths_both_conventions(
            self, input_path, folder_name, paths, file_type
    ):
        """ Finds the paths for both the old and the new convention given
        the main directory of the ntuples and the sample name to search for.
        """
        wild_card_path = os.path.join(
            input_path, folder_name, 'central', file_type + '.root')
        paths = glob.glob(wild_card_path)
        if len(paths) == 0:
            wild_card_path = os.path.join(
                input_path, folder_name, file_type + '.root')
            paths = glob.glob(wild_card_path)
        return paths

    def set_sample_info(self, folder_name, path):
        if 'signal' in folder_name:
            return self.set_signal_sample_info(folder_name)
        else:
            return self.set_background_sample_info(folder_name, path)

    def set_signal_sample_info(self, folder_name):
        target = 1
        if 'nonresonant' in folder_name:
            process = 'signal_ggf_nonresonant_hh'
        else:
            process = folder_name.replace('_' + folder_name.split('_')[-1], '')
        return process, target

    def set_background_sample_info(self, folder_name, path):
        target = 0
        background_catfile = os.path.join(
            os.path.expandvars('$CMSSW_BASE'),
            'src/machineLearning/machineLearning/info',
            'HH',
            'background_categories.json'
        )
        background_categories = ut.read_json_cfg(background_catfile)
        possible_processes = []
        sample_dict = {}
        for category in background_categories:
            possible_samples = background_categories[category]
            sample_dict.update(possible_samples)
            for sample in possible_samples.keys():
                if sample in path:
                    possible_processes.append(sample)
        process = sample_dict[max(possible_processes, key=len)]
        return process, target

    def data_imputer(
        self, chunk_df, folder_name, target, data
    ):
        if 'nonres' not in self.global_settings['scenario']:
            data = self.resonant_data_manipulation(
                chunk_df, folder_name, target, data)
        else:
            data = self.nonresonant_data_manipulation(
                chunk_df, folder_name, target, data)
        return data

    def resonant_data_manipulation(
        self, chunk_df, folder_name, target, data
    ):
        if target == 1:
            for mass in self.preferences['masses']:
                if str(mass) in folder_name:
                    chunk_df["gen_mHH"] = float(mass)
            data = data.append(chunk_df, ignore_index=True, sort=False)
        else:
            if self.global_settings['bkg_mass_rand'] == "default":
                chunk_df["gen_mHH"] = float(np.random.choice(
                    self.preferences['masses'], size=len(chunk_df)))
                data = data.append(chunk_df, ignore_index=True, sort=False)
            elif self.global_settings['bkg_mass_rand'] == "oversampling":
                for mass in self.preferences['masses']:
                    chunk_df["gen_mHH"] = float(mass)
                    data = data.append(chunk_df, ignore_index=True, sort=False)
            else:
                raise ValueError(
                    'Cannot use ' + self.global_settings['bkg_mass_rand'] + \
                    " as mass_randomization"
                )
        return data

    def nonresonant_data_manipulation(
        self, chunk_df, folder_name, target, data
    ):
        for i in range(len(self.preferences['nonResScenarios'])):
            chunk_df_node = chunk_df.copy()
            scenario = self.preferences['nonResScenarios'][i]
            chunk_df_node['nodeX'] = i
            for idx, node in enumerate(self.preferences['nonResScenarios']):
                chunk_df_node[node] = 1 if idx == i else 0
            chunk_df_node['nodeXname'] = scenario
            if target == 1:
                if scenario is not "SM":
                    nodeWeight = 1.
                    if 'Base' not in scenario:
                        nodeWeight = chunk_df_node['Weight_' + scenario]
                    nodeWeight /= chunk_df_node['Weight_SM']
                    chunk_df_node['totalWeight'] *= nodeWeight
            data = data.append(chunk_df_node, ignore_index=True, sort=False)
        return data

    def data_reweighing(
            self,
            data,
            skip_int_vars=True
    ):
        """Reweighs the dataframe in order to reduce the importance of gen_mHH

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
        """
        for trainvar in self.preferences['trainvars']:
            if trainvar in self.cancelled_trainvars:
                continue
            filename = '_'.join(['TProfile_signal_fit_func', trainvar]) + '.root'
            file_path = str(os.path.expandvars(os.path.join(
                self.preferences['weight_dir'], filename
            )))
            tfile = ROOT.TFile.Open(file_path)
            fit_function_name = str('_'.join(['fitFunction', trainvar]))
            function = tfile.Get(fit_function_name)
            if bool(self.preferences['all_trainvar_info'][trainvar]) and skip_int_vars:
                data[trainvar] = data[trainvar].astype(int)
                continue
            for mass in self.preferences['masses']:
                data.loc[
                    data['gen_mHH'] == mass, [trainvar]] /= function.Eval(mass)
            tfile.Close()

    def prepare_data(self, data):
        data = data.copy()
        if 'nonres' not in self.global_settings['scenario']:
            self.data_reweighing(data)
        normalizer = self.data_normalizer(
            data, self.preferences, self.global_settings)
        normalized_data = normalizer.data_normalization()
        return normalized_data
