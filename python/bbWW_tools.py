import os
import numpy as np
import pandas
import ROOT
from root_numpy import tree2array
from machineLearning.machineLearning.hh_tools import HHDataLoader, HHDataNormalizer
from machineLearning.machineLearning import universal_tools as ut


class bbWWDataNormalizer(HHDataNormalizer):
    def __init__(self, data, preferences, global_settings):
        print("Using bbWWDataNormalizer")
        super(bbWWDataNormalizer, self).__init__(
            data, preferences, global_settings
        )

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
        if 'SUM_HH' in self.global_settings['bdtType']:
            sample_normalizations = self.preferences['tauID_application']
            for sample in sample_normalizations.keys():
               sample_name = sample.replace('datacard', '')
               sample_weights = self.data.loc[self.data['process'] == sample_name, [self.weight]]
               sample_factor = sample_normalizations[sample]/sample_weights.sum()
               self.data.loc[self.data['process'] == sample_name, [self.weight]] *= sample_factor
        self.merge_processes()
        self.print_background_yield()

    def print_background_yield(self):
        print('Fraction of each Background process')
        sumall = 0
        condition_bkg = self.data["target"] == 0
        for process in set(self.data['process']):
            sumall += self.data.loc[(self.data["process"] == process) & (condition_bkg)]["totalWeight"].sum()
        for process in set(self.data['process']):
            print(process + ': ' + str('%0.3f' %(self.data.loc[(self.data["process"] == process) & (condition_bkg)]\
                 ["totalWeight"].sum()/sumall)))

    def merge_processes(self):
        Other = ["TTW", "TTWW", "WW", "WZ", "ZZ", "TTH", "TH", "VH", "Other", "XGamma", "TTZ"] #TTH sample missing from bdt ntuple???
        for process in set(self.data['process']):
            if process in ['TTTo2L2Nu', 'TTToSemiLeptonic']:
                self.data.loc[self.data['process'] == process, 'process'] = 'TT'
            elif process in Other:
                self.data.loc[self.data['process'] == process, 'process'] = 'Other'

class bbWWLoader(HHDataLoader):
    def __init__(
            self, data_normalizer, preferences, global_settings,
            nr_events_per_file=-1, weight='totalWeight',
            cancelled_trainvars=['gen_mHH'], normalize=True,
            reweigh=True, remove_negative_weights=True
    ):
        print('Using bbWW flavor of the HHDataLoader')
        super(bbWWLoader, self).__init__(
            data_normalizer, preferences, global_settings, nr_events_per_file,
            weight, cancelled_trainvars, normalize, reweigh,
            remove_negative_weights
        )

    def update_to_be_dropped_list(self, process):
        if 'nonres' in self.global_settings['scenario']:
            self.nonres_weights = [
                str('Weight_') + scenario
                for scenario in self.preferences['nonResScenarios']
            ]
            if 'signal' in process and 'vbf' not in process and 'cHHH1' not in process:
                if 'Base' in self.preferences['nonResScenarios']:
                    self.to_be_dropped.append('Weight_Base')
                    self.to_be_loaded.append('Weight_SM')
                else:
                    self.to_be_loaded.extend(self.nonres_weights)
                    if 'Weight_SM' not in self.to_be_loaded:
                        self.to_be_loaded.append('Weight_SM')
            self.to_be_dropped.extend(
                list(self.preferences['nonResScenarios']))
            self.to_be_dropped.extend(['nodeX'])
        else:
            self.to_be_dropped.extend(['gen_mHH'])
        self.to_be_loaded.append('isHbb_boosted')

    def process_data_imputer(
            self, chunk_df, folder_name, target, data
    ):
        '''chunk_df.loc[chunk_df["process"].isin(
            ["TTW", "TTWW", "WW", "WZ", "ZZ", "TTH", "TH", "VH", "Other"]
        ), "process"] = "Other"'''
        #chunk_df.loc[
         #   chunk_df["process"].str.contains('signal'),
          #  "process"
        #] = "signal_HH"
        if 'nonres' not in self.global_settings['scenario']:
            data = self.resonant_data_imputer(
                chunk_df, folder_name, target, data)
        else:
            data = self.nonresonant_data_imputer(
                chunk_df, folder_name, target, data)
        return data

    def get_ntuple_paths(self, input_path, folder_name, file_type='hadd*Tight.root'):
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
                    input_path, bkg_element, file_type=file_type)
                print('--------------')
                print(bkg_element)
                print(bkg_element_paths)
                paths.extend(bkg_element_paths)
        else:
            paths = self.find_paths_both_conventions(
                input_path, folder_name + '*', file_type=file_type)
        return paths

    def do_loading(self):
        eras = self.preferences['included_eras']
        data = pandas.DataFrame({})
        for era in eras:
            input_path_key = 'inputPath' + str(era)
            self.preferences['era_inputPath'] = self.preferences[input_path_key]
            self.preferences['era_keys'] = self.preferences['keys' + str(era)]
            self.print_info()
            era_data = self.load_data_from_one_era()
            era_data['era'] = era
            data = data.append(era_data, ignore_index=True, sort=False)
        if self.global_settings['dataCuts'] != 0:
            total_data = self.data_cutting(data)
        return self.final_data(total_data)

    def final_data(self, data):
        finalData = pandas.DataFrame({})
        for process in set(data['process']):
            if (data.loc[data['process'] == process]['target'] == 1).all():
                finalData = finalData.append(data.loc[data['process'] == process])
                print(process + ': ' + len(finalData.loc[finalData['process'] == process]))
            else:
                finalData = finalData.append(data.loc[data['process'] == process].head(100000))
                print(process, + ': ' + len(finalData.loc[finalData['process'] == process]))
        self.print_nr_signal_bkg(finalData)
        finalData.loc[finalData['process'].str.contains('signal_ggf_nonresonant_hh'), "process"] = "signal_HH"
        finalData.loc[finalData['process'].str.contains('signal_vbf'), "process"] = "signal_HH"
        return finalData   

    def load_data_from_tfile(
            self, process, folder_name, target, path, input_tree
    ):
        if 'TT' in folder_name:
            self.nr_events_per_file = 3000000
        elif 'ggf' in folder_name:
            self.nr_events_per_file = -1
        else:
            self.nr_events_per_file = -1
        tfile = ROOT.TFile(path)
        try:
            tree = tfile.Get(input_tree)
            self.set_variables_to_be_loaded(process)
            chunk_arr = tree2array(
                tree, branches=self.to_be_loaded,
                stop=self.nr_events_per_file
            )
            chunk_df = pandas.DataFrame(chunk_arr)
            tfile.Close()
            if 'TTTo2L2Nu' in folder_name or 'TTToSemiLeptonic' in folder_name:
                process = path.split('/')[-2]
            data = self.data_imputer(
                chunk_df, process, folder_name, target)
            return data
        except TypeError:
            print('Incorrect input_tree: ' + str(input_tree))

    def set_background_sample_info(self, path):
        if 'ST' in path:
            return 'ST', 0
        if 'TTToHadronic' in path:
            return '', 0
        else:
            return HHDataLoader.set_background_sample_info(self, path)

    def set_signal_sample_info(self, folder_name):
        target = 1
        if 'ggf_nonresonant' in folder_name:
            if 'cHHH1' not in folder_name:
                process = 'signal_ggf_nonresonant_hh'
            else:
                process = 'signal_ggf_nonresonant_cHHH1_hh'
        elif 'vbf_nonresonant' in folder_name:
            process = folder_name.replace('_2b2v_sl_dipoleRecoilOff', '')
        else:
            process = folder_name.replace('_' + folder_name.split('_')[-1], '')
        return process, target

    def nonresonant_data_imputer(
            self, chunk_df, folder_name, target, data
    ):
        if target == 1 and 'ggf' in folder_name and 'cHHH1' not in folder_name:
            for i in range(len(self.preferences['nonResScenarios'])):
                chunk_df_node = chunk_df.copy()
                scenario = self.preferences['nonResScenarios'][i]
                chunk_df_node['nodeX'] = i
                for idx, node in enumerate(self.preferences['nonResScenarios']):
                    chunk_df_node[node] = 1 if idx == i else 0
                chunk_df_node['nodeXname'] = scenario
                if scenario is not "SM":
                    nodeWeight = chunk_df_node['Weight_' + scenario]
                    nodeWeight /= chunk_df_node['Weight_SM']
                    chunk_df_node['totalWeight'] *= nodeWeight
                data = data.append(chunk_df_node, ignore_index=True, sort=False)
        elif 'cHHH1' not in folder_name:
            chunk_df_node = chunk_df.copy()
            chunk_df_node['nodeXname'] = np.random.choice(
                self.preferences['nonResScenarios'], size=len(chunk_df_node))
            for idx, node in enumerate(self.preferences['nonResScenarios']):
                if len(chunk_df_node.loc[chunk_df_node['nodeXname'] == node]):
                    chunk_df_node.loc[chunk_df_node['nodeXname'] == node, node] = 1
                    chunk_df_node.loc[chunk_df_node['nodeXname'] != node, node] = 0
                    chunk_df_node.loc[chunk_df_node['nodeXname'] == node, 'nodeX'] = idx
            data = data.append(chunk_df_node, ignore_index=True, sort=False)
        else:
            return chunk_df
        return data
