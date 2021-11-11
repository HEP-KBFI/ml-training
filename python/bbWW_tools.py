import os
import numpy as np
import pandas
import re
from machineLearning.machineLearning.hh_tools import HHDataLoader, HHDataNormalizer
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning.data_loader import DataLoader as dlt

class bbWWDataNormalizer(HHDataNormalizer):
    def __init__(self, data, preferences, global_settings):
        print("Using bbWWDataNormalizer")
        self.multiclass = global_settings['ml_method'] != 'xgb'
        super(bbWWDataNormalizer, self).__init__(
            data, preferences, global_settings
        )

    def normalization_step1(self):
        pass

    def flatten_nonres_distributions(self):
        if not self.multiclass:
            HHDataNormalizer.flatten_nonres_distributions(self)
        else:
            signal_weight = self.preferences['signal_weight'] if 'signal_weight' in self.preferences.keys()\
                            else 1
            for node in set(self.data['nodeXname'].astype(str)):
                for process in set(self.data["process"]):
                    condition_node = self.data['nodeXname'].astype(str) == str(node)
                    condition_sig = self.data['process'].astype(str) == process
                    node_sig_weight = self.data.loc[
                        condition_sig & condition_node, [self.weight]]
                    sig_node_factor = 100000./node_sig_weight.sum() if 'HH' not in process else\
                                      100000./(node_sig_weight.sum()*signal_weight)
                    self.data.loc[
                        condition_sig & condition_node,
                        [self.weight]] *= sig_node_factor
        self.print_background_yield()

    def flatten_resonant_distributions(self):
        pass

    def print_background_yield(self):
        print('Fraction of each Background process')
        sumall = 0
        data = self.data
        for process in set(data['process']):
            sumall += data.loc[(data["process"] == process)]["totalWeight"].sum()
        for process in set(data['process']):
            print(process + ': ' + str('%0.3f' %(data.loc[(data["process"] == process)]\
                 ["totalWeight"].sum()/sumall)))

class bbWWLoader(HHDataLoader):
    def __init__(
            self, data_normalizer, preferences, global_settings, split_ggf_vbf=False, mergeWjets=False,
            use_NLO=False, use_NLOweightonly=False,
            nr_events_per_file=-1, weight='totalWeight',
            cancelled_trainvars=['gen_mHH'], normalize=True,
            reweigh=False, remove_negative_weights=True,
            load_bkg=True
    ):
        print('Using bbWW flavor of the HHDataLoader')
        self.use_NLO = use_NLO
        self.use_NLOweightonly = use_NLOweightonly
        self.mergeWjets = mergeWjets
        self.split_ggf_vbf = split_ggf_vbf
        self.bmScenario_in_sample = {'SM':'sm', 'BM1':'1', 'BM2':'2', 'BM3':'3', 'BM4':'4',
                                     'BM5':'5', 'BM6':'6', 'BM7':'7', 'BM8':'8', 'BM9':'9',
                                     'BM10':'10', 'BM11':'11', 'BM12':'12'}
        self.load_bkg = load_bkg
        super(bbWWLoader, self).__init__(
            data_normalizer, preferences, global_settings, nr_events_per_file,
            weight, cancelled_trainvars, normalize, reweigh,
            remove_negative_weights
        )

    def update_to_be_dropped_list(self, process):
        if 'nonres' in self.global_settings['scenario']:
            if self.use_NLOweightonly == False:
                self.nonres_weights = [
                    str('Weight_') + scenario
                    for scenario in self.preferences['nonResScenarios']
            ]
            else:
                self.nonres_weights = [
                    str('Weight_') + scenario + str('_nloOnly')
                    for scenario in self.preferences['nonResScenarios']
            ]
            if 'signal' in process and 'vbf' not in process and 'cHHH' not in process:
                if 'Base' in self.preferences['nonResScenarios']:
                    self.to_be_dropped.append('Weight_Base')
                    self.to_be_loaded.append('Weight_SM')
                else:
                    self.to_be_loaded.extend(self.nonres_weights)
            self.to_be_dropped.extend(
                list(self.preferences['nonResScenarios']))
            self.to_be_dropped.extend(['nodeX'])
        else:
            self.to_be_dropped.extend(['gen_mHH'])
        self.to_be_loaded.append('isHbb_boosted')

    def process_data_imputer(
            self, chunk_df, folder_name, target, data
    ):
        merge_process = ["TTW", "TTWW", "VV", "TTH", "tHq", "tHW", "ZH", "WH", "Other", "XGamma", "TTZ", "ggH", "qqH", "VVV", "TTWH", "TTZH"]
        merge_process_Other = ["TTW", "TTWW", "VV", "Other", "XGamma", "TTZ", "VVV", "DY"]
        merge_process_H = ["TTH", "ZH", "WH", "ggH", "qqH", "tHq", "tHW", "TTWH", "TTZH"]
        if self.mergeWjets:
            merge_process.append("W")
        chunk_df.loc[chunk_df["process"].isin(
            merge_process_Other
        ), "process"] = "Other"

        chunk_df.loc[chunk_df["process"].isin(
            merge_process_H
        ), "process"] = "H"

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
        finalData = data
        for process in set(data['process']):
            print(process + ': ' + str(len(finalData.loc[finalData['process'] == process])))
        self.print_nr_signal_bkg(finalData)
        if self.split_ggf_vbf:
            finalData.loc[finalData['process'].str.contains('signal_ggf'), "process"] = "GGF_HH"
            finalData.loc[finalData['process'].str.contains('signal_vbf'), "process"] = 'VBF_HH'
        else:
            finalData.loc[finalData['process'].str.contains('signal_ggf'), "process"] = "signal_HH"
            finalData.loc[finalData['process'].str.contains('signal_vbf'), "process"] = 'signal_HH'
        return finalData

    def load_data_from_tfile(
            self, process, folder_name, target, path, input_tree
    ):
        if 'TT' in folder_name and self.load_bkg:
            self.nr_events_per_file = -1
        elif 'spin' in folder_name:
            self.nr_events_per_file = 0
            for mass in self.preferences['masses']:
                if str(mass) in folder_name:
                    self.nr_events_per_file = -1
                    break
        elif 'ggf' in folder_name:
            if self.use_NLO == True or self.use_NLOweightonly == True:
                self.nr_events_per_file = -1
            else:
                self.nr_events_per_file = 30000
        else:
            if not self.load_bkg:
                self.nr_events_per_file = 0
            else:
                self.nr_events_per_file = -1

        return dlt.load_data_from_tfile(
            self,
            process,
            folder_name,
            target,
            path,
            input_tree
        )

    def set_background_sample_info(self, path):
        if 'ST' in path:
            return 'ST', 0
        if 'TTToHadronic' in path:
            return '', 0
        return HHDataLoader.set_background_sample_info(self, path)

    def set_signal_sample_info(self, folder_name):
        target = 1
        if 'ggf_nonresonant' in folder_name:
            if 'cHHH' not in folder_name and not self.use_NLO:
                process = 'signal_ggf_nonresonant_hh'
            elif 'cHHH' in folder_name  and self.use_NLO:
                cHHH = re.findall('(cHHH1|cHHH0|cHHH5|cHHH2p45)', folder_name)[0]
                process = 'signal_ggf_nonresonant_%s_hh' %cHHH
            else:
                process = ''
        elif 'vbf_nonresonant' in folder_name:
            process = folder_name.replace('_2b2v_sl_dipoleRecoilOff', '')
            process = process.replace('_2b2v_dipoleRecoilOff', '')
            process = process.replace('_2b2t_dipoleRecoilOff', '')
            '''if 'dipoleRecoilOff' in folder_name:
                process = 'signal_vbf_nonresonant_1_1_1_hh'
            else:
                process = 'signal_vbf_nonresonant_1_1_1_hh_dipoleRecoilOn' '''
        else:
            process = folder_name.replace('_' + folder_name.split('_')[-1], '')
        return process, target

    def nonresonant_data_imputer(
            self, chunk_df, folder_name, target, data
    ):
        if target == 1 and 'ggf' in folder_name and 'cHHH' not in folder_name:
            for i in range(len(self.preferences['nonResScenarios'])):
                chunk_df_node = chunk_df.copy()
                scenario = self.preferences['nonResScenarios'][i]
                chunk_df_node['nodeX'] = i
                for idx, node in enumerate(self.preferences['nonResScenarios']):
                    chunk_df_node[node] = 1 if idx == i else 0
                chunk_df_node['nodeXname'] = scenario
                if self.use_NLOweightonly:
                    if self.bmScenario_in_sample[self.preferences['nonResScenarios'][i]] + '_' not in folder_name:
                        continue
                    nodeWeight = chunk_df_node['Weight_' + scenario + '_nloOnly']
                else:
                    nodeWeight = chunk_df_node['Weight_' + scenario]
                chunk_df_node['totalWeight'] *= nodeWeight
                data = data.append(chunk_df_node, ignore_index=True, sort=False)
        else:
            chunk_df_node = chunk_df.copy()
            chunk_df_node['nodeXname'] = np.random.choice(
                self.preferences['nonResScenarios'], size=len(chunk_df_node))
            for idx, node in enumerate(self.preferences['nonResScenarios']):
                if len(chunk_df_node.loc[chunk_df_node['nodeXname'] == node]):
                    chunk_df_node.loc[chunk_df_node['nodeXname'] == node, node] = 1
                    chunk_df_node.loc[chunk_df_node['nodeXname'] != node, node] = 0
                    chunk_df_node.loc[chunk_df_node['nodeXname'] == node, 'nodeX'] = idx
            data = data.append(chunk_df_node, ignore_index=True, sort=False)
        return data

    def data_reweighing(
        self,
        data,
        skip_int_vars=True,
        skip_vars=[]
    ):
        skip_vars = ['tau21_Hbb', 'm_Hbb_regCorr']
        HHDataLoader.data_reweighing(self, data, skip_vars=skip_vars)
