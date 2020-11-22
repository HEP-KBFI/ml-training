import ROOT
from root_numpy import tree2array
import pandas
import glob
import numpy as np
import os
from machineLearning.machineLearning import universal_tools as ut


class DataLoader:
    def __init__(
            self,
            data_loader_class,
            data_normalizer,
            global_settings,
            preferences
    ):
        print('In DataLoader')
        self.data = pandas.DataFrame(columns=preferences['trainvars'])
        self.process_loader = data_loader_class(
            data_normalizer, preferences, global_settings)
        self.global_settings = global_settings
        self.preferences = preferences
        self.set_variables_to_be_loaded()
        self.do_loading()

    def set_variables_to_be_loaded(self):
        self.to_be_loaded = list(self.preferences['trainvars'])
        self.to_be_loaded.extend(['evtWeight', 'event'])
        self.to_be_dropped = []
        if self.global_settings['debug']:
            self.to_be_loaded.extend(['luminosityBlock', 'run'])
        self.to_be_loaded.extend(self.process_loader.to_be_loaded)
        self.to_be_dropped.extend(self.process_loader.to_be_dropped)

    def set_sample_info(self, folder_name, path):
        return self.process_loader.set_sample_info(
            folder_name, path
        )

    def get_ntuple_paths(self, input_path, folder_name, file_type='hadd*'):
        return self.process_loader.get_ntuple_paths(
            input_path, folder_name, file_type=file_type
        )

    def prepare_data(self, data):
        return self.process_loader.prepare_data(data)

    def data_imputer(self, process, chunk_df, folder_name, target):
        chunk_df['process'] = process
        chunk_df['key'] = folder_name
        chunk_df['target'] = int(target)
        chunk_df['totalWeight'] = chunk_df['evtWeight']
        return self.process_loader.data_imputer(
            chunk_df, folder_name, target, self.data
        )

    def load_data_from_tfile(
            self,
            folder_name,
            target,
    ):
        tfile = ROOT.TFile(path)
        tree = tfile.Get(input_tree)
        chunk_arr = tree2array(
            tree, branches=self.to_be_loaded,
            stop=self.process_loader.nr_events_per_file
        )
        chunk_df = pandas.DataFrame(chunk_arr)
        tfile.Close()
        data = data_imputer(
            chunk_df, process, folder_name, target)
        return data

    def signal_background_calc(self, folder_name):
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
        try:
            key_condition = data.key.values == folder_name
            nS = len(
                data.loc[(data.target.values == 1) & key_condition])
            nB = len(
                data.loc[(data.target.values == 0) & key_condition])
            nNW = len(
                data.loc[(data['totalWeight'].values < 0) & key_condition])
            print('Signal: ' + str(nS))
            print('Background: ' + str(nB))
            print('Event weight: ' + str(data.loc[
                    (data.key.values == folder_name)]['evtWeight'].sum()))
            print('Total data weight: ' + str(data.loc[
                    (data.key.values == folder_name)]['totalWeight'].sum()))
            print('Events with negative weights: ' + str(nNW))
            print(':::::::::::::::::')
        except:
            if len(data) == 0:
                print('Error: No data (!!!)')

    def load_data(self):
        data = self.do_loading()
        for trainvar in preferences['trainvars']:
            if str(data[trainvar].dtype) == 'object':
                try:
                    data[trainvar] = data[trainvar].astype(int)
                except:
                    continue
        data = self.prepare_data(data)
        if remove_neg_weights:
            print('Removing events with negative weights')
            data = data.loc[data[self.weight] >= 0]
        return data

    def load_from_sample_paths(self, folder_name, path):
        sample_name, target = self.set_sample_info(folder_name, path)
        input_tree = str(os.path.join(
            preferences['channelInTree'],
            'sel/evtntuple', sample_name, 'evtTree'
        ))
        print(':::::::::::::::::')
        print('Sample name:\t' + str(folder_name))
        print('Tree path:\t' + input_tree + '\n')
        print('Loading from: ' + path)
        return self.load_data_from_tfile(folder_name, target)

    def print_nr_signal_bkg(self, data):
        n = len(data)
        nS = len(data.ix[data.target.values == 1])
        nB = len(data.ix[data.target.values == 0])
        print('For ' + preferences['channelInTree'] + ':')
        print('\t Signal: ' + str(nS))
        print('\t Background: ' + str(nB))

    def do_loading(self):
        print_info(self.global_settings, self.preferences)
        eras = self.preferences['included_eras']
        self.data = pandas.DataFrame({})
        for era in eras:
            input_path_key = 'inputPath' + str(era)
            self.preferences['era_inputPath'] = self.preferences[input_path_key]
            self.preferences['era_keys'] = self.preferences['keys' + str(era)]
            data = self.load_data_from_one_era()
            data['era'] = era
            self.data = total_data.append(data)
        if global_settings['dataCuts'] != 0:
            self.data = data_cutting(total_data, global_settings)
        return self.data

    def load_data_from_one_era(self):
        columns = preferences['trainvars'].expand(
            ['process', 'key', 'target', 'totalWeight'])
        columns.expand(self.process_loader.extra_df_columns)
        data = pandas.DataFrame(columns=columns)
        for folder in self.preferences['era_keys']:
            paths = self.get_ntuple_paths(
                self.preferences['era_inputPath'], folder_name)
            folder_data = load_from_sample_paths(self, folder_name, path)
            data.append(folder_data)
        return data

    def data_cutting(self):
        ''' TO DO '''
        package_dir = os.path.join(
            os.path.expandvars('$CMSSW_BASE'),
            'src/machineLearning/machineLearning/'
        )
        if 'nonres' in self.global_settings['bdtType']:
            addition = 'nonRes'
        else:
            addition = 'res/%s' %(self.global_settings['spinCase'])
        if self.global_settings['dataCuts'] == 1:
            cut_file = os.path.join(
                package_dir, 'info', self.global_settings['process'],
                self.global_settings['channel'], addition, 'cuts.json'
            )
        else:
            cut_file = os.path.join(
                package_dir,
                'info',
                self.global_settings['process'],
                self.global_settings['channel'],
                addition,
                self.global_settings['dataCuts']
            )
        if os.path.exists(cut_file):
            cut_dict = ut.read_json_cfg(cut_file)
            if cut_dict == {}:
                print('No cuts given in the cut file %s' %(cut_file))
            else:
                cut_keys = list(cut_dict.keys())
                for key in cut_keys:
                    try:
                        min_value = cut_dict[key]['min']
                        data = data.loc[(data[key] >= min_value)]
                    except KeyError:
                        print('Minimum condition for %s not implemented' %(key))
                    try:
                        max_value = cut_dict[key]['max']
                        data = data.loc[(data[key] <= max_value)]
                    except KeyError:
                        print('Maximum condition for %s not implemented' %(key))
        else:
            print('Cut file %s does not exist' %(cut_file))
        return data

    def print_info(self, global_settings, preferences):
        '''Prints the data loading preferences info'''
        print('In data_manager')
        print(':::: Loading data ::::')
        print('inputPath: ' + str(preferences['era_inputPath']))
        print('channelInTree: ' + str(preferences['channelInTree']))
        print('-----------------------------------')
        print('variables:')
        self.print_columns(preferences['trainvars'])
        print('bdt_type: ' + str(global_settings['bdtType']))
        print('channel: ' + str(global_settings['channel']))
        print('keys: ')
        self.print_columns(preferences['era_keys'])
        print('masses: ' + str(preferences['masses']))
        print('mass_randomization: ' + str(global_settings['bkg_mass_rand']))

    def print_columns(self, to_print):
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

    def save_to_csv(self):
        file_path = os.path.join(
            self.global_settings['output_dir'],
            self.global_settings['channel'] + '_data.csv'
        )
        self.data.to_csv(file_path, index=False)
