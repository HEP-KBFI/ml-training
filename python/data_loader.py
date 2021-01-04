import os
import pandas
import ROOT
from root_numpy import tree2array
from machineLearning.machineLearning import universal_tools as ut

class DataLoader(object):
    def __init__(
            self,
            preferences,
            global_settings,
            normalize=True,
            remove_negative_weights=True,
            nr_events_per_file=-1,
            weight='totalWeight'
    ):
        print('In DataLoader')
        self.data = pandas.DataFrame(columns=preferences['trainvars'])
        self.global_settings = global_settings
        self.preferences = preferences
        self.nr_events_per_file = nr_events_per_file
        self.remove_neg_weights = remove_negative_weights
        self.weight = weight
        self.normalize = normalize
        self.to_be_dropped = []
        self.to_be_loaded = []
        self.extra_df_columns = []

    def set_variables_to_be_loaded(self, process):
        self.to_be_loaded = list(self.preferences['trainvars'])
        self.to_be_loaded.extend(['evtWeight', 'event'])
        if self.global_settings['debug']:
            self.to_be_loaded.extend(['luminosityBlock', 'run'])
        self.update_to_be_dropped_list(process)
        for drop in self.to_be_dropped:
            if drop in self.to_be_loaded:
                self.to_be_loaded.remove(drop)
    def update_to_be_dropped_list(self, process):
        raise NotImplementedError(
            'Please implement creating to_be_dropped list step for your class'
        )

    def set_sample_info(self, folder_name, path):
        raise NotImplementedError(
            "Please implement getting ntuple paths step for your class"
        )

    def get_ntuple_paths(self, input_path, folder_name, file_type='hadd*'):
        raise NotImplementedError(
            "Please implement getting ntuple paths step for your class"
        )

    def prepare_data(self, data):
        raise NotImplementedError(
            "Please implement data preparation step for your class"
        )

    def data_imputer(self, chunk_df, process, folder_name, target):
        chunk_df['process'] = process
        chunk_df['key'] = folder_name
        chunk_df['target'] = int(target)
        chunk_df['totalWeight'] = chunk_df['evtWeight']
        return self.process_data_imputer(
            chunk_df, folder_name, target, self.data
        )

    def process_data_imputer(self, chunk_df, folder_name, target, data):
        raise NotImplementedError(
            "Please implement process_data_imputer for your class"
        )

    def load_data_from_tfile(
            self,
            process,
            folder_name,
            target,
            path,
            input_tree
    ):
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
            data = self.data_imputer(
                chunk_df, process, folder_name, target)
            return data
        except TypeError:
            print('Incorrect input_tree: ' + str(input_tree))

    def signal_background_calc(self, data, folder_name):
        """Calculates the signal and background

        Parameters:
        ----------
        folder_name : str
            Name of the folder in the keys currently loading the data from

        Returns:
        -------
        Nothing
        """
        try:
            key_condition = data.key.values == folder_name
            n_signal = len(
                data.loc[(data.target.values == 1) & key_condition])
            n_background = len(
                data.loc[(data.target.values == 0) & key_condition])
            n_neg_weights = len(
                data.loc[
                    (data['totalWeight'].values < 0) & key_condition
                ]
            )
            print('Signal: ' + str(n_signal))
            print('Background: ' + str(n_background))
            print('Event weight: ' + str(data.loc[
                    (data.key.values == folder_name)]['evtWeight'].sum()))
            print('Total self.data weight: ' + str(data.loc[
                    (data.key.values == folder_name)]['totalWeight'].sum()))
            print('Events with negative weights: ' + str(n_neg_weights))
            print(':::::::::::::::::')
        except:
            if len(data) == 0:
                print('Error: No data (!!!)')

    def load_data(self):
        data = self.do_loading()
        for trainvar in self.preferences['trainvars']:
            if str(data[trainvar].dtype) == 'object':
                try:
                    data[trainvar] = data[trainvar].astype(int)
                except:
                    continue
        if self.remove_neg_weights:
            print('Removing events with negative weights')
            data = data.loc[data[self.weight] >= 0]
        if self.normalize:
            data = self.prepare_data(data)
        return data

    def load_from_sample_paths(self, folder_name, path):
        process, target = self.set_sample_info(folder_name, path)
        input_tree = str(os.path.join(
            self.preferences['channelInTree'],
            'sel/evtntuple', process, 'evtTree'
        ))
        print(':::::::::::::::::')
        print('Sample name:\t' + str(folder_name))
        print('Tree path:\t' + input_tree + '\n')
        print('Loading from: ' + path)
        return self.load_data_from_tfile(
            process, folder_name, target, path, input_tree)

    def print_nr_signal_bkg(self, data):
        n_signal = len(data.ix[data.target.values == 1])
        n_background = len(data.ix[data.target.values == 0])
        print('For ' + self.preferences['channelInTree'] + ':')
        print('\t Signal: ' + str(n_signal))
        print('\t Background: ' + str(n_background))

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
            data = self.data_cutting(data)
        self.print_nr_signal_bkg(data)
        return data

    def load_data_from_one_era(self):
        columns = list(self.preferences['trainvars'])
        columns.extend(['process', 'key', 'target', 'totalWeight'])
        columns.extend(self.extra_df_columns)
        data = pandas.DataFrame(columns=columns)
        for folder in self.preferences['era_keys']:
            paths = self.get_ntuple_paths(
                self.preferences['era_inputPath'], folder)
            for path in paths:
                folder_data = self.load_from_sample_paths(folder, path)
                data = data.append(folder_data, ignore_index=True, sort=False)
            self.signal_background_calc(data, folder)
        return data

    def data_cutting(self, data):
        package_dir = os.path.join(
            os.path.expandvars('$CMSSW_BASE'),
            'src/machineLearning/machineLearning/'
        )
        if 'nonres' in self.global_settings['scenario']:
            addition = self.global_settings['scenario']
        else:
            addition = 'res/%s' %(self.global_settings['scenario'])
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
                print('No cuts given in the cut file %s' % cut_file)
            else:
                cut_keys = list(cut_dict.keys())
                for key in cut_keys:
                    try:
                        min_value = cut_dict[key]['min']
                        data = data.loc[(data[key] >= min_value)]
                    except KeyError:
                        print('Minimum condition for %s not implemented' % key)
                    try:
                        max_value = cut_dict[key]['max']
                        data = data.loc[(data[key] <= max_value)]
                    except KeyError:
                        print('Maximum condition for %s not implemented' % key)
        else:
            print('Cut file %s does not exist' % cut_file)
        return data

    def print_info(self):
        """Prints the data loading preferences info"""
        print('In data_manager')
        print(':::: Loading data ::::')
        print('inputPath: ' + str(self.preferences['era_inputPath']))
        print('channelInTree: ' + str(self.preferences['channelInTree']))
        print('-----------------------------------')
        print('variables:')
        ut.print_columns(self.preferences['trainvars'])
        print('bdt_type: ' + str(self.global_settings['bdtType']))
        print('channel: ' + str(self.global_settings['channel']))
        print('keys: ')
        ut.print_columns(self.preferences['era_keys'])
        if 'nonres' in self.global_settings['scenario']:
            print('Benchmark points: ' + str(self.preferences['nonResScenarios']))
        else:
            print('masses: ' + str(self.preferences['masses']))
        print('mass_randomization: ' + str(self.global_settings['bkg_mass_rand']))

    def save_to_csv(self):
        file_path = os.path.join(
            os.path.expandvars(self.global_settings['output_dir']),
            'data.csv'
        )
        self.data.to_csv(file_path, index=False)
