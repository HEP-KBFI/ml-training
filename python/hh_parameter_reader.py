import os
import json
from machineLearning.machineLearning import universal_tools as ut

class HHParameterReader:
    def __init__(self, channel_dir, scenario):
        ''' Reads the HH parameters from the info directory

        Parameters:
        ----------
        channel_dir : str
            Path to the channel info directory
        scenario : str
            Options: [nonres, spin0, spin2]

        Returns:
        --------
        preferences : dict
            Dictionary containing the necessary info for loading the data
            and normalizing / weighing it
        '''
        self.scenario = 'res/' + scenario if 'nonres' not in scenario else scenario
        self.info_dir = os.path.join(channel_dir, self.scenario)
        self.trainvars_path = os.path.join(channel_dir, self.scenario, 'trainvars.json')
        self.all_trainvars_path = os.path.join(channel_dir, 'all_trainvars.json')
        self.interpret_info_file()

    def read_trainvar_info(self, path):
        '''Reads the trainvar info

        Parameters:
        -----------
        path : str
            Path to the training file

        Returns:
        -------
        trainvar_info : dict
            Dictionary containing trainvar info (e.g is the trainvar supposed to
            be an integer or not)
        '''
        trainvar_info = {}
        info_dicts = ut.read_parameters(path)
        for single_dict in info_dicts:
            trainvar_info[single_dict['key']] = single_dict['true_int']
        return trainvar_info

    def find_input_paths(
            self,
            info_dict,
            tau_id_training,
    ):
        '''Finds era-wise inputPaths

        Parameters:
        ----------
        info_dict : dict
            Dict from info.json file
        tau_id_trainings : list of dicts
            info about all tau_id_training for inputPaths
        tau_id_training : str
            Value of tau_id_training node

        Returns:
        -------
        input_paths_dict : dict
            Dict conaining the inputPaths for all eras
        '''
        eras = info_dict['included_eras']
        input_paths_dict = {}
        tauID_training = info_dict['tauID_training'][tau_id_training]
        for era in eras:
            key = 'inputPath' + era
            input_paths_dict[key] = tauID_training[key]
        return input_paths_dict

    def load_era_keys(self, keys):
        samples = keys.keys()
        era_wise_keys = {'keys16': [], 'keys17': [], 'keys18': []}
        for sample in samples:
            included_eras = keys[sample]
            if 16 in included_eras:
                era_wise_keys['keys16'].append(sample)
            if 17 in included_eras:
                era_wise_keys['keys17'].append(sample)
            if 18 in included_eras:
                era_wise_keys['keys18'].append(sample)
        return era_wise_keys

    def load_trainvars(self):
        trainvars_path = os.path.join(self.info_dir, 'trainvars.json')
        trainvars = []
        with open(trainvars_path, 'rt') as in_file:
            for line in in_file:
                trainvars.append(str(json.loads(line)['key']))
        return trainvars

    def interpret_info_file(self):
        info_path = os.path.join(self.info_dir, 'info.json')
        info_dict = ut.read_json_cfg(info_path)
        self.parameters = {}
        tau_id_application = info_dict.pop('tauID_application')
        default_tauID = info_dict['default_tauID_application']
        tau_id_training = info_dict['tauID_training_key']
        self.parameters['tauID_application'] = tau_id_application[default_tauID]
        self.parameters.update(self.find_input_paths(info_dict, tau_id_training))
        self.parameters.update(self.load_era_keys(info_dict.pop('keys')))
        self.parameters['trainvars'] = self.load_trainvars()
        self.parameters['all_trainvar_info'] = self.read_trainvar_info(
            self.all_trainvars_path)
        self.parameters.update(info_dict)

