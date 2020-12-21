""" The tools for performing feature selection based on performance for
BDT (XGBoost), neural net (NN) and Lorentz Boost Network (LBN)
"""
import numpy as np
import json
import os
from machineLearning.machineLearning import hh_tools as hht
from machineLearning.machineLearning import xgb_tools as xt
from machineLearning.machineLearning import nn_tools as nt


class TrainvarOptimizer(object):
    """ The base class used for feature selection optimization """
    def __init__(
            self, data, preferences, global_settings, corr_threshold=0.8,
            min_nr_trainvars=10, step_size=5, weight='totalWeight'
    ):
        """ Initializes the trainvar optimizer"""
        self.data = data
        self.preferences = preferences
        self.global_settings = global_settings
        self.corr_threshold = corr_threshold
        self.min_nr_trainvars = min_nr_trainvars
        self.step_size = step_size
        self.tracker = {}
        self.weight = weight

    def optimization(self, trainvars):
        """ The main part of the trainvar optimization, where the least
        performing features are dropped until a requested number of features
        is left

        Args:
            trainvars: list
                The list of trainvars that is left after removing the highly
                correlated trainvars

        Returns:
            trainvars: list
                The list of optimized trainvars with the length of the
                requested size of 'min_nr_trainvars'
        """
        data_dict = {'train': self.data, 'trainvars': trainvars}
        iteration = 0
        while len(trainvars) > self.min_nr_trainvars:
            feature_importances = self.get_feature_importances(data_dict)
            trainvars = self.drop_not_used_variables(
                feature_importances, trainvars, iteration
            )
            trainvars = self.drop_worst_performing_ones(
                feature_importances, trainvars, iteration
            )
            iteration += 1
        del data_dict
        return trainvars

    def drop_not_used_variables(
            self, feature_importances, trainvars, iteration
    ):
        """ Drops the trainvars that weren't used in the training at all

        Args:
            feature_importances: dict
                Dictionary containing the feature importance of each trainvar
            trainvars: list
                The list of trainvars that were used for creating the model
            iteration: int
                Number of the iteration
        """
        used_features = feature_importances.keys()
        for trainvar in trainvars:
            if trainvar not in used_features:
                if 'nonres' in self.global_settings['scenario']:
                    if trainvar not in self.preferences['nonResScenarios']:
                        print(
                            'Removing  -- %s -- due to it not being used at \
                            all' %(trainvar))
                        self.feature_drop_tracking(
                            'optimization_%s' % iteration, trainvar,
                            feature_importances[trainvar], feature_importances
                        )
                        trainvars.remove(trainvar)
        return trainvars

    def drop_worst_performing_ones(
            self, feature_importances, trainvars, iteration
    ):
        """ Drops the least performing trainvars by the given step size

        Args:
            feature_importances: dict
                Dictionary containing the feature importance of each trainvar
            trainvars: list
                The list of trainvars that were used for creating the model
            iteration: int
                Number of the iteration
        """
        finished = False
        if 'nonres' in self.global_settings['scenario']:
            BM_in_trainvars = False
            for nonRes_scenario in self.preferences['nonResScenarios']:
                if nonRes_scenario in trainvars:
                    BM_in_trainvars = True
                    break
            if BM_in_trainvars:
                keys = list(feature_importances.keys())
                sumBM = 0
                for nonRes_scenario in self.preferences['nonResScenarios']:
                    try:
                        sumBM += feature_importances[nonRes_scenario]
                        keys.remove(nonRes_scenario)
                    except KeyError:
                        continue
                keys.append('sumBM')
                feature_importances['sumBM'] = sumBM
                values = [feature_importances[key] for key in keys]
                if len(keys) <= (min_nr_trainvars + step_size):
                    step_size = len(trainvars) - self.min_nr_trainvars
                    finished = True
                index = np.argpartition(values, self.step_size)[:step_size]
                n_worst_performing = np.array(keys)[index]
                for element in n_worst_performing:
                    if element == 'sumBM':
                        print('Removing benchmark scenarios')
                        for nonRes_scenario in preferences['nonResScenarios']:
                            trainvars.remove(nonRes_scenario)
                        self.feature_drop_tracking(
                            'optimization_%s' % iteration, element,
                            feature_importances[element], feature_importances,

                        )
                    else:
                        print('Removing ' + str(element))
                        self.feature_drop_tracking(
                            'optimization_%s' % iteration, element,
                            feature_importances[element], feature_importances
                        )
                        trainvars.remove(element)
            else:
                trainvars = self.remove_nonBM_trainvars(
                    trainvars, feature_importances, iteration
                )
        else:
            trainvars = self.remove_nonBM_trainvars(
                trainvars, feature_importances, iteration)
        if finished:
            self.feature_drop_tracking('', '', '', '', finished=True)
        return trainvars

    def remove_nonBM_trainvars(
            self, trainvars, feature_importances, iteration
    ):
        """ Removes the least performing features from the list for the case
        if the trainvar does not belong to the nonResScenarios.

        Args:
            trainvars: list
                The list of trainvars that were used for creating the model
            feature_importances: dict
                Dictionary containing the feature importance of each trainvar

        Returns:
            trainvars: list
                The list of trainvars that were used for creating the model
        """
        if len(trainvars) < (self.min_nr_trainvars + self.step_size):
            step_size = len(trainvars) - self.min_nr_trainvars
        keys = np.array(feature_importances.keys())
        values = np.array(feature_importances.values())
        index = np.argpartition(values, self.step_size)[:self.step_size]
        n_worst_performing = keys[index]
        for element in n_worst_performing:
            print('Removing ' + str(element))
            self.feature_drop_tracking(
                'optimization_%s' % iteration, element,
                feature_importances[element], feature_importances
            )
            trainvars.remove(element)
        return trainvars

    def drop_highly_currelated_variables(self, trainvars_initial):
        """ Since having two highly correlated variables doesn't benefit
        the model, one of them should be removed. The one to be removed is
        chosen to be the one which performes more poorly

        Args:
            trainvars_initial: list
                List of trainvars whose correlations will be checked and
                highly correlated ones will be removed.

        Returns:
            trainvars: list
                List of trainvars with the highest correlated ones removed.
        """
        correlations = self.data[trainvars_initial].corr()
        trainvars = list(trainvars_initial)
        for trainvar in trainvars:
            trainvars_copy = list(trainvars)
            trainvars_copy.remove(trainvar)
            for item in trainvars_copy:
                corr_value = abs(correlations[trainvar][item])
                if corr_value > self.corr_threshold:
                    trainvars.remove(item)
                    print(
                        "Removing " + str(item) + ". Correlation with "
                        + str(trainvar) + " is " + str(corr_value)
                    )
                    self.feature_drop_tracking(
                        'correlations', item,
                        {}, {}
                    )
        del correlations
        return trainvars

    def update_trainvars(self, trainvars):
        """ If the trainvars used for the parametrization are not in the list
        of trainvars after optimization, these will be appended to the end of
        the list - otherwise a parametrized training will be impossible

        Args:
            trainvars: list
                List of trainvars where the features needed for parametrization
                are added if missing
        """
        if 'nonres' in self.global_settings['scenario']:
            for scenario in self.preferences['nonResScenarios']:
                if scenario not in trainvars:
                    trainvars.append(scenario)
        else:
            if 'gen_mHH' not in trainvars:
                trainvars.append('gen_mHH')
        return trainvars

    def check_trainvars_integrity(self, trainvars):
        """ Checks if all the necessary trainvars are included for the model
        and the optimization, e.g for nonres/default scenario all BM points
        are added to the list of all trainvars to be used

        Args:
            trainvars: list
                The list of trainvars whos integrity is to be checked
        """
        if 'nonres' in self.global_settings['scenario']:
            for nonRes_scenario in self.preferences['nonResScenarios']:
                if nonRes_scenario not in trainvars:
                    trainvars.append(nonRes_scenario)
                    self.preferences['trainvar_info'].update(
                        {nonRes_scenario: 1}
                    )
        else:
            if 'gen_mHH' not in trainvars:
                trainvars.append('gen_mHH')
                self.preferences['trainvar_info'].update(
                    {'gen_mHH': 1}
                )
        return trainvars

    def save_optimized_trainvars(self, trainvars_file):
        """ Saves the given number of best performing trainvars to the
        provided file

        Args:
            trainvars_file: str
                Path to the file where trainvars will be written
        """
        with open(trainvars_file, 'wt') as outfile:
            for trainvar in self.trainvars:
                trainvar_info = self.preferences['all_trainvar_info']
                if trainvar in trainvar_info.keys():
                    true_int = trainvar_info[trainvar]
                else:
                    true_int = 1
                trainvar_dict = {
                    'key': trainvar,
                    'true_int': true_int
                }
                json.dump(trainvar_dict, outfile)
                outfile.write('\n')

    def set_outfile_path(self):
        """ Finds the correct path where trainvars are to be saved

        Returns:
            trainvars_file: str
                The path to the trainvars.json file of the given scenario
        """
        cmssw_base = os.path.expandvars('$CMSSW_BASE')
        scenario = self.global_settings['scenario']
        scenario = 'res/' + scenario if 'nonres' not in scenario else scenario
        scenario_dir = os.path.join(
            cmssw_base, 'src/machineLearning/machineLearning/info',
            self.global_settings['process'], self.global_settings['channel'],
            scenario
        )
        trainvars_file = os.path.join(scenario_dir, 'trainvars.json')
        return trainvars_file

    def get_feature_importances(self, data_dict):
        """ Stub for get_feature_importances

        Args:
            data_dict: dict
                Contains the keys 'train' with the data to be used for creating
                the model, and 'trainvars' with the list of features to be
                used from this data

        Returns:
            feature_importances: dict
                Dictionary containining the feature names as keys and the
                importance score as the value for the feature.
        """
        raise NotImplementedError(
            'Please implement get_feature_importances for your sub-class'
        )

    def trainvar_importance_ordering(self):
        """ Before removing the highly correlated variables from the correlated
        pairs, we order the trainvars according to the importance of the
        features. This is done to remove the less important variable from the
        pair, since otherwise some performance can be lost.

        Returns:
            ordered_features: list
                List of trainvars to be used that is ordered by the importance
                of each variable.
        """
        data_dict = {
            'train': self.data,
            'trainvars': list(self.preferences['all_trainvar_info'].keys())
        }
        feature_importances = self.get_feature_importances(data_dict)
        ordered_features = [
            i[0] for i in sorted(
                feature_importances.items(), key=lambda x: -x[1]
            )
        ]
        del data_dict
        return ordered_features

    def feature_drop_tracking(
            self, step_name, feature, value, feature_importances,
            finished=False
    ):
        """ Tracks the dropping of the features & saves the trackinginfo to
        a file when the optimization is finished

        Args:
            step_name: str
                The name of the step when the feature is dropped
            feature: str
                Name of the feature that was dropped
            value: float
                The feature importance of the variable that was dropped
            finished: bool
                [default: False] Whether the optimization is finished
        """
        if not finished:
            tracking= {
                'feature': feature,
                'feature_importance': value
            }
            if step_name not in self.tracker.keys():
                self.tracker[step_name] = {
                    'feature_importances': feature_importances,
                    'dropped': [tracking]
                }
            else:
                self.tracker[step_name]['dropped'].append(tracking)
        else:
            tracking_file = os.path.join(
                os.path.expandvars(self.global_settings['output_dir']),
                'trainvarOpt_tracking.log'
            )
            with open(tracking_file, 'wt') as out_file:
                for track in self.tracker:
                    out_file.write(track + '\n')

    def optimization_collector(self):
        """ Collects all the necessary components of the trainvar optimization
        together to be executed """
        ordered_trainvars = self.trainvar_importance_ordering()
        trainvars = self.drop_highly_currelated_variables(ordered_trainvars)
        trainvars = self.optimization(trainvars)
        self.trainvars = self.update_trainvars(trainvars)
        trainvars_file = self.set_outfile_path()
        self.save_optimized_trainvars(trainvars_file)


class XGBTrainvarOptimizer(TrainvarOptimizer):
    """ The XGBoost flavor wrapper for the TrainvarOptimizer"""
    def __init__(
            self, data, preferences, global_settings, hyperparameters,
            corr_threshold=0.8, min_nr_trainvars=10, step_size=5,
            weight='totalWeight'
    ):
        """ Initializes the XGBoost version of the trainvar optimizer

        Args:
            data: pandas.DataFrame
                Dataframe containing the full data
            preferences: dict
                Dictionary containing the preferences for a given scenario
            global_settings: dict
                Dictionary containing the preferences for the given run
        """
        super(XGBTrainvarOptimizer, self).__init__(
            data, preferences, global_settings, corr_threshold,
            min_nr_trainvars, step_size, weight
        )
        self.hyperparameters = hyperparameters

    def get_feature_importances(self, data_dict):
        """ Stub for get_feature_importances

        Args:
            data_dict: dict
                Contains the keys 'train' with the data to be used for creating
                the model, and 'trainvars' with the list of features to be
                used from this data

        Returns:
            feature_importances: dict
                Dictionary containining the feature names as keys and the
                importance score as the value for the feature.
        """
        model = xt.create_model(
            self.hyperparameters, data_dict, self.global_settings['nthread'],
            'auc', self.weight
        )
        feature_importances = model.get_booster().get_fscore()
        del model
        return feature_importances


class NNTrainvarOptimizer(TrainvarOptimizer):
    """ The neural network flavor wrapper for the TrainvarOptimizer"""
    def __init__(
            self, data, preferences, global_settings, hyperparameters,
            corr_threshold=0.8, min_nr_trainvars=10, step_size=5,
            weight='totalWeight'
    ):
        """ Initializes the neural net version of the trainvar optimizer

        Args:
            data: pandas.DataFrame
                Dataframe containing the full data
            preferences: dict
                Dictionary containing the preferences for a given scenario
            global_settings: dict
                Dictionary containing the preferences for the given run
        """
        super(XGBTrainvar_optimizer, self).__init__(
            data, preferences, global_settings, corr_threshold,
            min_nr_trainvars, step_size, weight
        )
        self.hyperparameters = hyperparameters

    def get_feature_importances(self, data_dict):
        """ Stub for get_feature_importances

        Args:
            data_dict: dict
                Contains the keys 'train' with the data to be used for creating
                the model, and 'trainvars' with the list of features to be
                used from this data

        Returns:
            feature_importances: dict
                Dictionary containining the feature names as keys and the
                importance score as the value for the feature.
        """
        importance_calculator = nt.NNFeatureImportances(
            self.model, self.data, self.trainvars, self.weight,
            self.target, permutations=self.permutations
        )
        feature_importances = importance_calculator.permutation_importance()
        return feature_importances


class LBNTrainvarOptimizer(TrainvarOptimizer):
    """ The Lorentz Boost Network flavor wrapper for the TrainvarOptimizer"""
    def __init__(
            self, data, preferences, global_settings, particles,
            corr_threshold=0.8, min_nr_trainvars=10, step_size=5,
            weight='totalWeight'
    ):
        """ Initializes the Lorentz Boost Network version of the trainvar
        optimizer

        Args:
            data: pandas.DataFrame
                Dataframe containing the full data
            preferences: dict
                Dictionary containing the preferences for a given scenario
            global_settings: dict
                Dictionary containing the preferences for the given run
        """
        super(XGBTrainvar_optimizer, self).__init__(
            data, preferences, global_settings, corr_threshold,
            min_nr_trainvars, step_size, weight
        )
        self.particles = particles


    def get_feature_importances(self, data_dict):
        """ Stub for get_feature_importances

        Args:
            data_dict: dict
                Contains the keys 'train' with the data to be used for creating
                the model, and 'trainvars' with the list of features to be
                used from this data

        Returns:
            feature_importances: dict
                Dictionary containining the feature names as keys and the
                importance score as the value for the feature.
        """
        importance_calculator = nt.LBNFeatureImportances(
            self.model, self.data, self.trainvars, self.weight,
            self.target, self.particles,
            permutations=self.permutations
        )
        feature_importances = importance_calculator.permutation_importance()
        return feature_importances

