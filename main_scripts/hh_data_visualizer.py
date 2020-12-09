""" Visualises the data """

import os
import json
from machineLearning.machineLearning import hh_parameter_reader as hhpr
from machineLearning.machineLearning import data_visualizer as dv
from machineLearning.machineLearning import hh_tools as hht
from machineLearning.machineLearning import data_loader as dl
from machineLearning.machineLearning import universal_tools as ut


def main():
    channel_dir, info_dir, global_settings = ut.find_settings()
    scenario = global_settings['scenario']
    reader = hpr.HHParameterReader(channel_dir, scenario)
    preferences = reader.parameters
    global_settings['output_dir'] = os.path.expandvars(
        global_settings['output_dir'])
    if not os.path.exists(global_settings['output_dir']):
        os.makedirs(global_settings['output_dir'])
    preferences['trainvars'] = preferences['all_trainvar_info'].keys()
    if os.path.exists(preferences['data_csv']):
        data = pandas.read_csv(preferences['data_csv'])
    else:
        normalizer = hht.HHDataNormalizer
        data_helper = hht.HHDataHelper
        loader = dl.DataLoader(
            data_helper, normalizer, global_settings, preferences
        )
        data = loader.data
    visualizer = dv.DataVisualizer(data, global_settings['output_dir'])






if __name__=='__main__':
    main()
