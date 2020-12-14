""" Visualises the data """

import os
import json
from machineLearning.machineLearning import hh_parameter_reader as hpr
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
    normalizer = hht.HHDataNormalizer
    data_helper = hht.HHDataHelper
    loader = dl.DataLoader(
        data_helper, normalizer, global_settings, preferences, reweigh=False
    )
    data = loader.data
    visualizer = dv.MPLDataVisualizer(data, global_settings['output_dir'])
    visualizer.visualize_data()


if __name__=='__main__':
    main()
