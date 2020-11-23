'''
Call with 'python'

Usage: save_eventYields.py
'''
import os
import docopt
from machineLearning.machineLearning import hh_visualization_tools as hhvt
from machineLearning.machineLearning import data_loader as dl
from machineLearning.machineLearning import hh_parameter_reader as hpr
from machineLearning.machineLearning import hh_tools as hht
from machineLearning.machineLearning import eventYield_creator as eyc
from machineLearning.machineLearning import universal_tools as ut


def main(bdtClass='evtLevelSUM'):
    cmssw_path = os.path.expandvars('$CMSSW_BASE')
    package_dir = os.path.join(
        cmssw_path,
        'src/machineLearning/machineLearning')
    settings_dir = os.path.join(package_dir, 'settings')
    global_settings = ut.read_settings(settings_dir, 'global')
    modes = ['nonres/default', 'spin0', 'spin2']
    table_infos = []
    output_file = os.path.expandvars(
        os.path.join(global_settings['output_dir'], 'EventYield.tex'))
    for mode in modes:
        global_settings['scenario'] = mode
        channel_dir = os.path.join(
            package_dir,
            'info',
            'HH',
            global_settings['channel']
        )
        reader = hpr.HHParameterReader(channel_dir, mode)
        preferences = reader.parameters
        normalizer = hht.HHDataNormalizer
        data_helper = hht.HHDataHelper
        loader = dl.DataLoader(
            data_helper, normalizer, global_settings, preferences
        )
        mode_data = loader.data
        for era in set(mode_data['era']):
            era_data = mode_data.loc[mode_data['era'] == era]
            channel = global_settings['channel']
            table_creator = eyc.EventYieldTable(era_data, channel, era, scenario)
            table_info = table_creator.create_table()
            table_infos.append(table_info)
    table_writer = eyc.EventYieldsFile(table_infos, output_file)
    table_writer.fill_document_file()
    print('File saved to %s' %output_file)

if __name__ == '__main__':
    main()