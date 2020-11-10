'''
Call with 'python'

Usage: save_eventYields.py
'''
import os
import docopt
from machineLearning.machineLearning import hh_aux_tools as hhat
from machineLearning.machineLearning import eventYield_creator as eyc


def main(bdtClass='evtLevelSUM')
    cmssw_path = os.path.expandvars('$CMSSW_BASE')
    package_dir = os.path.join(
        cmssw_path,
        'src/machineLearning/machineLearning')
    settings_dir = os.path.join(package_dir, 'settings')
    global_settings = ut.read_settings(settings_dir, 'global')
    global_settings['debug'] = False
    modes = ['nonRes', 'res/spin0', 'res/spin2']
    table_infos = []
    output_file = os.path.join(global_settings['output_dir'], 'EventYield.tex')
    for mode in modes:
        if mode == 'nonRes':
            global_settings['bdtType'] = '_'.join(
                [bdtClass, 'HH', global_settings['channel'], 'nonres']
            )
        else:
            global_settings['bdtType'] = '_'.join(
                [bdtClass, 'HH', global_settings['channel'], 'res'])
        channel_dir = os.path.join(
            package_dir,
            'info',
            'HH',
            global_settings['channel']
        )
        info_dir = os.path.join(channel_dir, mode)
        preferences = hhat.get_hh_parameters(
            channel_dir,
            global_settings['tauID_training'],
            info_dir
        )
        mode_data = hhat.load_hh_data(preferences, global_settings)
        for era in set(mode_data['era']):
            era_data = mode_data.loc[data['era'] == era]
            table_creator = EventYieldTable(era_data, channel, era, scenario)
            table_info = table_creator.create_table()
            table_infos.append(table_info)
    table_writer = EventYieldsFile(table_infos, output_file)
    table_writer.fill_document_file()
    print('File saved to %s' %output_file)

if __name__ == '__main__':
    main()