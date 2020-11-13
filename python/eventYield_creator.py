import os


class EventYieldTable:
    '''Creates Event yield table for a given data'''

    def __init__(self, data, channel, era, scenario, file_format='beamer'):
        '''

        Parameters:
        ----------
        data : pandas.DataFrame
            Data to create the event yields
        channel : str
            Channel for which the table is created
        era : int or str
            Era of the data
        scenario : str
            Scenario of the given data. e.g 'nonRes', 'spin0', 'spin2'
        file_format : str
            Format oif the ouput file. Options:
                1) 'beamer' [Default]
                2) 'document'
        '''
        self.data = data
        self.channel = channel
        self.file_format = file_format
        self.era = str(era)
        self.scenario = scenario
        self.set_table_size(file_format)
        self.categorize_events()

    def initialize_table(self):
        ''' Initializes the table with a header'''
        self.table = [
            '\\begin{table}',
            '\\centering',
            self.font_size,
            '\\begin{tabular}{| c | c | c | c | c | c |}',
            'Key & Signal & Bkg & negWeights & totalWeight & eventWeight\\\\'
            '\\hline'
        ]

    def set_table_size(self, file_format):
        '''Sets the font size in the table based on the file_format'''
        if file_format == 'beamer':
            self.font_size = '\\tiny'
        elif file_format == 'document':
            self.font_size = '\\fontsize{7pt}{8}\\selectfont'
        else:
            print('Warning: invalid file_format selected. Defaulting to beamer')
            self.font_size = '\\tiny'

    def save_current_table(self, output_dir='$PWD'):
        ''' Saves the LaTeX file to a file without any LaTeX file preamble.

        Parameters:
        -----------
        output_dir : str
            Directory of the output LaTeX file. If value not provided, table
            will be saved to the $PWD [Default: None]
        '''
        output_dir = os.path.expandvars(output_dir)
        output_path = os.path.join(output_dir, self.era + '_eventYields.tex')
        with open(output_path, 'wt') as out_file:
            for row in self.table:
                out_file.write(row + '\n')

    def finalize_table(self, split_end):
        ''' Finalizes the table

        Parameters:
        -----------
        split_end : bool
            In case of beamer, not the whole table always fits on one slide.
            This is to split the table and not add the row for the total
            yields
        '''
        if not split_end:
            self.table.append('\n\\hline')
            self.table.append(self.total_row)
            self.table.append('\n\\hline')
        self.table.extend([
            '\\end{tabular}'
            '\\end{table}'
        ])

    def categorize_events(self):
        '''Categorizes events to signal and background classes'''
        keys = set(self.data['process'])
        self.process_infos = {}
        for key in keys:
            info = {}
            condition_key = self.data['process'].values == key
            info['signal'] = len(
                self.data.loc[
                    (self.data['target'].values == 1) & condition_key
                ]
            )
            info['background'] = len(
                self.data.loc[
                    (self.data['target'].values == 0) & condition_key
                ]
            )
            info['neg_weight_events'] = len(
                self.data.loc[
                    (self.data['totalWeight'].values < 0) & condition_key
                ]
            )
            info['total_weight'] = self.data.loc[
                    (self.data['process'].values == key)]['totalWeight'].sum()
            info['event_weight'] = self.data.loc[
                    (self.data['process'].values == key)]['evtWeight'].sum()
            self.process_infos[key] = info

    def create_main_table_content(self):
        self.main_content = []
        processes = sorted(self.process_infos.keys())
        for process in processes:
            info = self.process_infos[process]
            self.main_content.append(
                '%s & %s & %s & %s & %s & %s \\\\' %(
                    process.replace('_', '\\_'),
                    info['signal'],
                    info['background'],
                    info['neg_weight_events'],
                    info['total_weight'],
                    info['event_weight']
                )
            )
        self.total_row = '%s & %s & %s & %s & %s & %s \\\\' %(
            'Total',
            sum([self.process_infos[key]['signal'] for key in processes]),
            sum([self.process_infos[key]['background'] for key in processes]),
            sum([self.process_infos[key]['neg_weight_events'] for key in processes]),
            sum([self.process_infos[key]['total_weight'] for key in processes]),
            sum([self.process_infos[key]['event_weight'] for key in processes]),
        )

    def create_table(self):
        tables = []
        self.create_main_table_content()
        while len(self.main_content) > 25 and self.file_format == 'beamer':
            self.initialize_table()
            table_fraction = self.main_content[:26]
            self.table.extend(table_fraction)
            self.finalize_table(True)
            self.main_content = self.main_content[26:]
            tables.append(self.table)
        self.initialize_table()
        self.table.extend(self.main_content)
        self.finalize_table(False)
        tables.append(self.table)
        self.table_info = {
            'tables': tables,
            'era': self.era,
            'channel': self.channel,
            'scenario': self.scenario
        }
        return self.table_info


class EventYieldsFile:
    def __init__(self, table_infos, output_file, file_format='beamer'):
        ''' Creates a LaTeX file with the provided event yield tables

        Parameters:
        ----------
        table_infos : list of dicts
            List containing dictionarys with the table and it's info
        output_file : file where the table(s) is (are) saved
        file_format : str
            Format oif the ouput file. Options:
                1) 'beamer' [Default]
                2) 'document'
        '''
        self.table_infos = table_infos
        self.output_file = output_file
        self.file_format = file_format
        self.main_content = []
        self.create_preamble()

    def create_preamble(self):
        ''' Creates the preamble for the LaTeX document depending on the
        file_format
        '''
        if self.file_format == 'beamer':
            self.preamble = [
                '\\documentclass{beamer}',
                '\\usetheme{Madrid}',
                '\\usecolortheme{beaver}',
                '\\usepackage{textpos}',
                '\\usepackage[english]{babel}',
                '\\usepackage[utf8]{inputenc}',
                '\\usepackage{graphicx}',
                '\\usepackage{grffile}',
                '\\usepackage{tabularx}',
                '\\usepackage{xcolor, colortbl}',
                '\\usepackage{transparent}',
                '\\usepackage{array}',
                '\\usepackage{booktabs}',
                '\n',
                '\\title[%s]{%s event yields}' %('Channel', 'Channel'),
                '\\author{The Presenter\\\\{\\small email@email.com}}',
                '\\institute[NICPB]{National Institute of Chemical Physics and Biophysics}',
                '\n',
                '\\begin{document}',
                '\n'
            ]
        elif self.file_format == 'document':
            self.preamble = [
                '\\documentclass[a4paper,12pt,oneside]{report}',
                '\n',
                '\\begin{document}',
                '\n'
            ]
        else:
            print('Warning: Incorrect file_format.')

    def fill_document_file(self):
        ''' Fill the file with the tables given by the table_infos'''
        with open(self.output_file, 'wt') as out_file:
            for row in self.preamble:
                out_file.write(row + '\n')
            out_file.write(4 * '\n')
            for table_info in self.table_infos:
                if self.file_format == 'beamer':
                    table_to_write = self.beamer_table_wrapping(table_info)
                else:
                    table_to_write = self.document_table_wrapping(table_info)
                for row in table_to_write:
                    out_file.write(row + '\n')
            out_file.write('\\end{document}')

    def beamer_table_wrapping(self, table_info):
        ''' Wraps the table into a frame '''
        table_to_write = []
        for table in table_info['tables']:
            table_to_write.extend([
                '\\begin{frame}{Channel: %s, Era: %s, Scenario: %s}' %(
                    table_info['channel'].replace('_', '\\_'),
                    table_info['era'],
                    table_info['scenario'])
            ])
            table_to_write.extend(table)
            table_to_write.extend(['\\end{frame}'])
        return table_to_write

    def document_table_wrapping(self, table_info):
        ''' Wraps the table with subsections '''
        table_to_write = []
        table_to_write.extend([
            '\\subsection{Channel: %s, Era: %s, Scenario: %s}' %(
                table_info['channel'].replace('_', '\\_'),
                table_info['era'],
                table_info['scenario'])
        ])
        for table in table_info['tables']:
            table_to_write.extend(table)
            table_to_write.append('\n')
        return table_to_write


def create_event_yields(data, channel, scenario, output_dir):
    ''' Creates the event yield file for a given data and outputs it to a
    given directory

    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing the data
    channel : str
        Channel for which the data belongs
    scenario : str
        spin0, spin2, nonRes.
    output_dir : directory where the event yields are saved
    '''
    table_infos = []
    output_file = os.path.join(output_dir, 'EventYield.tex')
    for era in set(data['era']):
        era_data = data.loc[data['era'] == era]
        table_creator = EventYieldTable(era_data, channel, era, scenario)
        table_info = table_creator.create_table()
        table_infos.append(table_info)
    table_writer = EventYieldsFile(table_infos, output_file)
    table_writer.fill_document_file()
    print('File saved to %s' %output_file)
