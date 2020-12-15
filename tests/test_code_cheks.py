import os
from pathlib import Path

INFO_DIR = os.path.dirname(os.path.realpath(__file__)).replace('tests', 'info')
file_list = []
for directory, subdirectories, files in os.walk(INFO_DIR):
    for filename in files:
        if filename.endswith('.json'):
            file_list.append(os.path.join(directory, filename))


def test_existing_tabs():
    faulty_files = []
    for file_path in file_list:
        with open(file_path, 'rt') as in_file:
            if '\t' in in_file.read():
                faulty_files.append(file_path)
    assert len(faulty_files) == 0, 'Files containing TAB character: %s' %(
        faulty_files
    )

