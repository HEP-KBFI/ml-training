from machineLearning.machineLearning import universal_tools as ut
import os
import numpy as np

resources_dir = os.path.join(
    os.path.expandvars('$CMSSW_BASE'),
    'src/machineLearning/machineLearning/tests/resources'
)

settings_dir = os.path.join(
    os.path.expandvars('$CMSSW_BASE'),
    'src/machineLearning/machineLearning/settings')


def test_read_parameters():
    path_to_test_file = os.path.join(
        resources_dir, 'best_parameters.json')
    result = ut.read_json_cfg(path_to_test_file)
    expected = {'a': 1, 'b': 2, 'c': 3}
    assert result == expected