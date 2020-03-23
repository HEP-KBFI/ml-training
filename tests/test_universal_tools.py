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
    result = ut.read_parameters(path_to_test_file)
    expected = [
        {'a': 1, 'b': 2, 'c': 3},
        {'stuff': 1}
    ]
    assert result == expected


def test_read_settings():
    pso_settings = ut.read_settings(settings_dir, 'pso')
    global_settings = ut.read_settings(settings_dir, 'global')
    assert len(pso_settings.keys()) == 7
    assert len(global_settings.keys()) == 14


def test_to_one_dict():
    list_of_dicts = [{'foo': 1}, {'bar': 2}, {'baz': 3}]
    all_in_one_dict = ut.to_one_dict(list_of_dicts)
    expected = {'foo': 1, 'bar': 2, 'baz': 3}
    assert all_in_one_dict == expected
