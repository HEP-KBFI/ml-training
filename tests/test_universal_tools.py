from machineLearning.machineLearning import universal_tools as ut
import os
import numpy as np



def test_read_parameters():
    path_to_test_file = os.path.join(
        dir_path, 'resources', 'best_parameters.json')
    result = ut.read_parameters(path_to_test_file)
    expected = [
        {'a': 1, 'b': 2, 'c': 3},
        {'stuff': 1}
    ]
    assert result == expected


def test_read_settings():
    pso_settings = ut.read_settings('pso')
    global_settings = ut.read_settings('global')
    assert len(pso_settings.keys()) == 7
    assert len(global_settings.keys()) == 14


def test_to_one_dict():
    list_of_dicts = [{'foo': 1}, {'bar': 2}, {'baz': 3}]
    all_in_one_dict = ut.to_one_dict(list_of_dicts)
    expected = {'foo': 1, 'bar': 2, 'baz': 3}
    assert all_in_one_dict == expected
