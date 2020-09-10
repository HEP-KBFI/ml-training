from machineLearning.machineLearning import evaluation_tools as et
import numpy as np


def test_calculate_compactness():
    parameter_dict1 = {
        'a': 1,
        'b': 10,
        'c': 5
    }
    parameter_dict2 = {
        'a': 2,
        'b': 20,
        'c': 10
    }
    parameter_dict3 = {
        'a': 3,
        'b': 30,
        'c': 15
    }
    parameter_dicts = [
        parameter_dict1,
        parameter_dict2,
        parameter_dict3
    ]
    result = et.calculate_compactness(parameter_dicts)
    expected = np.sqrt(2./3)/2
    np.testing.assert_almost_equal(
        result,
        expected,
        7
    )


def test_calculate_dict_mean_coeff_of_variation():
    list_dict = {
        'a': [1, 2, 3],
        'b': [10, 20, 30],
        'c': [5, 10, 15]
    }
    result = et.calculate_dict_mean_coeff_of_variation(list_dict)
    expected = np.sqrt(2./3)/2
    np.testing.assert_almost_equal(
        result,
        expected,
        7
    )

