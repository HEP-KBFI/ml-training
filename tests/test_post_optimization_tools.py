from machineLearning.machineLearning import post_optimization_tools as pot
import os
import numpy as np

resources_dir = os.path.join(
    os.path.expandvars('$CMSSW_BASE'),
    'src/machineLearning/machineLearning/tests/resources'
)

def test_create_result_lists():
    output_dir = os.path.join(resources_dir)
    result = pot.create_result_lists(output_dir, 'pred_test')
    expected = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ], dtype=int)
    assert (result == expected).all()


# def test_lists_from_file():