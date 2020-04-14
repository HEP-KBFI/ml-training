'''Tools for collecting and processing the created files'''
import numpy as np
import csv
import glob
import os
from machineLearning.machineLearning import slurm_tools as st


def create_result_lists(output_dir, pred_type):
    '''Creates the result list that is ordered by the sample number

    Parameters:
    ----------
    output_dir : str
        Path to the directory of the output
    pred_type : str
        Type of the prediction (pred_test or pred_train)

    Returns:
    -------
    ordering_list : list
        Ordered list of the results
    '''
    samples = os.path.join(output_dir, 'samples')
    wild_card_path = os.path.join(samples, '*', pred_type + '.lst')
    ordering_list = []
    for path in glob.glob(wild_card_path):
        sample_nr = st.get_sample_nr(path)
        row_res = lists_from_file(path)
        ordering_list.append([sample_nr, row_res])
    ordering_list = sorted(ordering_list, key=lambda x: x[0])
    ordering_list = np.array([i[1] for i in ordering_list], dtype=float)
    return ordering_list


def lists_from_file(path):
    '''Creates a list from a file that contains data on different rows

    Parameters:
    ----------
    path : str
        Path to the file

    Returns:
    -------
    row_res : list
        List with the data
    '''
    with open(path, 'r') as file:
        rows = csv.reader(file)
        row_res = []
        for row in rows:
            row_res.append(row)
    return row_res
