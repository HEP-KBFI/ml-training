from machineLearning.machineLearning import slurm_tools as st
import os
import glob
import shutil
import numpy as np
import timeout_decorator
resources_dir = os.path.join(
    os.path.expandvars('$CMSSW_BASE'),
    'src/machineLearning/machineLearning/tests/resources'
)
tmp_folder = os.path.join(resources_dir, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)


def test_parameters_to_file():
    output_dir = os.path.join(tmp_folder, 'slurm')
    parameter_dict1 = {'a': 1, 'b': 2, 'c': 3}
    parameter_dict2 = {'a': 1, 'b': 2, 'c': 3}
    parameter_dict3 = {'a': 1, 'b': 2, 'c': 3}
    parameter_dicts = [
        parameter_dict1,
        parameter_dict2,
        parameter_dict3
    ]
    st.parameters_to_file(output_dir, parameter_dicts)
    wild_card_path = os.path.join(output_dir, '*', '*')
    number_files = len(glob.glob(wild_card_path))
    assert number_files == 3


def test_prepare_job_file():
    global_settings = {
        'ml_method': 'xgb',
        'output_dir': tmp_folder,
        'nthread': 2,
        'process': 'HH',
        'channel': '2l_2tau'
    }
    nthread = 2
    parameter_file = os.path.join(resources_dir, 'xgb_parameters.json')
    sample_dir = os.path.join(resources_dir, 'samples')
    job_nr = 1
    output_dir = os.path.join(resources_dir, 'tmp')
    templateDir = resources_dir
    st.prepare_job_file(
        parameter_file, job_nr, global_settings)
    job_file = os.path.join(resources_dir, 'tmp', 'parameter_1.sh')
    with open(job_file, 'r') as f:
        number_lines = len(f.readlines())
    assert number_lines == 13

def test_check_parameter_file_sizes():
    wild_card_path = os.path.join(resources_dir, 'parameter_*.sh')
    number_zero_sized = st.check_parameter_file_sizes(wild_card_path)
    assert number_zero_sized == 0


def test_read_fitness():
    result = st.read_fitness(resources_dir, fitness_key='d_roc')
    expected = [3, 3, 3]
    assert result == expected


def test_get_sample_nr():
    path = "/foo/1/bar.baz"
    expected = 1
    result = st.get_sample_nr(path)
    assert result == expected


@timeout_decorator.timeout(10)
def test_wait_iteration():
    working = False
    try:
        st.wait_iteration(resources_dir, 2)
    except SystemExit: 
        working = True
    assert working


def test_move_previous_files():
    previous_files_dir = os.path.join(resources_dir, 'previous_files')
    parameter_file = os.path.join(resources_dir, 'parameter_0.sh')
    st.move_previous_files(resources_dir, previous_files_dir)
    wild_card_path = os.path.join(previous_files_dir, 'iteration_*')
    nr_folders_moved = len(glob.glob(wild_card_path))
    parameter_file_deleted = False
    if not os.path.exists(parameter_file):
        parameter_file_deleted = True
    assert nr_folders_moved == 4
    assert parameter_file_deleted


def test_find_iter_number():
    iter_3_dir = os.path.join(
        resources_dir, 'previous_files', 'iteration_3')
    sample_dir = os.path.join(resources_dir, 'samples')
    if os.path.exists(iter_3_dir):
        shutil.copytree(iter_3_dir, sample_dir)
        shutil.rmtree(iter_3_dir)
    previous_files_dir = os.path.join(resources_dir, 'previous_files')
    iter_nr = st.find_iter_number(previous_files_dir)
    assert iter_nr == 3


def test_check_error():
    error = False
    try:
        st.check_error(resources_dir)
    except:
        error = True
    assert error


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    parameter_file = os.path.join(resources_dir, 'parameter_0.sh')
    with open(parameter_file, 'wt') as outfile:
        outfile.write('foobar')