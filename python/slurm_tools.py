''' Toolset for computation with slurm
'''
import time
from pathlib import Path
import os
import subprocess
import json
import csv
import glob
from shutil import copyfile
import shutil
import numpy as np
from machineLearning.machineLearning import universal_tools as ut


def get_fitness_score(
        hyperparameter_sets,
        global_settings,
        sample_size=0
):
    '''The main function call that is the slurm equivalent of ensemble_fitness
    in xgb_tools

    Parameters:
    ----------
    hyperparameter_sets : list of dicts
        Parameter-sets for all particles
    global_settings : dict
        Global settings for the hyperparameter optimization
    sample_size: integer
        Sample size in case where it does not correspond to the value given
        in the settings file

    Returns:
    -------
    scores : list of floats
        Fitnesses for each hyperparameter-set
    '''
    output_dir = os.path.expandvars(global_settings['output_dir'])
    previous_files_dir = os.path.join(output_dir, 'previous_files')
    if not os.path.exists(previous_files_dir):
        os.makedirs(previous_files_dir)
    settings_dir = os.path.join(output_dir, 'run_settings')
    if sample_size == 0:
        opt_settings = ut.read_settings(
            settings_dir, global_settings['optimization_algo'])
        sample_size = opt_settings['sample_size']
    parameters_to_file(output_dir, hyperparameter_sets)
    wild_card_path = os.path.join(
        output_dir, 'samples', '*', 'parameters.json')
    zero_sized = 1
    while zero_sized != 0:
        zero_sized = check_parameter_file_sizes(wild_card_path)
        time.sleep(2)
    for parameter_file in glob.glob(wild_card_path):
        sample_nr = get_sample_nr(parameter_file)
        job_file = prepare_job_file(
            parameter_file, sample_nr, global_settings
        )
        subprocess.call(['sbatch', job_file])
    wait_iteration(output_dir, sample_size)
    time.sleep(30)
    scores = read_fitness(output_dir, global_settings['fitness_fn'])
    move_previous_files(output_dir, previous_files_dir)
    return scores


def parameters_to_file(output_dir, hyperparameter_sets):
    '''Saves the parameters to the subdirectory (name=sample number) of the
    output_dir into a parameters.json file

    Parameters:
    ----------
    output_dir : str
        Path to the output directory
    hyperparameter_sets : list dicts
        Parameter-sets of all particles

    Returns:
    -------
    Nothing
    '''
    samples = os.path.join(output_dir, 'samples')
    if not os.path.exists(samples):
        os.makedirs(samples)
    for number, parameter_dict in enumerate(hyperparameter_sets):
        nr_sample = os.path.join(samples, str(number))
        if not os.path.exists(nr_sample):
            os.makedirs(nr_sample)
        parameter_file = os.path.join(nr_sample, 'parameters.json')
        with open(parameter_file, 'w') as file:
            json.dump(parameter_dict, file)


def prepare_job_file(
        parameter_file,
        sample_nr,
        global_settings
):
    '''Writes the job file that will be executed by slurm

    Parameters:
    ----------
    parameter_file : str
        Path to the parameter file
    sample_nr : int
        Number of the sample (parameter-set)
    global_settings : dict
        Global settings for the run

    Returns:
    -------
    job_file : str
        Path to the script to be executed by slurm
    '''
    main_dir = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning')
    output_dir = os.path.expandvars(global_settings['output_dir'])
    template_dir = os.path.join(main_dir, 'settings')
    job_file = os.path.join(output_dir, 'parameter_' + str(sample_nr) + '.sh')
    template_file = os.path.join(template_dir, 'submit_template.sh')
    error_file = os.path.join(output_dir, 'error' + str(sample_nr))
    output_file = os.path.join(output_dir, 'output' + str(sample_nr))
    file_title = '_'.join([
        'slurm', global_settings['ml_method'], global_settings['process']])
    batch_job_file = file_title + '.py'
    run_script = os.path.join(main_dir, 'evaluation_scripts', batch_job_file)
    copyfile(template_file, job_file)
    with open(job_file, 'a') as filehandle:
        filehandle.writelines('''
#SBATCH --cpus-per-task=%s
#SBATCH -e %s
#SBATCH -o %s
env
python %s --parameter_file %s --output_dir %s
        ''' % (global_settings['nthread'], error_file, output_file, run_script,
               parameter_file, output_dir))
    return job_file


def check_parameter_file_sizes(wild_card_path):
    '''Checks all files in the wild_card_path for their size. Returns the
    number of files with zero size

    Paramters:
    ---------
    wild_card_path : str
        Wild card path for glob to parse

    Returns:
    -------
    zero_sized : int
        Number of zero sized parameter files
    '''
    zero_sized = 0
    for parameter_file in glob.glob(wild_card_path):
        size = os.stat(parameter_file).st_size
        if size == 0:
            zero_sized += 1
    return zero_sized


def read_fitness(output_dir, fitness_key='d_roc'):
    '''Creates the list of score dictionaries of each sample. List is ordered
    according to the number of the sample

    Parameters:
    ----------
    output_dir : str
        Path to the directory of output

    Returns:
    -------
    scores : list of floats
        List of fitnesses
    '''
    samples = os.path.join(output_dir, 'samples')
    wild_card_path = os.path.join(samples, '*', 'score.json')
    number_samples = len(glob.glob(wild_card_path))
    score_dicts = []
    for number in range(number_samples):
        path = os.path.join(samples, str(number), 'score.json')
        score_dict = ut.read_parameters(path)[0]
        score_dicts.append(score_dict)
    scores = [score_dict[fitness_key] for score_dict in score_dicts]
    return scores


def get_sample_nr(path):
    '''Extracts the sample number from a given path

    Parameters:
    ----------
    path : str
        Path to the sample

    Returns : int
        Number of the sample
    '''
    path1 = Path(path)
    parent_path = str(path1.parent)
    sample_nr = int(parent_path.split('/')[-1])
    return sample_nr


def wait_iteration(output_dir, sample_size):
    '''Waits until all batch jobs are finised and in case of and warning
    or error that appears in the error file, stops running the optimization

    Parameters:
    ----------
    output_dir : str
        Path to the directory of output
    sample_size : int
        Number of particles (parameter-sets)

    Returns:
    -------
    Nothing
    '''
    wild_card_path = os.path.join(output_dir, 'samples', '*', 'score.json')
    while len(glob.glob(wild_card_path)) != sample_size:
        check_error(output_dir)
        time.sleep(5)


def move_previous_files(output_dir, previous_files_dir):
    '''Deletes the files from previous iteration

    Parameters:
    -------
    output_dir : str
        Path to the directory of the output

    Returns:
    -------
    Nothing
    '''
    iter_nr = find_iter_number(previous_files_dir)
    samples_dir = os.path.join(output_dir, 'samples')
    iter_dir = os.path.join(previous_files_dir, 'iteration_' + str(iter_nr))
    shutil.copytree(samples_dir, iter_dir)
    shutil.rmtree(samples_dir)
    wild_card_path = os.path.join(output_dir, 'parameter_*.sh')
    for path in glob.glob(wild_card_path):
        os.remove(path)


def find_iter_number(previous_files_dir):
    '''Finds the number iterations done

    Parameters:
    ----------
    previous_files_dir : str
        Path to the directory where old iterations are saved

    Returns:
    -------
    iter_number : int
        Number of the current iteration
    '''
    wild_card_path = os.path.join(previous_files_dir, 'iteration_*')
    iter_number = len(glob.glob(wild_card_path))
    return iter_number


def check_error(output_dir):
    '''In case of warnings or errors during batch job that is written to the
    error file, raises SystemExit(0)

    Parameters:
    ----------
    output_dir : str
        Path to the directory of the output, where the error file is located

    Returns:
    -------
    Nothing
    '''
    number_errors = 0
    error_list = ['FAILED', 'CANCELLED', 'ERROR', 'Error']
    output_error_list = ['Usage']
    error_files = os.path.join(output_dir, 'error*')
    # output_files = os.path.join(output_dir, 'output*')
    for error_file in glob.glob(error_files):
        if os.path.exists(error_file):
            with open(error_file, 'rt') as file:
                lines = file.readlines()
                for line in lines:
                    for error in error_list:
                        if error in line:
                            number_errors += 1
    # for output_file in glob.glob(output_files):
    #     if os.path.exists(output_file):
    #         with open(output_file, 'rt') as file:
    #             lines = file.readlines()
    #             for line in lines:
    #                 for error in output_error_list:
    #                     if error in line:
    #                         number_errors += 1
    if number_errors > 0:
        print("Found errors: " + str(number_errors))
        raise SystemExit(0)


def save_prediction_files(pred_train, pred_test, save_dir):
    '''Saves the prediction files to a .lst file

    Parameters:
    ----------
    pred_train : list of lists
        Predicted labels of the training dataset
    pred_test : list of lists
        Predicted labels of the testing dataset
    save_dir : str
        Directory of output

    Returns:
    -------
    Nothing
    '''
    train_path = os.path.join(save_dir, 'pred_train.lst')
    test_path = os.path.join(save_dir, 'pred_test.lst')
    with open(train_path, 'w') as pred_file:
        for elem in pred_train:
            pred_file.write(str(elem) + '\n')
    with open(test_path, 'w') as file:
        for elem in pred_test:
            pred_file.write(str(elem) + '\n')
