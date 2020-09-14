from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import data_loading_tools as dlt
import machineLearning.machineLearning as ml
import glob
import os


def test_check_info_files():
    package_path = ml.__path__[0].replace('python', 'src')
    res_wildcard = os.path.join(
        package_path, 'info', 'HH', '*', 'res', '*', 'info.json')
    nonRes_wildcard = os.path.join(
        package_path, 'info', 'HH', '*', 'nonRes', 'info.json')
    faulty_files = []
    for resInfo in glob.glob(res_wildcard):
        try:
            ut.read_json_cfg(resInfo)
        except:
            faulty_files.append(resInfo)
    for nonResInfo in glob.glob(nonRes_wildcard):
        try:
            ut.read_json_cfg(nonResInfo)
        except:
            faulty_files.append(nonResInfo)
    assert len(faulty_files) == 0, "Faulty files: " + str(faulty_files)


def test_check_weights_dir_existance():
    package_path = ml.__path__[0].replace('python', 'src')
    res_wildcard = os.path.join(
        package_path, 'info', 'HH', '*', 'res', '*', 'info.json')
    missing_directories = []
    for path in glob.glob(res_wildcard):
        info_dict = ut.read_json_cfg(path)
        weight_dir = info_dict['weight_dir']
        if not os.path.exists(weight_dir):
            missing_directories.append(path)
    assert len(missing_directories) == 0, "Missing weight_dirs in: " + str(missing_directories)


def test_check_input_path_existance():
    package_path = ml.__path__[0].replace('python', 'src')
    res_wildcard = os.path.join(
        package_path, 'info', 'HH', '*', 'res', '*', 'info.json')
    nonRes_wildcard = os.path.join(
        package_path, 'info', 'HH', '*', 'nonRes', 'info.json')
    missing_inputPaths = []
    for resInfo in glob.glob(res_wildcard):
        info_dict = ut.read_json_cfg(resInfo)
        tauID_wps = info_dict['tauID_training']
        for wp in tauID_wps.keys():
            input_paths = tauID_wps[wp]
            for inPath in input_paths.keys():
                if not os.path.exists(input_paths[inPath]):
                    missing_inputPaths.append(input_paths[inPath])
    for nonResInfo in glob.glob(nonRes_wildcard):
        info_dict = ut.read_json_cfg(nonResInfo)
        tauID_wps = info_dict['tauID_training']
        for wp in tauID_wps.keys():
            input_paths = tauID_wps[wp]
            for inPath in input_paths.keys():
                if not os.path.exists(input_paths[inPath]):
                    missing_inputPaths.append(input_paths[inPath])
    assert len(missing_inputPaths) == 0, "Missing ntuple directories: " + str(set(missing_inputPaths))