from machineLearning.machineLearning import universal_tools as ut
import machineLearning.machineLearning as ml
import glob
import os


def check_info_files():
    package_path = ml.__path__[0]
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
    assert(len(faulty_files) == 0, "Faulty files: " + str(faulty_files))