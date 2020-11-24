'''
Call with 'python'

Usage:
    pkl2xml.py
    pkl2xml.py [--pklFile=PTH]

Options:
    -i --pklFile=PTH         .pkl file to be converted [default: None]
'''
import docopt
from machineLearning.machineLearning import converter_tools as ct
try:
    import cPickle as pickle
except:
    import pickle


def main(pklFile):
    xml_file = pklFile.replace('.pkl', '.xml')
    try:
        with open(pklFile, 'rb') as pklOpen:
            pkl_data = pickle.load(pklOpen)
            print('pklData loaded')
    except IOError:
        print('IOError when loading pklData from the file')
    bst = pkl_data.get_booster()
    features = bst.feature_names
    bdt_model = ct.BDTxgboost(pkl_data, features, ['Background', 'Signal'])
    bdt_model.to_tmva(xml_file)
    print('.xml BDT model saved to ' + str(xml_file))


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        pklFile = arguments['--pklFile']
        main(pklFile)
    except docopt.DocoptExit as e:
        print(e)