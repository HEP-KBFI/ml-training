'''
Call with 'python'

Usage:
    pkl2xml.py
    pkl2xml.py [--pklFile=PTH --logFile=PTH]

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
    xmlFile = pklFile.replace('.pkl', '.xml')
    try:
        with open(pklFile, 'rb') as pklOpen:
            pklData = pickle.load(pklOpen)
            print('pklData loaded')
    except IOError:
        print('IOError when loading pklData from the file')
    features = pklData.feature_names
    bdtModel = ct.BDTxgboost(pklData, features, ['Background', 'Signal'])
    bdtModel.to_tmva(xmlFile)
    print('BDT model saved to ' + str(xmlFile))


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        pklFile = arguments['--pklFile']
        main(pklFile)
    except docopt.DocoptExit as e:
        print(e)