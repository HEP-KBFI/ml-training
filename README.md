# Machine Learning and Optimization

This is the general package for the machine learning for HH and ttH analyses. This package includes:

- ML model creation (BDT & ANN)
- Feature selection optimization
- Creation of fitting functions
- Hyperparameter optimization of the ML algorithms

For further info look inthe **main_scripts** directory.

## Installation

In order to install the package, do the following:


````console
git clone https://github.com/HEP-KBFI/ml-training $CMSSW_BASE/src/machineLearning/machineLearning
cd $CMSSW_BASE/src
scram b -j 8
cd $CMSSW_BASE/src/machineLearning/machineLearning
pip install -r requirements.txt --user
````




