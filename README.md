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

## Analysis & info folders

For each process and channel, the responsible person should keep the folder up-to-date. For example person responsible for the channel 0l_4tau in HH analysis should create/update the **info/HH/0l_4tau** folder.
When running the analysis, be sure to use the correct channel information in the **global_settings.json**


### Brief description of HH analysis info folder content:

**histo_dict.json**: used for storing the fit function shapes for each variable. Recommended that all trainvars are listed there.

**hyperparameters.json**: the hyperparameters used in creating the BDT model

**info.json**: Used to store the info, which masses, benchmark points etc. to use and where the weights for reweighing are stored.

**keys.json**: Used for storing the information, which sampoles to use

**tauID_application.json**: Contains the datacard numbers

**tauID_training.json**: Contains the locations of the ntuples for different tauID workingpoints

**trainvars.json**: Contains the columns (branches from the ntuple) to be used in the training.




