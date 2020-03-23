# EvolutionaryAlgorithms
General package for BDT/ANN training including evolutionary algorithms for hyperparameter optimization and training of BDT (XGBoost) and NN for the purposes of Higgs analysis (ttH and HH)


## Installation

CMSSW existance is needed in order to use the package to full capabilities.


````console
git clone https://github.com/HEP-KBFI/ml-training $CMSSW_BASE/src/machineLearning/machineLearning
cd $CMSSW_BASE/src
scram b -j 8
cd $CMSSW_BASE/src/machineLearning/machineLearning
pip install -r requirements.txt --user
````
* attrs==19.1.0 version needed due to 19.2.0 breaking with older version of pytest.
[update pytest to pytest==5.2.0](https://stackoverflow.com/questions/58189683/typeerror-attrib-got-an-unexpected-keyword-argument-convert)

Also in order for feature importances to work with NN, eli5 package is needed:

````console
pip install --user eli5
````


### Tests

After installation please run the unittests (in $CMSSW_BASE/src/machineLearning/machineLearning) with:

````console
pytest tests/
````


