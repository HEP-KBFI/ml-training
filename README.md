# Machine Learning and Optimization

This is the general package for the machine learning for HH and ttH analyses. This package includes:

- ML model creation (BDT & ANN)
- Feature selection optimization
- Creation of fitting functions
- Hyperparameter optimization of the ML algorithms

For further info look inthe **main_scripts** directory.

## Installation

In order to install the package, do the following:


```console
git clone https://github.com/HEP-KBFI/ml-training $CMSSW_BASE/src/machineLearning/machineLearning
cd $CMSSW_BASE/src
scram b -j 8
cd $CMSSW_BASE/src/machineLearning/machineLearning
pip install -r requirements.txt --user
```


After installing the package, please run the tests:
```console
pytest tests/
```


## Project structure

The package consists out of several folders with a specific purpose that is described below:

---
**evaluation_scripts/***

```
This folder contains the scripts that are run when doing hyperparameter optimization
with slurm. The convention of the naming is the following slurm_[ml_method]_[process].py
Both the values for the options are taken from settings/global_settings.json
```

---
**info/***
```
This folder contains all the necessary info for each channel for loading the correct files
for each scenario and weighing them correctly. Furthermore, the hyperparameters for model
building are located there. Currently there is the structure for ttH and HH analysis both,
though only the latter is up-to-date and active. For further information, read the README
in that folder.
```

---

**main_scripts/***
```
This folder contains the scripts for running various tools for the analyses. The tools
are described in more detail in that folder's own README.
```

---

**python/***
```
The python/* folder contains various tools used by (main_)scripts. For the information
about the purpose of each file, read the README in the python/ folder.
```

---

**settings/***
```
Settings folder contains all the necessary settings for running the scripts. The description
of each setting file is explained in settings/README.md
```

---

**tests/***
```
The test folder contains the tests for checking the code integrity and the existance
of all the necessary information. To run the tests, do:
    $ pytest tests/
```

---
## Lorentz Boost Network

To run this, several other conditions need to be fulfilled:

CMSSW_11_2_0_pre1

````console
pip install uproot_methods --user
pip install eli5 --user
pip install --upgrade pluggy --user
````
