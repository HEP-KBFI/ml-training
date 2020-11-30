# HH info & it's folder structure


## Analysis types

The HH info folder contains the necesasry files for machine learning purposes
both for multilepton as well as for the bbWW analysis. The channels implemented
are listed in the following:

Multilepton analysis channels:

    - 0l_4tau
    - 1l_3tau
    - 2l_2tau
    - 2lss
    - 3l_0tau
    - 3l_1tau
    - 4l


bbWW analysis channels:

    - bb1l
    - bb2l

## Structure of the HH info directory

For the HH, the general structure is the following, though it can contain
also other miscellaneous files for a given channel:

```
[HH]
  |
  |__[channel]
  |       |
  |       |__[nonres]
  |       |     |
  |       |     |__[default]
  |       |     |    |__[hyperparameters.json]
  |       |     |    |__[info.json]
  |       |     |    |__[trainvars.json]
  |       |     |
  |       |     |__[base]
  |       |         |__[hyperparameters.json]
  |       |         |__[info.json]
  |       |         |__[trainvars.json]
  |       |
  |       |__[res]
  |       |     |
  |       |     |__[spin0]
  |       |     |    |__[histo_dict.json]
  |       |     |    |__[hyperparameters.json]
  |       |     |    |__[info.json]
  |       |     |    |__[trainvars.json]
  |       |     |
  |       |     |__[spin2]
  |       |          |__[histo_dict.json]
  |       |          |__[hyperparameters.json]
  |       |          |__[info.json]
  |       |          |__[trainvars.json]
  |       |
  |       |__[all_trainvars.json]
  |
  *  ________________
  * |                |
  * | OTHER CHANNELS |
  * |                |
  *  ________________
  |
  |__[background_categories.json]
```

### Explanation of files and the contents
----
hyperparameters.json:

```
Contains the following keys by default:
    - n_estimators : number of trees (or rounds) for the XGBClassifier. Equivalent to the num_boost_rounds.
                     n_estimators is used by the scikit-learn wrapper.

    - subsample : limits the number of training events that are used to grow each tree to a fraction of the
                  full training sample.

    - colsample_bytree : specifies the number of different features that are used in a tree

    - gamma : represents a regularization parameter, which aims to reduce overfitting. Large values of this
              parameter prevent the splitting of leaf nodes before the maximum depth of a tree is reached.

    - learning_rate : controls the effect that trees added at a later stage of the boosted iterations have
                      on the output of the BDT relative to the effect of trees added at an earlier stage.
                      Small values of the learningrate parameter decrease the effect of trees added
                      during the boosting iterations, thereby reducing the effect of boosting on the BDT output.

    - max_depth : specifies the maximum depth of a tree

    - min_child_weight : specifies the minimum number of events that is required in each leaf node. 
```
---

all_trainvars.json:
```
This file contains all the training variables / features that can be used by the model. This list should contain
only the variables present in the .root files & shouldn't contain values such as weights or generator level
information, raw_*, etc.
```
---

histo_dict.json
```
This file contains the info about what order polynomial to use for creating TProfiles for each variable.
Currently the only relevalt keys are:
    nBins : number of bins for the fit
    fitFunc_AllMassTraining : Value of this key is the polynomial. E.g 'pol6'
    Variable : name of the variable for which the fit function belongs to.

P.S Used only for the resonant cases (spin0 and spin2).
```
---

trainvars.json
```
This file contains only those training variables to be used by the model. For each scenario this
list can be different. This list can contain also variables that are not present in .root files - this
is the case for parametrized training. For 'nonresonant/default' additionally the benchmark points scpecified
in info.json file are added. For resonant cases 'gen_mHH' value is present. All the additional
variables are added automatically after trainvar optimization.
```
---

info.json
```
This file is used for data loading and specifying how and which data to load for a given scenario.
Meaning of the keys:

    class : Specifies the the class to which this info file belongs. Not used, only an indicator

    masses : Needs to be specified only in resonant cases. Indicates which mass points for which
             background is duplicated and associated. Afterwards normalization divides the weights
             by the number of mass points.

    masses_test : [Currently not used] Specified only in resonant cases. Otherwise used only
                  for testing purposes

    masses_low : [Currently not used] Specified only in resonant cases.

    masses_test_low : [Currently not used] Specified only in resonant cases. Otherwise used only
                      for testing purposes

    masses_high : [Currently not used] Specified only in resonant cases.

    masses_test_high : [Currently not used] Specified only in resonant cases. Otherwise used for only
                       testing purposes

    nonResScenarios : Needs to be specified only in nonresonant cases. List of BM points to be used.

    nonResScenarios_test : [Currently not used] Specified only in nonresonant cases. Otherwise used for only
                           testing purposes

    channelInTree : Specifies the main directory in the .root files

    default_tauID_application : The tauID used for 'default' mode. Used for choosing the normalization
                                from under the 'tauID_application' key

    weight_dir : Needs to be specified only in resonant cases. Specifies the location of the directory
                 where the fit functions are stored that were created by the 'gen_mHH_profiling.py'
                 script.

    included_eras : Specifies the eras for which the data is loaded.

    tauID_training_key : Specifies the tauID used in 'forBDTtraining'. Used to choose paths from
                         under the 'tauID_training' key.

    tauID_application : The normalizations for different tauID WP for the 'default' mode.

    tauID_training : Contains the paths for 'forBDTtraining' tauID WP.

    data_csv : Specifies the location of the .csv file of the loaded data. If given path doesn't
               exist, data needs to be loaded from the .root files. Having the .csv file speeds
               up the data loading significantly. In the .csv file all data manipulations should
               be already done (normalization, reweighing, etc).

    keys : Which keys to use for data loading. If key exists in background_categories.json file,
           only those samples are loaded. If the key does not exist there, a wildcard with the
           given key is used. Each entry has the format:
                {
                   "key_to_be_used": [list of eras for which is applicable]
                }
```
---

background_categories.json
```
This file contains the background categories and the process name of each
sample. This file is when choosing which files to load and set the process name.
File basic layout:
{
    [key1]: {
        [sample1] : [process],
        [sample2] : [process]
    },
    [key2]: {
        [sample1] : [process],
        [sample2] : [process]
    }
}
```

