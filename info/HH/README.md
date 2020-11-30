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
    - n_estimators : number of trees (or rounds) for the XGBClassifier. Equivalent
                     to the num_boost_rounds. n_estimators is used by the scikit-learn
                     wrapper.
    - subsample : 
    - colsample_bytree
    - gamma
    - learning_rate
    - max_depth : maximum depth of 
    - min_child_weight
```

---
