## Settings folder

### Files
___
**global_settings.json**

```Used for specifying the configuration for a script.```

| Key | Description |
| :-------------- | :-------------- |
| output_dir | The directory where the output is written. Needs always to be specified. Usually environment variables such as $HOME can be used.|
| ml_method | The machine learning method to be used. Options: [xgb, nn, lbn] |
| feature_importance | Whether to calculate feature importances. Currently used only by main_scripts/hh_nnTraining.py, since for neural networks calculating the feature importances is done by permutation importance, thus the calculations take a long time. Options: [0, 1]|
| process | The process for which the data is loaded. Options [HH, ttH] |
| bdtType | The type of normalization done for the data. Options: [evtLevelSUM_HH] |
| bkg_mass_rand | The type of background mass randomization to be used. Options: [oversampling, default] |
| channel | The channel for which the settings are used. For possible options look in the info/[process]/ folder |
| fitness_fn | The fitness function to be used in the hyperparameter optimization. Options: [d_roc, d_ams]. |
| use_kfold | Whether or not to used KFold cross-validation in hyperparameter optimization. Options: [0, 1] |
| nthread | Number of threads to be used in the model building. |
| kappa | The overtraining penalty term used in hyperparameter optimization. |
| scenario | The scenario for which the data is loaded. For HH the options are [spin0, spin2, nonres/default, nonres/base]|
| dataCuts | Whether to use some additional cuts in the pandas.DataFrame. If no cuts are to be applied, the value should be set to 0, if the info/[process]/[channel]/[scenario]/cuts.json is to be used, the value should be set to 1. For a special cut-file, the value can also be set to a path. Option: [0, 1, PATH] |
| debug | Whether to output the model prediction for each event into a .json file. RLE values are used for each event. |


___
**nn_parameters.json**
```
[Currently not used] Used for specifying the ranges for each hyperparameter that can be tuned
in hyperparameter optimization.
```

___
**pso_settings.json**

```The settings used by the Particle Swarm Optimization. See also https://arxiv.org/abs/2011.04434 ```

| Key | Description |
| :--- | :--- |
| iterations | Number of iterations to be done in the optimization |
| sample_size | Number of particles in the swarm looking for the global optimum |
| compactness_threshold | The threshold when the optimization is stopped, since the particles have converged to very similar parameter values |
| nr_informants | Number of informants that report their best known position per iteration each particle has. | 

___
**submit_template.sh**
```
Used as a stub for the creation a sbatch job.
```

___
**test_cuts.json**
```
Template of a cut file.
```

___
**xgb_parameters.json**
```
The ranges of the XGBoost hyperparameters used in hyperparameter optimization. Can be used to
reduce or increase the hyperparameter space to be scanned by the particles.
```

