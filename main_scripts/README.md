## Creating gen_mHH profiles

In order to create the gen_mHH profiles, one needs to run **gen_mHH_profiling.py**.
The script has 3 modes available, that can be used both stand-alone and also simultaneously.

The modes together with the respective flags are the following:
* **create_info -i** : Creates histo_dict.json to your channel info directory using the training variables specified in the channel specific trainvars.json file. WARNING: Will overwrite the previous histo_dict.json in the info_dir. If histo_dict.json already present in the channel info directory, it will use the info when creating the new histo_dict.json.
* **create_profile -p** : Creates the TProfiles of each variable vs gen_mHH. This is without fitting. After having the fits, one can create the TProfiles before and after reweighing. Other arguments needed: --weight_dir and --masses_type
* **fit -f** : Creates the TProfiles of each variable vs gen_mHH and does a fit. The fit function is chosen according to the specified function name in the histo_dict.json. Other arguments needed: --weight_dir and --masses_type


````console
Options:
    -f --fit=BOOL                     Fit the TProfile. WARNING: Will overwrite the previous histo_dict.json in the info_dir [default: 0]
    -i --create_info=BOOL             Create new histo_dict.json [default: 0]
    -p --create_profile=BOOL          Creates the TProfile without the fit. [default: 0]
    -w --weight_dir=DIR               Directory where the weights will be saved [default: $HOME/gen_mHH_weight_dir]
    -m --masses_type=STR              'low', 'high' or 'all' [default: all]
````

For example, in order to only create the **histo_dict.json**, one needs to call as following:

````console
python gen_mHH_profiling.py -i 1
````

or for both creating the histo_dict.json and fitting in one go:

````console
python gen_mHH_profiling.py -i 1 -f 1
````


## Optimizing the training variables

Having too many features (or trainig variables) will cause the model to be overfitted, so it generalizes bad on the unseen data.
To overcome this overtraining problem, one needs to reduce the number of training variables.
For this purpose, one can use the cript **trainvar_optimization.py**.

This script starts with removing the highly correlated trainvars (one from the pair). The cut is made according to the parameter **corr_threshold**, which defaults to 0.8.
Then a model is built with the remaining trainvars and the importance for each feature is evaluated according to the model.get_fscore (this metric simply sums up how many times each feature is split on; similar to the frequency metric).
After having the importance of each feature, the worst performing **step_size** variables are removed from the list of trainvars.
The new model will be trained with the remaining trainvars. This process repeats until **min_nr_trainvars** is reached.

As a result, **trainvars.json** file will be saved to your channel info directory. All trainings use the default hyperparameters (**info/default_hyperparameters.json**).

**Precondition:** The starting trainvars should be listed in a file called **all_trainvars.json** in your channel info directory.

The options for the scripts are the following:

````console
    -c --corr_threshold=FLOAT       Threshold from which trainvar is dropped [default: 0.8]
    -n --min_nr_trainvars=INT       Number trainvars to end up with [default: 10]
    -s --step_size=INT              Number of trainvars dropped per iteration [default: 5]
````

An example of how to run the code:

````console
python gen_mHH_profiling.py -c 0.9 -n 15 -s 3
````

Or simply by using the defaults:

````console
python gen_mHH_profiling.py
````


## Optimizing the training variables

Choosing the optimal set of hyperparameters for the machine learning model is vital in order to gain performance and to reduce overtraining. Often this task is done either manually or using some not optimal method (e.g grid search or random guessing).
In this package this optimization is done by a algorithm called 'Particle Swarm Optimization' (PSO).

To start the optimization, no extra commandline arguments are needed. The settings for the are located in **settings/pso_settings.json**. The recommended defaults are already pre-set:

````console
    iterations:                     Maximum number of evolutions of the swarm [default: 50]
    sample_size:                    Number of particles in the swarm [default: 70]
    compactness_threshold:          A measure of the relative similarity of the different particles. Used as a stopping criteria: Also known as the 'mean coefficient of variation'. [default: 0.1]
    nr_informants:                  Number of particles informing each particle each iteration about their personally found best location. [default: 10]
````

To run the hyperparameter optimization one also needs to set the correct settings in the **settings/global_settings.json**

````console
    output_dir:         The directory where the information of the optimization will be outputted. This is also used by gen_mHH_profiling.py, hh_bdtTraining.py and trainvar_optimization.py [default: '$HOME/foobar']
    ml_method:          The ML method to be used for the training. Other possibility was 'nn', which is currently deprecated. [default: 'xgb']
    process:            The process for which the data will be loaded [default: 'HH']
    bdtType:            Type of the boosted decision tree. Usually either resonant or nonresonant. [default: 'evtLevelSUM_HH_2l_2tau_res']
    tauID_training:     The name of the working point to be used. [default: 'deepVSjVVVLoose']
    bkg_mass_rand:      The background randomization method. [default: 'oversampling']
    channel:            The name of the channel for which the data is loaded. [default: '2l_2tau']
    fitness_fn:         Name of the fitness function to be used in the optimization. [default: 'd_roc'] Other possibility is 'd_ams'.
    use_kfold:          Whether to use k-fold cross validation when evaluationg each set of hyperparameters. [default: 1]
    nthread:            Number of threads to be used in the optimization. Please be considerate and don't use more than necessary. [default: 8]
    kappa:              The weight of the poenalty term to avoid overtraining. [default: 2.0]
    spinCase:           Applies only for the HH resonant case. [default: 'spin0'] Other possibility is 'spin2'
````

Usually one only needs to change the 'output_dir', 'bdtType' and 'process' and 'channel'.

To perform the hyperparameter optimization, simply run the following:

````console
python hyperparameterOptimization.py
````
