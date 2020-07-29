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