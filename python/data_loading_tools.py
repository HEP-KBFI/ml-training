import numpy as np

def tree_to_lorentz(data, name="Jet"):
    return TLorentzVectorArray.from_ptetaphim(
        np.array(data["%s_pt" % name]).astype(np.float64),
        np.array(data["%s_eta" % name]).astype(np.float64),
        np.array(data["%s_phi" % name]).astype(np.float64),
        np.array(data["%s_mass" % name]).astype(np.float64)
    )

def tree_to_array(data, name="Jet"):
    array = np.array([data[var] for var in ["%s_%s" %(name, ll_var) for ll_var in ["e", "px", "py", "pz"]]])
    array = np.moveaxis(array, 0, 1)
    return array

def get_low_level(data, particles):
    ll_variables = []
    for part in particles:
         ll_variables.append(tree_to_array(data, name=part))
    events = np.stack([ll_var for ll_var in ll_variables], axis=1)
    return events


def get_high_level(tree, particles, variables):
    low_level_var = ["%s_%s" %(part, var) for part in particles
                    for var in ["e", "px", "py", "pz"]]
    output = np.array([np.array(tree[variable].astype(np.float32)) for variable in variables if variable not in low_level_var])
    output = np.moveaxis(output, 0, 1)
    return output

def find_correct_dict(key, value, list_of_dicts):
    """Finds the correct dictionary based on the requested key
    Parameters:
    ----------
    key : str
        Name of the key to find
    value: str
        Value the requested key should have
    list_of_dicts : list
        Contains dictionaries to be parsed
    Returns:
    -------
    requested_dict : dict
    """
    new_dictionary = {}
    for dictionary in list_of_dicts:
        if dictionary[key] == value:
            new_dictionary = dictionary.copy()
            new_dictionary.pop(key)
    if new_dictionary == {}:
        print(
            'Given parameter for ' + str(key) + ' missing. Using the defaults')
        for dictionary in list_of_dicts:
            if dictionary[key] == 'default':
                new_dictionary = dictionary.copy()
                new_dictionary.pop(key)
    return new_dictionary
