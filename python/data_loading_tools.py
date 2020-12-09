import numpy as np
import uproot_methods

TLorentzVectorArray = uproot_methods.classes.TLorentzVector.TLorentzVectorArray


def tree_to_lorentz(data, name="Jet"):
    return TLorentzVectorArray.from_ptetaphim(
        np.array(data["%s_pt" % name]).astype(np.float64),
        np.array(data["%s_eta" % name]).astype(np.float64),
        np.array(data["%s_phi" % name]).astype(np.float64),
        np.array(data["%s_mass" % name]).astype(np.float64)
    )


def tree_to_array(data, name="Jet"):
    lorentz = tree_to_lorentz(data, name=name)
    array = np.array([
        lorentz.E[:],
        lorentz.x[:],
        lorentz.y[:],
        lorentz.z[:],
    ])
    array = np.moveaxis(array, 0, 1)
    return array


def get_low_level(data):
    b1jets = tree_to_array(data, name="bjet1")
    b2jets = tree_to_array(data, name="bjet2")
    w1jets = tree_to_array(data, name="wjet1")
    w2jets = tree_to_array(data, name="wjet2")
    leptons = tree_to_array(data, name="lep")
    events = np.stack([b1jets, b2jets, w1jets, w2jets, leptons], axis=1)
    return events


def get_high_level(tree, variables):
    output = np.array([np.array(tree[variable].astype(np.float32)) for variable in variables])
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
