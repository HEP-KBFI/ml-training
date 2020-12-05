import numpy as np


def roc_curve(labels, pred_vectors, weights):
    '''Calculate the ROC values using the method used in Dianas thesis.

    Parameters:
    ----------
    labels : list
        List of true labels
    pred_vectors: list of lists
        List of lists that contain the probabilities for an event to belong to
        a certain label
    weights : list
        List containing the weights of each event.

    Returns:
    -------
    false_positive_rates : list
        List of false positives for given thresholds
    true_positive_rates : list
        List of true positives for given thresholds
    '''
    thresholds = np.linspace(0, 1, num=50)
    number_bkg = len(pred_vectors[0]) - 1
    unique_labels = range(len(pred_vectors[0]))
    true_positive_rates = []
    false_positive_rates = []
    weights = np.array(weights)
    mask_dict = {}
    total_weights = {}
    for label in unique_labels:
        mask_dict[label] = (np.array(labels) == label).astype(int)
        total_weights[label] = sum(weights * mask_dict[label])
    for threshold in thresholds:
        true_positive_rate = 0
        false_positive_rate = 0
        total_bkg_weights = 0
        total_false_positives = 0
        total_true_positives = 0
        signal_vector = (pred_vectors >= threshold).astype(int)
        for label in unique_labels:
            bkg_labels = list(unique_labels)
            bkg_labels.pop(label)
            total_true_positives += sum(
                signal_vector[:, label] * mask_dict[label] * weights)
            for bkg_label in bkg_labels:
                total_bkg_weights += total_weights[bkg_label]
                total_false_positives += sum(
                    signal_vector[:, bkg_label] * weights)
        true_positive_rate = total_true_positives / sum(weights)
        false_positive_rate = total_false_positives / (len(unique_labels) * total_bkg_weights)
        true_positive_rates.append(true_positive_rate)
        false_positive_rates.append(false_positive_rate)
    return false_positive_rates, true_positive_rates


def multiclass_encoding(data, use_Wjet=True, label_column='process'):
    classes = set(data[label_column])
    mapping = {}
    '''for i, m_class in enumerate(classes):
        mapping[m_class] = i
    data['multitarget'] = data[label_column].map(mapping)'''
    if use_Wjet:
        data.loc[data['target']==1, 'multitarget'] = 0
        data.loc[data['process'] == 'TT', 'multitarget'] = 1
        data.loc[data['process'] == 'ST', 'multitarget'] = 2
        data.loc[data['process'] == 'Other', 'multitarget'] = 3
        data.loc[data['process'] == 'W', 'multitarget'] = 4
        data.loc[data['process'] == 'DY', 'multitarget'] = 5
    else:
        data.loc[data['target']==1, 'multitarget'] = 0
        data.loc[data['process'] == 'TT', 'multitarget'] = 1
        data.loc[data['process'] == 'ST', 'multitarget'] = 2
        data.loc[data['process'] == 'Other', 'multitarget'] = 3
        data.loc[data['process'] == 'DY', 'multitarget'] = 4

    return data
