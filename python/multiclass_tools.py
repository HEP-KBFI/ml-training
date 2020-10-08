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

    Returns:
    -------
    false_positive_rate : list
        List of false positives for given thresholds
    true_positive_rate : list
        List of true positives for given thresholds
    '''
    thresholds = np.arange(0, 1, 0.01)
    number_bg = len(pred_vectors[0]) - 1
    true_positive_rate = []
    false_positive_rate = []
    for threshold in thresholds:
        signal = []
        for vector, weight in zip(pred_vectors, weights):
            sig_vector = np.array(vector) >= threshold
            sig_vector = sig_vector.tolist()
            result = []
            for i, element in enumerate(sig_vector):
                if element:
                    result.append(i)
            signal.append(result)
        pairs = list(zip(labels, signal))
        sig_score = 0
        bg_score = 0
        for pair in pairs:
            for i in pair[1]:
                if int(pair[0]) == i:
                    sig_score += weight
                else:
                    bg_score += weight
        true_positive_rate.append(float(sig_score)/len(labels))
        false_positive_rate.append(float(bg_score)/(number_bg*len(labels)))
    return false_positive_rate, true_positive_rate


def multiclass_encoding(data, label_column='process'):
    classes = set(data[label_column])
    mapping = {}
    for i, m_class in enumerate(classes):
        mapping[m_class] = i
    data['multitarget'] = data[label_column].map(mapping)
    return data