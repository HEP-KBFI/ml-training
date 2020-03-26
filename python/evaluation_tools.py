'''Tools to be used for evaluation of the model'''
import numpy as np
import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def kfold_cv(
        evaluation,
        prepared_data,
        trainvars,
        global_settings,
        hyperparameters,
):
    ''' Splits the dataset into 5 parts that are to be used in all combinations
    as training and testing sets

    Parameters:
    ----------
    evaluation : method
        What evaluation to use
    prepared_data : pandas DataFrame
        The loaded data to be used
    trainvars : list
        List of training variables to be used.
    global_settings : dict
        Preferences for the optimization
    hyperparamters : dict
        hyperparameters for the model to be created
    evaluations : method
        Method of evaluation

    Returns:
    -------
    final_score : float
        Calculated as the average minus stdev of all the cv scores
    '''

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []
    for train_index, test_index in kfold.split(prepared_data):
        train = prepared_data.iloc[train_index]
        test = prepared_data.iloc[test_index]
        data_dict = {
            'trainvars': trainvars,
            'train': train,
            'test': test,
            'traindataset': np.array(train[trainvars].values),
            'testdataset': np.array(test[trainvars].values),
            'training_labels': train['target'].astype(int),
            'testing_labels': test['target'].astype(int)
        }
        score = evaluation(hyperparameters, data_dict, global_settings)[0]
        scores.append(score)
    avg_score = np.mean(scores)
    stdev_scores = np.std(scores)
    final_score = avg_score - stdev_scores
    return final_score

def get_evaluation(
        evaluation,
        prepared_data,
        trainvars,
        global_settings,
        hyperparameters,
):
    ''' Splits the data to test (20%) and train (80%) respectively

    Parameters:
    ----------
    evaluation : method
        What evaluation to use
    prepared_data : pandas DataFrame
        The loaded data to be used
    trainvars : list
        List of training variables to be used.
    global_settings : dict
        Preferences for the optimization
    hyperparamters : dict
        hyperparameters for the model to be created
    evaluations : method
        Method of evaluation

    Returns:
    -------
    score : float
        The score corresponding to the fitness_fn
    pred_train : list of lists
        Predicted labels of the training dataset
    pred_test : list of lists
        Predicted labels of the testing dataset
    '''
    if bool(global_settings['split_Odd_Even']):
        train = prepared_data.loc[(prepared_data["event"].values % 2 == 0)] 
        test  = prepared_data.loc[~(prepared_data["event"].values % 2 == 0)]
    else:    
        train, test = train_test_split(
            prepared_data, test_size=0.2, random_state=1)
    data_dict = {
        'trainvars': trainvars,
        'train': train,
        'test': test,
        'traindataset': np.array(train[trainvars].values),
        'testdataset': np.array(test[trainvars].values),
        'training_labels': train['target'].astype(int),
        'testing_labels': test['target'].astype(int)
    }
    score, pred_train, pred_test = evaluation(
        hyperparameters, data_dict, global_settings)
    return score, pred_train, pred_test


def calculate_d_score(train_score, test_score, kappa=1.5):
    ''' Calculates the d_score with the given kappa, train_score and
    test_score. Can be used to get d_auc, d_ams or other similar.

    Parameters:
    ----------
    train_score : float
        Score of the training sample
    test_score : float
        Score of the testing sample
    kappa : float
        Weighing factor for the difference between test and train auc

    Returns:
    -------
    d_roc : float
        Score based on D-score and AUC
    '''
    difference = max(0, train_score - test_score)
    weighed_difference = kappa * (1 - difference)
    denominator = kappa + 1
    d_score = (test_score + weighed_difference) / denominator
    return d_score


########################################################



def roc(labels, pred_vectors):
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
        for vector in pred_vectors:
            sig_vector = np.array(vector) > threshold
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
                if pair[0] == i:
                    sig_score += 1
                else:
                    bg_score += 1
        true_positive_rate.append(float(sig_score)/len(labels))
        false_positive_rate.append(float(bg_score)/(number_bg*len(labels)))
    return false_positive_rate, true_positive_rate


def calculate_auc(data_dict, pred_train, pred_test):
    '''Calculates the area under curve for training and testing dataset using
    the predicted labels

    Parameters:
    ----------
    data_dict : dict
        Dictionary that contains the labels for testing and training. Keys are
        called 'testing_labels' and 'training_labels'
    pred_train : list of lists
        Predicted labels of the training dataset
    pred_test : list of lists
        Predicted labels of the testing dataset

    Returns:
    -------
    train_auc : float
        Area under curve for training dataset
    test_aud : float
        Area under curve for testing dataset
    '''
    x_train, y_train = roc(
        np.array(data_dict['training_labels']), pred_train)
    x_test, y_test = roc(
        np.array(data_dict['testing_labels']), pred_test)
    test_auc = (-1) * np.trapz(y_test, x_test)
    train_auc = (-1) * np.trapz(y_train, x_train)
    info = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test
    }
    return train_auc, test_auc, info


def calculate_d_roc(data_dict, pred_train, pred_test, kappa=1.5):
    '''Calculates the d_roc score

    Parameters:
    ----------
    pred_train : list of lists
        Predicted labels of the training dataset
    pred_test : list of lists
        Predicted labels of the testing dataset
    data_dict : dict
        Dictionary that contains the labels for testing and training. Keys are
        called 'testing_labels' and 'training_labels'

    Returns:
    -------
    d_roc : float
        the AUC calculated using the d_score function
    '''
    train_auc, test_auc = calculate_auc(data_dict, pred_train, pred_test)[:2]
    d_roc = calculate_d_score(train_auc, test_auc, kappa)
    return d_roc


def ams(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm
    """
    br = 10.0
    radicand = 2 *( (s+b+br) * np.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print 'radicand is negative. Exiting'
        exit()
    else:
        return np.sqrt(radicand)


def try_different_thresholds(predicted, data_dict, label_type, threshold=None):
    '''Tries different thresholds if no threshold given to find out the maximum
    AMS score.

    Parameters:
    ----------
    predicted : list
        List of lists, containing the predictions (either pred_train
        or pred_test)
    data_dict : dict
        Contains all the necessary information for the evaluation.
    label_type: str
        Type of the label (either 'test' or 'train')
    [threshold : float]
        [Default: None] If None, then best threshold will be found. Otherwise
        use the given threshold

    Returns:
    -------
    ams_score [or best_ams_score] : float
        Depending on input variable 'threshold', will return either the best
        found ams_score or the ams_score corresponding to the given threshold
    [best_threshold] : float
        If no threshold is give, it will return what was the threshold used
        for finding the biggest ams_score
    '''
    label_key = label_type + 'ing_labels'
    weights = data_dict[label_type]['evtWeight']
    thresholds = np.arange(0, 1, 0.001)
    ams_scores = []
    signals = []
    backgrounds = []
    prediction = pandas.Series(i[1] for i in predicted)
    if threshold != None:
        th_prediction = pandas.Series(
            [1 if pred >= threshold else 0 for pred in prediction])
        signal, background = calculate_s_and_b(
            th_prediction, data_dict[label_key], weights, weighed=False)
        ams_score = ams(signal, background)
        return ams_score
    else:
        for test_threshold in thresholds:
            th_prediction = pandas.Series(
                [1 if pred >= test_threshold else 0 for pred in prediction])
            signal, background = calculate_s_and_b(
                th_prediction, data_dict[label_key], weights, weighed=False)
            ams_score = ams(signal, background)
            ams_scores.append(ams_score)
        index = np.argmax(ams_scores)
        best_ams_score = ams_scores[index]
        best_threshold = thresholds[index]
        return best_ams_score, best_threshold


def calculate_s_and_b(prediction, labels, weights, weighed=True):
    '''Calculates amount of signal and background. When given weights, possible
    to have weighed signal and background

    Parameters:
    ----------
    prediction : list
        Prediction for each event. (list of int)
    labels : list / pandas Series
        True label for each event
    weights : list
        list of floats. Weight for each event
    [weighed=True] : bool
        Whether to use the weights for calculating singal and background

    Returns:
    -------
    signal : int (float)
        Number of (weighed) signal events in the ones classified as signal
    background : int
        Number of (weighed) background events in the ones classified as signal
    '''
    signal = 0
    background = 0
    prediction = np.array(prediction)
    labels = np.array(labels)
    weights = np.array(weights)
    for i in range(len(prediction)):
        if prediction[i] == 1:
            if labels[i] == 1:
                if weighed:
                    signal += weights[i]
                else:
                    signal += 1
            elif labels[i] == 0:
                if weighed:
                    background += weights[i]
                else:
                    background += 1
    return signal, background


def calculate_d_ams(
        pred_train,
        pred_test,
        data_dict,
        kappa=1.5
):
    '''Calculates the d_ams score

    Parameters:
    ----------
    pred_train : list of lists
        Predicted labels of the training dataset
    pred_test : list of lists
        Predicted labels of the testing dataset
    data_dict : dict
        Dictionary that contains the labels for testing and training. Keys are
        called 'testing_labels' and 'training_labels'

    Returns:
    -------
    d_ams : float
        the ams score calculated using the d_score function
    '''
    train_ams, best_threshold = try_different_thresholds(
        pred_train, data_dict, 'train')
    test_ams = try_different_thresholds(
        pred_test, data_dict, 'test', threshold=best_threshold)
    d_ams = calculate_d_score(train_ams, test_ams, kappa)
    return d_ams


###############################################################3



def calculate_compactness(parameter_dicts):
    '''Calculates the improvement based on how similar are different sets of
    parameters

    Parameters:
    ----------
    parameter_dicts : list of dicts
        List of dictionaries to be compared for compactness.

    Returns:
    -------
    mean_cov : float
        Coefficient of variation of different sets of parameters.
    '''
    keys = parameter_dicts[0].keys()
    list_dict = values_to_list_dict(keys, parameter_dicts)
    mean_cov = calculate_dict_mean_coeff_of_variation(list_dict)
    return mean_cov



def values_to_list_dict(keys, parameter_dicts):
    '''Adds same key values from different dictionaries into a list w

    Parameters:
    ----------
    keys : list
        list of keys for which same key values are added to a list
    parameter_dicts : list of dicts
        List of parameter dictionaries.

    Returns:
    -------
    list_dict: dict
        Dictionary containing lists as valus.
    '''
    list_dict = {}
    for key in keys:
        key = str(key)
        list_dict[key] = []
        for parameter_dict in parameter_dicts:
            list_dict[key].append(parameter_dict[key])
    return list_dict


def calculate_dict_mean_coeff_of_variation(list_dict):
    '''Calculate the mean coefficient of variation for a given dict filled
    with lists as values

    Parameters:
    ----------
    list_dict : dict
        Dictionary containing lists as values

    Returns:
    -------
    mean_coeff_of_variation : float
        Mean coefficient of variation for a given dictionary haveing lists as
        values
    '''
    coeff_of_variations = []
    for key in list_dict:
        values = list_dict[key]
        coeff_of_variation = np.std(values)/np.mean(values)
        coeff_of_variations.append(coeff_of_variation)
    mean_coeff_of_variation = np.mean(coeff_of_variations)
    return mean_coeff_of_variation
