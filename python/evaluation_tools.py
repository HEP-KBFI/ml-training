'''Tools to be used for evaluation of the model'''
import numpy as np
import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm


def kfold_cv(
        evaluation,
        prepared_data,
        trainvars,
        global_settings,
        hyperparameters,
        weight='totalWeight',
        n_folds=2
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

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    scores = []
    tests = []
    trains = []
    for train_index, test_index in kfold.split(prepared_data):
        train = prepared_data.iloc[train_index]
        test = prepared_data.iloc[test_index]
        data_dict = {
            'trainvars': trainvars,
            'train': train,
            'test': test
        }
        score, train, test = evaluation(hyperparameters, data_dict, global_settings)
        scores.append(score)
        tests.append(test)
        trains.append(trains)
    avg_score = np.mean(scores)
    stdev_scores = np.std(scores)
    final_score = avg_score - stdev_scores
    return final_score, np.mean(trains), np.mean(tests)


def get_evaluation(
        evaluation,
        prepared_data,
        trainvars,
        global_settings,
        hyperparameters,
        weight='totalWeight'
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
    train, test = train_test_split(
        prepared_data, test_size=0.2, random_state=1)
    data_dict = {
        'trainvars': trainvars,
        'train': train,
        'test': test,
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
    fr_difference = difference / (1 - test_score)
    d_score = test_score - kappa * fr_difference
    return d_score


def calculate_auc(data_dict, prediction, data_class, weights):
    ''' Calculates the ROC curve AUC using sklearn.metrics package.

    Parameters:
    ----------
    data_dict : dict
        Dictionary that contains the DMatrices for train and test (dtrain and dtest)
    prediction : lists of lists
        Predicted labels of train/test data set.
    data_class : str
        Type of the data. ('train' or 'test')
    [weights] : str
        [Default: 'totalWeight'] data label to be used as the weight.
    '''
    labels = np.array(data_dict[data_class]['target']).astype(int)
    weights = np.array(data_dict[data_class]['totalWeight']).astype(float)
    fpr, tpr, thresholds_train = skm.roc_curve(
        labels,
        prediction,
        sample_weight=weights
    )
    auc_score = skm.auc(fpr, tpr, reorder=True)
    return auc_score


def calculate_d_roc(
        data_dict,
        pred_train,
        pred_test,
        weights='totalWeight',
        kappa=1.5
):
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
    test_auc = calculate_auc(data_dict, pred_test, 'test', weights)
    train_auc = calculate_auc(data_dict, pred_train, 'train', weights)
    d_roc = calculate_d_score(train_auc, test_auc, kappa)
    return d_roc


def ams(s, b):
    ''' Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )
    where b_r = 10, b = background, s = signal, log is natural logarithm
    '''
    br = 10.0
    radicand = 2 * ((s + b + br) * np.log(1.0 + s / (b + br)) - s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        return np.sqrt(radicand)


def try_different_thresholds(
        predicted,
        data_dict,
        label_type,
        weights,
        threshold=None
):
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
    weights = data_dict[label_type][weights]
    thresholds = np.arange(0, 1, 0.001)
    ams_scores = []
    signals = []
    backgrounds = []
    prediction = pandas.Series(i[1] for i in predicted)
    if threshold is not None:
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
        weights='totalWeight',
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
        pred_train, data_dict, 'train', weights)
    print(train_ams)
    test_ams = try_different_thresholds(
        pred_test, data_dict, 'test', weights, threshold=best_threshold)
    print(test_ams)
    d_ams = calculate_d_score(train_ams, test_ams, kappa)
    return d_ams


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
        Mean coefficient of variation for a given dictionary having lists as
        values
    '''
    coeff_of_variations = []
    for key in list_dict:
        values = list_dict[key]
        coeff_of_variation = np.std(values)/np.mean(values)
        coeff_of_variations.append(coeff_of_variation)
    mean_coeff_of_variation = np.mean(coeff_of_variations)
    if np.isnan(mean_coeff_of_variation):
        mean_coeff_of_variation = 10 # just a big number, so opt will continue
    return mean_coeff_of_variation


def calculate_improvement(avg_scores, improvements, threshold):
    '''Calculates the improvement based on the average scores. Purpose:
    stopping criteria. Currently used only in GA algorithm.

    Parameters:
    -----------
    avg_scores : list
        Average scores of each iteration in the evolutionary algorithm
    improvements : list
        List of improvements of previous iterations
    threshold : float
        Stopping criteria.

    Returns:
    --------
    improvements : list
        List of improvements
    imporvement : float
        Improvement for comparing

    Comments:
    ---------
    Elif clause used in order to have last 2 iterations less than the threshold
    '''
    if len(avg_scores) > 1:
        improvements.append(
            (float(avg_scores[-1]-avg_scores[-2])) / avg_scores[-2])
        improvement = improvements[-1]
    if len(improvements) < 2:
        improvement = 1
    elif improvement <= threshold:
        improvement = improvements[-2]
    return improvements, improvement
