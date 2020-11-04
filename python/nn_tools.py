'''Tools for creating a neural network model and evaluating it
'''
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
import keras
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Nadam
import numpy as np
import tensorflow as tf
import eli5
from eli5.formatters.as_dataframe import format_as_dataframe
from eli5.sklearn import PermutationImportance
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import evaluation_tools as et
from machineLearning.machineLearning.lbn import LBN, LBNLayer
from machineLearning.machineLearning import multiclass_tools as mt


def model_evaluation_main(nn_hyperparameters, data_dict, global_settings):
    ''' Collected functions for CGB model evaluation

    Parameters:
    ----------
    nn_hyperparamters : dict
        hyperparameters for the model to be created
    data_dict : dict
        Contains all the necessary information for the evaluation.
    global_settings : dict
        Preferences for the optimization

    Returns:
    -------
    score : float
        The score calculated according to the fitness_fn
    '''
    k_model = parameter_evaluation(
        nn_hyperparameters,
        data_dict,
        global_settings['nthread'],
        global_settings['num_classes'],
    )
    score, pred_train, pred_test = evaluate(
        k_model, data_dict, global_settings
    )
    return score, pred_train, pred_test

def Normal(ref, const=None, ignore_zeros=False, name=None, **kwargs):
    """
    Normalizing layer according to ref.
    If given, variables at the indices const will not be normalized.
    """
    if ignore_zeros:
        mean = np.nanmean(np.where(ref == 0, np.ones_like(ref) * np.nan, ref), **kwargs)
        std = np.nanstd(np.where(ref == 0, np.ones_like(ref) * np.nan, ref), **kwargs)
    else:
        mean = ref.mean(**kwargs)
        std = ref.std(**kwargs)
    print mean, '\t', type(mean)
    if const is not None:
        mean[const] = 0
        std[const] = 1
    std = np.where(std == 0, 1, std)
    mul = 1.0 / std
    add = -mean / std
    return tf.keras.layers.Lambda((lambda x: (x * mul) + add), name=name)


def create_nn_model(
        nr_trainvars,
        num_class,
        input_var,
        categorical_var_index,
        lbn=False
):
    ''' Creates the neural network model. The normalization used is
    batch normalization. Kernel is initialized by the Kaiming initializer
    called 'he_uniform'

    Parameters:
    ----------
    nn_hyperparameters : dict
        Dictionary containing the hyperparameters for the neural network. The
        keys contained are ['dropout_rate', 'learning_rate', 'schedule_decay',
        'nr_hidden_layers']
    nr_trainvars : int
        Number of training variables, will define the number of inputs for the
        input layer of the model
    num_class : int
        Default: 3 Number of categories one wants the data to be classified.
    number_samples : int
        Number of samples in the training data
    metrics : ['str']
        What metrics to use for model compilation

    Returns:
    -------
    model : keras.engine.sequential.Sequential
        Sequential keras neural network model created.
    '''
    if lbn:
        ll_inputs = tf.keras.Input(shape=(5, 4), name="LL")
        hl_inputs = tf.keras.Input(shape=(nr_trainvars,), name="HL")
        lbn_layer = LBNLayer(
            ll_inputs.shape, 16,
            boost_mode=LBN.PAIRS,
            features=["E", "pt", "eta", "phi", "m", "pair_cos"]
        )
        lbn_features = lbn_layer(ll_inputs)
        normalized_lbn_features = tf.keras.layers.BatchNormalization()(lbn_features)
        normalized_hl_inputs = Normal(ref=input_var, const=categorical_var_index, axis=1)(hl_inputs)
        x = tf.keras.layers.concatenate([normalized_lbn_features, normalized_hl_inputs])
        for layer in range(0,6) :
            x = tf.keras.layers.Dense(1024, activation="softplus",
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0003))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.0)(x)
        outputs = tf.keras.layers.Dense(num_class, activation='softmax')(x)
        model = tf.keras.Model(
            inputs=[ll_inputs, hl_inputs],
            outputs=outputs,
            name='lbn_dnn'
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0003),
            loss='sparse_categorical_crossentropy',
            metrics=["accuracy"]
        )
    else:
        inputs = tf.keras.Input(shape=(nr_trainvars,), name="input_var")
        normalized_vars = Normal(ref=input_var, const=categorical_var_index, axis=1)(inputs)
        x = tf.keras.layers.Layer()(normalized_vars)
        for layer in range(0,6) :
            x = tf.keras.layers.Dense(1024, activation="softplus",
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0003))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.0)(x)
        outputs = tf.keras.layers.Dense(num_class, activation='softmax')(x)
        model = tf.keras.Model(
            inputs=[inputs],
            outputs=outputs
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0003),
            loss='sparse_categorical_crossentropy',
            metrics=["accuracy"]
        )
    return model


def parameter_evaluation(
        nn_hyperparameters,
        data_dict,
        nthread,
        num_class,
):
    '''Creates the NN model according to the given hyperparameters

    Parameters:
    ----------
    nn_hyperparameters : dict
        hyperparameters for creating the nn model
    data_dict : dict
        Contains all the necessary information for the evaluation.
    nthread : int
        Number of threads to be used for the model
    num_class : int
        Number of classes the samples belong to

    Returns:
    -------
    k_model : KerasClassifier
        The created NN model
    '''
    K.set_session(
        tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=nthread,
                inter_op_parallelism_threads=nthread,
                allow_soft_placement=True,
            )
        )
    )
    nr_trainvars = len(data_dict['train'].values[0])
    number_samples = len(data_dict['train'])
    k_model = KerasClassifier(
        build_fn=create_nn_model,
        epochs=nn_hyperparameters['epochs'],
        batch_size=nn_hyperparameters['batch_size'],
        verbose=2,
        nn_hyperparameters=nn_hyperparameters,
        nr_trainvars=nr_trainvars,
        num_class=num_class,
        number_samples=number_samples
    )
    return k_model


def evaluate(k_model, data_dict, global_settings):
    '''Evaluates the nn k_model

    Parameters:
    ----------
    k_model : KerasClassifier
        NN model.
    data_dict : dict
        Contains all the necessary information for the evaluation.
    global_settings : dict
        Preferences for the optimization

    Returns:
    -------
    score : float
        The score calculated according to the fitness_fn
    '''
    trainvars = data_dict['trainvars']
    fit_result = k_model.fit(
        data_dict['train'][trainvars].values,
        data_dict['train']['target'],
        data_dict['train']['totalWeight'].values,
        validation_data=(
            data_dict['test'][trainvars],
            data_dict['test']['target'],
            data_dict['test']['totalWeight'].values
        )
    )
    pred_train = k_model.predict_proba(data_dict['train'])
    pred_test = k_model.predict_proba(data_dict['test'])
    kappa = global_settings['kappa']
    if global_settings['fitness_fn'] == 'd_roc':
        score = et.calculate_d_roc(pred_train, pred_test, data_dict, kappa)
    elif global_settings['fitness_fn'] == 'd_ams':
        score = et.calculate_d_ams(pred_train, pred_test, data_dict, kappa)
    else:
        print('This fitness_fn is not implemented')
    return score, pred_train, pred_test


def get_feature_importances(model, data_dict, trainvars, data="even"):
    '''Returns the feature importance relevant for neural network case using
    the eli5 package. Note: calculating the feature importances takes a while
    due to it calculating all the permutations.

    Parameters:
    ----------
    model:The nn_model
        The NN model created by create_nn_model
    data_dict : dict
        Contains all the necessary information for the evaluation.

    Returns:
    -------
    feature_importances : dict
        The feature importances equivalent for nn using the eli5 package.
    '''
    perm = PermutationImportance(model, scoring=mt.roc_curve).fit(
        data_dict[data+"_data"][trainvars].values,
        data_dict[data+"_data"]['multitarget'],
        sample_weight=data_dict[data+"_data"]['totalWeight']
    )
    weights = eli5.explain_weights(perm, feature_names=data_dict[trainvars])
    weights_df = format_as_dataframe(weights).sort_values(
        by='weight', ascending=False).rename(columns={'weight': 'score'})
    list_of_dicts = weights_df.to_dict('records')
    feature_importances = {}
    for single_variable_dict in list_of_dicts:
        key = single_variable_dict['feature']
        feature_importances[key] = single_variable_dict['score']
    return feature_importances


def calculate_number_nodes_in_hidden_layer(
        number_classes,
        number_trainvars,
        number_samples,
        alpha
):
    '''Calculates the number of nodes in a hidden layer

    Parameters:
    ----------
    number_classes : int
        Number of classes the data is to be classified to.
    number_trainvars : int
        Number of training variables aka. input nodes for the NN
    number_samples : int
        Number of samples in the data
    alpha : float
        number of non-zero weights for each neuron

    Returns:
    -------
    number_nodes : int
        Number of nodes in each hidden layer

    Comments:
    --------
    Formula used: N_h = N_s / (alpha * (N_i + N_o))
    N_h: number nodes in hidden layer
    N_s: number samples in train data set
    N_i: number of input neurons (trainvars)
    N_o: number of output neurons
    alpha: usually 2, but some reccomend it in the range [5, 10]
    '''
    number_nodes = number_samples / (
        alpha * (number_trainvars + number_classes)
    )
    return number_nodes


def create_hidden_net_structure(
        number_hidden_layers,
        number_classes,
        number_trainvars,
        number_samples,
        alpha=2
):
    '''Creates the hidden net structure for the NN

    Parameters:
    ----------
    number_hidden_layers : int
        Number of hidden layers in our NN
    number_classes : int
        Number of classes the data is to be classified to.
    number_trainvars : int
        Number of training variables aka. input nodes for the NN
    number_samples : int
        Number of samples in the data
    [alpha] : float
        [Default: 2] number of non-zero weights for each neuron

    Returns:
    -------
    hidden_net : list
        List of hidden layers with the number of nodes in each.
    '''
    number_nodes = calculate_number_nodes_in_hidden_layer(
        number_classes,
        number_trainvars,
        number_samples,
        alpha
    )
    number_nodes = int(np.floor(number_nodes/number_hidden_layers))
    hidden_net = [number_nodes] * number_hidden_layers
    return hidden_net


def custom_permutation_importance(
        model, data, weights,
        trainvars, labels, permutations=5
):
    print('Starting permutation importance')
    score_dict = {}
    prediction = model.predict(data)
    original_score = calculate_acc_with_weights(prediction, labels, weights)
    print('Reference score: ' + str(original_score))
    for trainvar in trainvars:
        print(trainvar)
        data_copy = data.copy()
        t_score = 0
        for i in range(permutations):
            print("Permutation nr: " + str(i))
            data_copy[trainvar] = np.random.permutation(data_copy[trainvar])
            prediction = model.predict(data_copy)
            score = calculate_acc_with_weights(prediction, labels, weights)
            print(score)
            t_score += score
        score_dict[trainvar] = abs(original_score - (t_score/permutations))
    sorted_sd = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
    print(sorted_sd)
    return score_dict


def calculate_acc_with_weights(prediction, labels, weights):
    num_classes = len(prediction[0])
    pred_labels = [np.argmax(event) for event in prediction]
    true_positives = 0
    true_negatives = 0
    total_positive = sum(weights)
    total_negative = sum(weights)
    for pred, true, weight in zip(pred_labels, labels, weights):
        if pred == true:
            true_positives += weight
            true_negatives += num_classes * weight
        else:
            true_negatives += (num_classes - 1) * weight
    true_negatives /= num_classes
    accuracy = (true_positives + true_negatives) / (total_positive + total_negative)
    return accuracy


def lbn_feature_importances(
        model, data_dict, trainvars,
        permutations=5, case='even'
):
    labels = data_dict[case + '_data']['multitarget']
    weights = data_dict[case + '_data']['evtWeight']
    ll_names = ['b1jets', 'b2jets', 'w1jets', 'w2jets', 'leptons']
    high_level = data_dict['hl_' + case]
    low_level = data_dict['ll_' + case]
    reference_prediction = model.predict(
        [low_level, high_level], batch_size=1024)
    reference_score = calculate_acc_with_weights(
        reference_prediction, labels, weights)
    print(reference_prediction)
    score_dict = {}
    for i, feature in enumerate(ll_names):
        print(feature)
        t_score = 0
        for permutation in range(permutations):
            print('Permutation: ' + str(i))
            shuffled_low_level = shuffle_low_level(low_level, i)
            prediction = model.predict(
                [shuffled_low_level, high_level], batch_size=1024)
            score = calculate_acc_with_weights(prediction, labels, weights)
            t_score += score
        score_dict[feature] = abs(reference_score - (t_score/permutations))
    return score_dict


def shuffle_low_level(low_level, i):
    low_level_ = low_level.copy()
    shuffled_elements = np.random.permutation(low_level_[:, i])
    low_level_[:, i] = shuffled_elements
    return low_level_
