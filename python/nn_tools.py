'''Tools for creating a neural network model and evaluating it
'''
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.activations import elu
from keras.layers import ELU
from keras.optimizers import Nadam
import numpy as np
from keras import backend as K
import tensorflow as tf
import keras
from keras.wrappers.scikit_learn import KerasClassifier
import json
from machineLearning.machineLearning import universal_tools as ut
import eli5
from eli5.formatters.as_dataframe import format_as_dataframe
from eli5.sklearn import PermutationImportance
from machineLearning.machineLearning.lbn import LBN, LBNLayer
from sklearn.utils.multiclass import type_of_target
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


def create_nn_model(
        nn_hyperparameters,
        nr_trainvars,
        num_class,
        number_samples,
        metrics=['accuracy'],
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
    '''model = keras.models.Sequential()
    model.add(
        Dense(
            2*nr_trainvars,
            input_dim=nr_trainvars,
            kernel_initializer='he_uniform',
            activation = "relu"
        )
    )
    model.add(BatchNormalization())
    #model.add(ELU())
    model.add(Dropout(nn_hyperparameters['visible_layer_dropout_rate']))
    hidden_layers = create_hidden_net_structure(
        nn_hyperparameters['nr_hidden_layers'],
        num_class,
        nr_trainvars,
        number_samples,
        nn_hyperparameters['alpha']
    )
    for hidden_layer in hidden_layers:
        model.add(Dense(hidden_layer, kernel_initializer='he_uniform', activation="relu"))
        model.add(BatchNormalization())
        #model.add(ELU())
        model.add(Dropout(nn_hyperparameters['hidden_layer_dropout_rate']))
    model.add(Dense(num_class, activation='softmax'))
    model.compile(
        loss='sparse_categorical_crossentropy',
        #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        #optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        optimizer=Nadam(
            lr=nn_hyperparameters['learning_rate'],
            schedule_decay=nn_hyperparameters['schedule_decay']
        ),
        metrics=metrics,
    )'''
    if lbn :
        ll_inputs = tf.keras.Input(shape=(5, 4), name = "LL")
        hl_inputs = tf.keras.Input(shape=(17,), name = "HL")
        lbn_layer = LBNLayer(ll_inputs.shape, 10, boost_mode=LBN.PAIRS, features=["E", "pt", "eta", "phi", "m", "pair_cos"])
        lbn_features = lbn_layer(ll_inputs)
        normalized_lbn_features = tf.keras.layers.BatchNormalization()(lbn_features)
        x = tf.keras.layers.concatenate([normalized_lbn_features, hl_inputs])
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.50)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.50)(x)
        outputs = tf.keras.layers.Dense(num_class, activation='softmax')(x)

        model = tf.keras.Model(inputs=[ll_inputs, hl_inputs], outputs= outputs, name='lbn_dnn')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=["accuracy"],
    )
    else :
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=[nr_trainvars]))
        model.add(BatchNormalization())
        model.add(keras.layers.Dense(256,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(keras.layers.Dense(256,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(num_class, activation='softmax'))
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Nadam(
                lr=.001
            ),
            metrics=['accuracy'],#metrics,
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
