"""Tools for creating a neural network model and evaluating it
"""
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import r2_score
import numpy as np
import tensorflow as tf
import universal_tools as ut
import evaluation_tools as et
from lbn import LBN, LBNLayer
import multiclass_tools as mt
import data_loading_tools as dlt
from visualization import hh_visualization_tools as hhvt
from grouped_entropy import GroupedXEnt as gce

PARTICLE_INFO = low_level_object = {
    'bb1l': ["bjet1", "bjet2", "wjet1", "wjet2", 'jet1', 'jet2', "lep", "met"],
    'bb2l': ["bjet1", "bjet2", "lep1", "lep2"]
}
tf.config.threading.set_intra_op_parallelism_threads(3)
tf.config.threading.set_inter_op_parallelism_threads(3)
class NNmodel(object):
    def __init__(
            self,
            train_data,
            val_data,
            trainvars,
            parameters,
            plot_history=True,
            split_ggf_vbf=False,
            output_dir='',
            addition='',
            model='classifier',
            weight='totalWeight',
            ResNet=True,
            ResNetLayer=2
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.trainvars = trainvars
        self.nr_trainvars = len(trainvars)
        self.dropout = 0#parameters['dropout']
        self.lr = parameters['lr']
        self.l2 = parameters['l2']
        self.epoch = 200 if 'epoch' not in parameters.keys() else parameters['epoch']
        self.batch_size = parameters['batch_size']
        self.layer = parameters['layer']
        self.node = parameters['node']
        self.nparticle = 12#parameters['nparticle']
        self.split_ggf_vbf = split_ggf_vbf
        self.weight = weight
        self.ResNet = ResNet
        self.ResNetLayer = ResNetLayer if 'resNetBlock' not in parameters.keys() else parameters['resNetBlock']
        if model == 'classifier':
            self.train_target = self.train_data['multitarget'].values.astype(np.float)
            self.val_target = self.val_data['multitarget'].values.astype(np.float)
            self.num_class = max((self.train_data['multitarget'])) + 1
            self.activation = 'relu'
            self.output_activation = 'softmax'
            self.loss = 'sparse_categorical_crossentropy'
            self.metrics = 'accuracy'
            print('layer: '+ str(self.layer))
        elif model == 'binary_classifier':
            self.train_target = self.train_data['target'].values.astype(np.float)
            self.val_target = self.val_data['target'].values.astype(np.float)
            self.num_class = 1
            self.loss = 'binary_crossentropy'
            self.activation = 'relu'
            self.output_activation = 'sigmoid'
            self.metrics = 'accuracy'
        else:
            self.train_target = self.train_data['logtarget'].values.astype(np.float)
            self.val_target = self.val_data['logtarget'].values.astype(np.float)
            self.num_class = 1
            #self.dropout = parameters['drop_out']
            self.activation = 'relu'
            self.output_activation = 'selu'
            self.loss = 'mean_squared_error'
            self.epoch = parameters["epoch"]
            self.metrics = 'mean_squared_error'

        '''if split_ggf_vbf:
            group_ids = mt.group_id(self.train_data)
            self.gce = gce(group_ids)
            self.train_target = train_data[['GGF_HH', 'VBF_HH', 'TT', 'ST', 'Other', 'W', 'DY']].values.astype(float)
            self.val_target = val_data[['GGF_HH', 'VBF_HH', 'TT', 'ST', 'Other', 'W', 'DY']].values.astype(float)'''

        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, min_delta=0.01
        )
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, min_delta=0.01,
            restore_best_weights=True
        )
        self.categorical_var_index = None
        self.categorical_vars = ["SM", "BM1", "BM2", "BM3", "BM4", "BM5", "BM6",\
             "BM7", "BM8", "BM9", "BM10", "BM11", "BM12"]
        self.plot_history = plot_history
        self.output_dir = output_dir
        self.addition = addition

    def Normal(self, ref, const=None, ignore_zeros=False, name=None, **kwargs):
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
        if const is not None:
            mean[const] = 0
            std[const] = 1
        std = np.where(std == 0, 1, std)
        mul = 1.0 / std
        add = -mean / std
        return tf.keras.layers.Lambda((lambda x: (x * mul) + add), name=name)

    def normalize_inputs(self, trainvars, inputs):
        input_var = np.array([self.train_data[var] for var in trainvars])
        categorical_var_index = [trainvars.index(categorical_var) for categorical_var in self.categorical_vars \
            if categorical_var in trainvars]
        self.normalized_vars = self.Normal(ref=input_var, const=None, axis=1)(inputs)

    def make_hidden_layer(self, x):
        for layer in range(0, self.layer):
            x = tf.keras.layers.Dense(self.node, activation=self.activation,
                     kernel_regularizer=tf.keras.regularizers.l2(self.l2))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
        return tf.keras.layers.Dense(self.num_class, activation=self.output_activation)(x)

    def res_net_block(self, inputs):
        x = inputs
        for layer in range(0, self.layer-1):
            print('layer: '+ str(layer))
            x = tf.keras.layers.Dense(self.node, activation=self.activation,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.l2))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(inputs.shape[-1],
                     kernel_regularizer=tf.keras.regularizers.l2(self.l2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.add([x, inputs])
        res_inputs = tf.keras.layers.Activation('relu')(x)
        return res_inputs

    def makeResNet(self, inputs):
        res_inputs = inputs
        for reslayer in range(0, self.ResNetLayer):
            print('resnet : '+ str(reslayer))
            res_inputs = self.res_net_block(res_inputs)
        x = tf.keras.layers.BatchNormalization()(res_inputs)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        return tf.keras.layers.Dense(self.num_class, activation=self.output_activation)(x)

    def compile_model(self):
        my_loss = self.loss
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.lr),
            loss=my_loss,
            weighted_metrics=[self.metrics]
        )

    def create_model(self):
        self.inputs = tf.keras.Input(shape=(self.nr_trainvars,), name="input_var")
        self.normalize_inputs(self.trainvars, self.inputs)
        x = tf.keras.layers.Layer()(self.normalized_vars)
        if self.ResNet:
            outputs = self.makeResNet(self.normalized_vars)
        else:
            outputs = self.make_hidden_layer(x)
        self.model = tf.keras.Model(
            inputs=[self.inputs],
            outputs=outputs
        )
        self.compile_model()
        self.fit_model()
        return self.model

    def fit_model(self):
        history = self.model.fit(
            self.train_data[self.trainvars].values,
            self.train_target,
            epochs=self.epoch,
            batch_size=self.batch_size,
            sample_weight=self.train_data[self.weight].values,
            validation_data=(
                self.val_data[self.trainvars],
                self.val_target,
                self.val_data[self.weight].values
            ),
            callbacks=[self.reduce_lr, self.early_stopping]
        )
        if self.plot_history:
            hhvt.plot_loss_accuracy(history, self.output_dir, self.addition)

class LBNmodel(NNmodel):
    def __init__(
            self,
            train_data,
            val_data,
            trainvars,
            channel,
            parameters,
            plot_history=True,
            split_ggf_vbf=False,
            output_dir='',
            addition=''
    ):
        super(LBNmodel, self).__init__(
            train_data,
            val_data,
            trainvars,
            parameters,
            plot_history,
            split_ggf_vbf,
            output_dir,
            addition
        )
        self.particles = {
            'bb1l': ["bjet1", "bjet2", "wjet1", "wjet2", 'jet1', 'jet2', "lep", "met"],
            'bb2l': ["bjet1", "bjet2", "lep1", "lep2"]
        }
        self.channel = channel
        assert self.channel in self.particles.keys(), 'channel %s is absent in self.particles %s'\
            %(self.channel, self.particles)

    def fit_model(self):
        history = self.model.fit(
            [dlt.get_low_level(self.train_data, self.particles[self.channel]),
             dlt.get_high_level(self.train_data, self.particles[self.channel], self.trainvars)],
            self.train_target,
            epochs=self.epoch,
            batch_size=self.batch_size,
            #class_weight={0:1/2, 1:1, 2:1, 3:1, 4:1, 5:1},
            sample_weight=self.train_data[self.weight].values,
            validation_data=(
                [dlt.get_low_level(self.val_data, self.particles[self.channel]),
                 dlt.get_high_level(self.val_data, self.particles[self.channel], self.trainvars)],
                self.val_target,
                self.val_data[self.weight].values
            ),
            callbacks=[self.reduce_lr, self.early_stopping]
        )
        if self.plot_history:
            hhvt.plot_loss_accuracy(history, self.output_dir, self.addition)

    def create_model(self):
        self.low_level_var = ["%s_%s" %(part, var) for part in self.particles[self.channel] \
             for var in ["e", "px", "py", "pz"]]
        self.nr_trainvars = len(self.trainvars)
        ll_inputs = tf.keras.Input(shape=(len(self.particles[self.channel]), 4), name="LL")
        hl_inputs = tf.keras.Input(shape=(self.nr_trainvars,), name="HL")
        lbn_layer = LBNLayer(
            ll_inputs.shape, self.nparticle,
            boost_mode=LBN.PAIRS,
            features=["E", "px", "py", "pz", "pt", "p", "m", "pair_cos"]
        )
        lbn_features = lbn_layer(ll_inputs)
        self.normalized_lbn_features = tf.keras.layers.BatchNormalization()(lbn_features)
        hl_var = [var for var in self.trainvars]# if var not in self.low_level_var]
        self.normalize_inputs(hl_var, hl_inputs)
        x = tf.keras.layers.concatenate([self.normalized_lbn_features, self.normalized_vars])
        if self.ResNet:
            outputs = self.makeResNet(x)
        else:
            outputs = self.make_hidden_layer(x)
        self.model = tf.keras.Model(
            inputs=[ll_inputs, hl_inputs],
            outputs=outputs,
            name='lbn_dnn'
        )
        self.compile_model()
        self.fit_model()

        return self.model

def parameter_evaluation(
        nn_hyperparameters,
        data_dict,
        nthread,
        num_class,
):
    """Creates the NN model according to the given hyperparameters
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
    """
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

def evaluate_model(data_dict, global_settings, model, lbn=True):
    """Evaluates the model for the XGBoost method
    Parameters:
    ----------
    data_dict : dict
        Contains all the necessary information for the evaluation.
    global_settings : dict
        Preferences for the optimization
    model : XGBoost Booster?
        Model created by the xgboost.
    Returns:
    -------
    score : float
        The score calculated according to the fitness_fn
    """
    trainvars = data_dict['trainvars']
    train_data = data_dict['train']
    test_data = data_dict['test']
    if lbn:
        particles = PARTICLE_INFO[global_settings['channel']]
        pred_train = model.predict(
            [dlt.get_low_level(train_data, particles),
             dlt.get_high_level(train_data, particles, trainvars)],
          batch_size=1024)
        pred_test = model.predict(
            [dlt.get_low_level(test_data, particles),
          dlt.get_high_level(test_data, particles, trainvars)],
          batch_size=1024)
    else:
        pred_train = model.predict(
          train_data[trainvars].values,
          batch_size=1024)
        pred_test = model.predict(
          test_data[trainvars].values,
          batch_size=1024)
    del train_data, test_data
    kappa = global_settings['kappa']
    if global_settings['fitness_fn'] == 'd_roc':
        return et.calculate_d_roc(
            data_dict, pred_train, pred_test, kappa=kappa, multiclass=True)
    elif global_settings['fitness_fn'] == 'd_ams':
        return et.calculate_d_ams(
            data_dict, pred_train, pred_test, kappa=kappa)
    else:
        raise ValueError(
            'The' + str(global_settings['fitness_fn'])
            + ' fitness_fn is not implemented'
        )

def evaluate_model_regression(data_dict, global_settings, model):
    trainvars = data_dict['trainvars']
    train_data = data_dict['train']
    test_data = data_dict['test']
    pred_train = model.predict(
        train_data[trainvars],
        batch_size=1024)
    pred_test = model.predict(
        test_data[trainvars],
        batch_size=1024)
    kappa = global_settings['kappa']
    train_score = r2_score(train_data['logtarget'], pred_train, sample_weight=train_data['genWeight'])
    test_score = r2_score(test_data['logtarget'], pred_test, sample_weight=test_data['genWeight'])
    return et.calculate_d_score(train_score, test_score, kappa=kappa), test_score, train_score


def evaluate(k_model, data_dict, global_settings):
    """Evaluates the nn k_model

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
    """
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
    """Returns the feature importance relevant for neural network case using
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
    """
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
    """Calculates the number of nodes in a hidden layer

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
    """
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
    """Creates the hidden net structure for the NN

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
    """
    number_nodes = calculate_number_nodes_in_hidden_layer(
        number_classes,
        number_trainvars,
        number_samples,
        alpha
    )
    number_nodes = int(np.floor(number_nodes/number_hidden_layers))
    hidden_net = [number_nodes] * number_hidden_layers
    return hidden_net


class NNFeatureImportances(object):
    """ Class for finding the feature importances for NN or LBN models"""
    def __init__(
            self, model, data, trainvars, weight='totalWeight',
            target='multitarget', permutations=5
    ):
        self.model = model
        self.trainvars = trainvars
        self.weight = weight
        self.weights = data[weight]
        self.target = target
        self.labels = data[target]
        self.data = data[trainvars]
        self.permutations = permutations

    def permutation_importance(self):
        print('Starting permutation importance')
        score_dict = {}
        prediction = self.predict_from_model(self.data)
        original_score = calculate_acc_with_weights(
            prediction, self.labels, self.weights
        )
        print('Reference score: ' + str(original_score))
        for trainvar in self.trainvars:
            print(trainvar)
            data_ = self.data.copy()
            t_score = 0
            for i in range(self.permutations):
                print("Permutation nr: " + str(i))
                data_[trainvar] = np.random.permutation(data_[trainvar])
                prediction = self.predict_from_model(data_)
                score = calculate_acc_with_weights(
                    prediction, self.labels, self.weights
                )
                print(score)
                t_score += score
            score_dict[trainvar] = abs(
                original_score - (t_score/self.permutations))
        sorted_sd = sorted(
            score_dict.items(), key=lambda kv: kv[1], reverse=True)
        print(sorted_sd)
        return score_dict

    def calculate_acc_with_weights(self, prediction, labels, weights):
        num_classes = len(prediction[0])
        pred_labels = [np.argmax(event) for event in prediction]
        true_positives = 0
        true_negatives = 0
        total_positive = sum(self.weights)
        #total_negative = sum(self.weights)
        for pred, true, weight in zip(pred_labels, self.labels, self.weights):
            if pred == true:
                true_positives += weight
                '''true_negatives += num_classes * weight
            else:
                true_negatives += (num_classes - 1) * weight
        true_negatives /= num_classes'''
        #accuracy = (true_positives + true_negatives) / (total_positive + total_negative)
        accuracy = (true_positives) / (total_positive)
        return accuracy

    def predict_from_model(self, data_):
        prediction = self.model.predict(data_)
        return prediction


class LBNFeatureImportances(NNFeatureImportances):
    def __init__(
        self, model, data, trainvars, channel, weight='totalWeight',
        target='multitarget', ml_method='lbn',
        permutations=5
    ):
        super(LBNFeatureImportances, self).__init__(
            model, data, trainvars, weight, target, permutations
        )
        self.particles = {
            'bb1l': ["bjet1", "bjet2", "wjet1", "wjet2", "jet1", "jet2", "lep", "met"],
            'bb2l': ["bjet1", "bjet2", "lep1", "lep2"]
        }
        self.channel = channel
        assert self.channel in self.particles.keys(), 'channel %s is absent in self.particles %s'\
            %(self.channel, self.particles)

    def predict_from_model(self, data_):
        ll = dlt.get_low_level(data_, self.particles[self.channel])
        hl = dlt.get_high_level(data_, self.particles[self.channel], self.trainvars)
        prediction = self.model.predict([ll, hl])
        return prediction

    def custom_permutation_importance(
        self
):
        print('Starting permutation importance')
        score_dict = {}
        prediction = self.predict_from_model(self.data)
        original_score = self.calculate_acc_with_weights(prediction, self.labels, self.weights)
        print('Reference score: ' + str(original_score))
        for trainvar in self.trainvars:
            print(trainvar)
            #data_copy = self.data.copy()
            trainvar_copy = np.copy(self.data[trainvar])
            t_score = 0
            for i in range(self.permutations):
                print("Permutation nr: " + str(i))
                #data_copy[trainvar] = np.random.permutation(data_copy[trainvar])
                self.data[trainvar] = np.random.permutation(self.data[trainvar])
                #prediction = self.predict_from_model(data_copy)
                prediction = self.predict_from_model(self.data)
                score = self.calculate_acc_with_weights(prediction, self.labels, self.weights)
                print(score)
                t_score += score
            self.data[trainvar] = trainvar_copy
            score_dict[trainvar] = abs(original_score - (t_score/self.permutations))
        sorted_sd = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
        print(sorted_sd)
        return score_dict

    def lbn_feature_importances(self, data_dict, case='even'):
        labels = data_dict[case + '_data'][self.target]
        weights = data_dict[case + '_data'][self.weight]
        ll_names = ['b1jets', 'b2jets', 'w1jets', 'w2jets', 'leptons']
        high_level = data_dict['hl_' + case]
        low_level = data_dict['ll_' + case]
        reference_prediction = self.model.predict(
            [low_level, high_level], batch_size=1024)
        reference_score = calculate_acc_with_weights(
            reference_prediction, labels, weights)
        print(reference_prediction)
        score_dict = {}
        for i, feature in enumerate(ll_names):
            print(feature)
            t_score = 0
            for permutation in range(self.permutations):
                print('Permutation: ' + str(i))
                shuffled_low_level = shuffle_low_level(low_level, i)
                prediction = self.model.predict(
                    [shuffled_low_level, high_level], batch_size=1024)
                score = calculate_acc_with_weights(prediction, labels, weights)
                t_score += score
            score_dict[feature] = abs(reference_score - (t_score/permutations))
        return score_dict

    def shuffle_low_level(self, low_level, i):
        low_level_ = low_level.copy()
        shuffled_elements = np.random.permutation(low_level_[:, i])
        low_level_[:, i] = shuffled_elements
        return low_level_

def model_evaluation_main(hyperparameters, data_dict, global_settings, model='lbn_classifier'):
    """ Collected functions for CGB model evaluation
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
    score : flot
        The score calculated according to the fitness_fn
    """
    '''k_model = parameter_evaluation(
        nn_hyperparameters,
        data_dict,
        global_settings['nthread'],
        global_settings['num_classes'],
    )
    score, pred_train, pred_test = evaluate(
        k_model, data_dict, global_settings
    )
    return score, pred_train, pred_test'''
    if model == 'lbn_classifier':
        LBNmodel_ = LBNmodel(
            data_dict['train'],
            data_dict['val'],
            data_dict['trainvars'],
            global_settings['channel'],
            hyperparameters,
            False
        )
        model = LBNmodel_.create_model()
        score, test, train = evaluate_model(
            data_dict, global_settings, model)
        return score, test, train
    elif model == 'nn_classifier':
        NNmodel_ = NNmodel(
            data_dict['train'],
            data_dict['val'],
            data_dict['trainvars'],
            hyperparameters,
            False,
            model='classifier',
            ResNet=True
        )
        model = NNmodel_.create_model()
        score, train, test = evaluate_model(
            data_dict, global_settings, model, lbn=False)
        return score, train, test
    else:
        NNmodel_ = NNmodel(
            data_dict['train'],
            data_dict['test'],
            data_dict['trainvars'],
            hyperparameters,
            False,
            model='regression',
            weight='genWeight'
        )
        model = NNmodel_.create_model()
        score, train, test = evaluate_model_regression(
            data_dict, global_settings, model)
        return score, train, test
