'''
Call with 'python'

Usage: 
    hh_nnTraining.py
    hh_nnTraining.py [--save_model=INT]

Options:
    -s --save_model=INT             Whether or not to save the model [default: 0]
'''
import os
import docopt
import numpy as np
import json
import pandas as pd
import itertools
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve
import matplotlib
from datetime import datetime
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.utils.multiclass import type_of_target
from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import nn_tools as nt
from machineLearning.machineLearning import hh_parameter_reader as hpr
from machineLearning.machineLearning import hh_tools as hht
from machineLearning.machineLearning import multiclass_tools as mt
from machineLearning.machineLearning.visualization import hh_visualization_tools as hhvt
import cmsml

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

PARTICLE_INFO = low_level_object = {
    'bb1l': ["bjet1", "bjet2", "wjet1", "wjet2", "lep"],
    'bb2l': ["bjet1", "bjet2", "lep1", "lep2"]
}

def plot_confusion_matrix(cm, class_names, output_dir):
    figure = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap="summer")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=5, rotation=70)
    plt.yticks(tick_marks, class_names, fontsize=5)
    cm = np.moveaxis(
        np.around(
            cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
            decimals=2),
        0, 1
    )
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(i, j, cm[i, j], horizontalalignment="center", size=5)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    outfile = os.path.join(output_dir, 'confusion_matrix_resolved.png') \
              if output_dir.find("resolved") != -1 \
                 else os.path.join(output_dir, 'confusion_matrix_boosted.png')
    plt.savefig(outfile, bbox_inches='tight')
    plt.close('all')


def main(output_dir, save_model):
    if output_dir == 'None':
        settings_dir = os.path.join(
            os.path.expandvars('$CMSSW_BASE'),
            'src/machineLearning/machineLearning/settings'
        )
    global_settings = ut.read_settings(settings_dir, 'global')
    if output_dir == 'None':
        output_dir = global_settings['output_dir']
    else:
        global_settings['output_dir'] = output_dir
    global_settings['output_dir'] = os.path.expandvars(
        global_settings['output_dir'])
    if not os.path.exists(global_settings['output_dir']):
        os.makedirs(global_settings['output_dir'])
    channel_dir, info_dir, _ = ut.find_settings()
    scenario = global_settings['scenario']
    reader = hpr.HHParameterReader(channel_dir, scenario)
    preferences = reader.parameters
    preferences = define_trainvars(global_settings, preferences)
    data_dict = create_data_dict(preferences, global_settings)
    even_model = create_model(
        preferences, global_settings, data_dict, "even_data", save_model)
    odd_model = create_model(
        preferences, global_settings, data_dict, "odd_data", save_model)
    print(odd_model.summary())
    even_train_info, even_test_info = evaluate_model(
        even_model, data_dict, global_settings, "even_data")
    odd_train_info, odd_test_info = evaluate_model(
        odd_model, data_dict, global_settings, "odd")
    if global_settings['ml_method'] != 'lbn' and global_settings['feature_importance'] == 1:
        trainvars = preferences['trainvars']
        data = data_dict['odd_data']
        importance_calculator = nt.NNFeatureImportances(
            even_model, data, trainvars, weight='evtWeight',
            target='multitarget'
        )
        score_dict = importance_calculator.permutation_importance()
    elif global_settings['feature_importance'] == 1:
        raise NotImplementedError(
            'Please implement the LBN part to be used with the LBN feature'
            'importances'
        )
        trainvars = preferences['trainvars']
        data = data_dict['odd_data']
        particles = PARTICLE_INFO[global_settings['channel']]
        importance_calculator = nt.LBNFeatureImportances(
            even_model, data, particles, trainvars, weight='evtWeight',
            target='multitarget'
        )
        score_dict = importance_calculator.permutation_importance()
    hhvt.plot_feature_importances_from_dict(
        score_dict, global_settings['output_dir'])
    hhvt.plotROC(
        [odd_train_info, odd_test_info],
        [even_train_info, even_test_info],
        global_settings
    )
    classes = set(data_dict["even_data"]["process"])
    for class_ in classes:
        multitarget = list(set(
            data_dict["even_data"].loc[
                data_dict["even_data"]["process"] == class_, "multitarget"
            ]
        ))[0]
        print(str(class_) + '\t' + str(multitarget))
        hhvt.plot_nn_sampleWise_bdtOutput(
            odd_model, data_dict["even_data"], preferences,
            global_settings, multitarget, class_, data_dict
        )


def create_data_dict(preferences, global_settings):
    if os.path.exists(preferences['data_csv']):
        data = pandas.read_csv(preferences['data_csv'])
    else:
        normalizer = hht.HHDataNormalizer
        loader = hht.HHDataLoader(
            normalizer,
            preferences,
            global_settings
        )
        data = loader.data
    for trainvar in preferences['trainvars']:
        if str(data[trainvar].dtype) == 'object':
            try:
                data[trainvar] = data[trainvar].astype(int)
            except:
                continue
    hhvt.plot_single_mode_correlation(
        data, preferences['trainvars'],
        global_settings['output_dir'], 'trainvar'
    )
    hhvt.plot_trainvar_multi_distributions(
        data, preferences['trainvars'],
        global_settings['output_dir']
    )
    sumall = data.loc[data["process"] == "TT"]["totalWeight"].sum() \
        + data.loc[data["process"] == "W"]["totalWeight"].sum() \
        + data.loc[data["process"] == "DY"]["totalWeight"].sum() \
        + data.loc[data["process"] == "ST"]["totalWeight"].sum() \
        + data.loc[data["process"] == "Other"]["totalWeight"].sum() \
        + data.loc[data["target"] == 1]["totalWeight"].sum()
    print(
        "TT:W:DY:HH \t" \
        + str(data.loc[data["process"] == "TT"]["totalWeight"].sum()/sumall) \
        + ":" + str(data.loc[data["process"] == "W"]["totalWeight"].sum()/sumall) \
        + ":" + str(data.loc[data["process"] == "DY"]["totalWeight"].sum()/sumall) \
        + ":" + str(data.loc[data["target"] == 1]["totalWeight"].sum()/sumall)
    )
    data = mt.multiclass_encoding(data)
    hhvt.plot_correlations(data, preferences["trainvars"], global_settings)
    even_data = data.loc[(data['event'].values % 2 == 0)]
    odd_data = data.loc[~(data['event'].values % 2 == 0)]
    if global_settings['ml_method'] == 'lbn':
        ll_odd = dlt.get_low_level(odd_data)
        ll_even = dlt.get_low_level(even_data)
        hl_odd = dlt.get_high_level(odd_data, preferences["trainvars"])
        hl_even = dlt.get_high_level(even_data, preferences["trainvars"])
        data_dict = {
            'trainvars': preferences['trainvars'],
            'odd_data':  odd_data,
            'even_data': even_data,
            'll_odd' : ll_odd,
            'll_even' : ll_even,
            'hl_odd' : hl_odd,
            'hl_even' : hl_even
        }
    else:
        data_dict = {
            'trainvars': preferences['trainvars'],
            'odd_data':  odd_data,
            'even_data': even_data
        }
    return data_dict


def create_model(
        preferences,
        global_settings,
        data_dict,
        choose_data,
        save_model
):
    lbn = 1 if global_settings['ml_method'] == 'lbn' else 0
    trainvars = preferences['trainvars']
    nr_trainvars = len(trainvars)
    num_class = max((data_dict['odd_data']['multitarget'])) + 1
    number_samples = len(data_dict[choose_data])
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.001
    )
    input_var = np.array([data_dict[choose_data][var] for var in preferences["trainvars"]])
    categorical_var_index = [data_dict[choose_data].columns.get_loc(c) for c in ["nodeX"] \
                             if c in data_dict[choose_data]]
    model_structure = nt.create_nn_model(
        nr_trainvars,
        num_class,
        input_var,
        categorical_var_index,
        lbn=lbn
    )
    if global_settings['ml_method'] == 'lbn':
        if choose_data == 'odd_data':
            train_data = {
                "ll": data_dict["ll_odd"],
                "hl": data_dict['hl_odd'],
                "train_data": data_dict["odd_data"]
            }
            val_data = {
                "ll": data_dict["ll_even"],
                "hl": data_dict['hl_even'],
                "val_data": data_dict["even_data"]
            }
        else:
            train_data = {
                "ll": data_dict["ll_even"],
                "hl": data_dict['hl_even'],
                "train_data": data_dict["even_data"]
            }
            val_data = {
                "ll": data_dict["ll_odd"],
                "hl": data_dict['hl_odd'],
                "val_data": data_dict["odd_data"]
            }
        fitted_model = model_structure.fit(
            [train_data["ll"], train_data["hl"]],
            train_data["train_data"]['multitarget'].values,
            epochs=25,
            batch_size=1024,
            sample_weight=train_data["train_data"]['totalWeight'].values,
            validation_data=(
                [val_data["ll"], val_data["hl"]],
                val_data["val_data"]["multitarget"],
                val_data["val_data"]["totalWeight"].values
            ),
            callbacks=[reduce_lr]
        )
    else:
        train_data = data_dict['odd_data'] if choose_data == "odd_data" else data_dict['even_data']
        val_data = data_dict['even_data']  if choose_data == "even_data" else data_dict['odd_data']
        fitted_model = model_structure.fit(
            train_data[trainvars].values,
            train_data['multitarget'].astype(np.int),
            epochs=25,
            batch_size=1024,
            sample_weight=train_data['totalWeight'].values,
            validation_data=(
                val_data[trainvars],
                val_data['multitarget'].astype(np.int),
                val_data['totalWeight'].values
            ),
            callbacks=[reduce_lr]
        )
    if save_model:
        pb_filename = os.path.join(global_settings["output_dir"], "multiclass_DNN_w%s_for_%s_%s_%s.pb"\
                                %(global_settings["ml_method"], global_settings["channel"], \
                                  global_settings["mode"], choose_data))
        log_filename = os.path.join(global_settings["output_dir"], "multiclass_DNN_w%s_for_%s_%s_%s.log"\
                                %(global_settings["ml_method"], global_settings["channel"], \
                                  global_settings["mode"], choose_data))
        cmsml.tensorflow.save_graph(pb_filename, model_structure, variables_to_constants=True)
        file = open(log_filename, "w")
        file.write(str(trainvars))
        file.close()
    fig1, ax = plt.subplots()
    pd.DataFrame(fitted_model.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()
    plt.yscale('log')
    plt.savefig("loss_sampleweight_%s.png" %choose_data)
    plt.close('all')

    return model_structure


def evaluate_model(model, data_dict, global_settings, choose_data):
    trainvars = data_dict['trainvars']
    train_data = data_dict["odd_data"] if choose_data == "odd_data" else data_dict["even_data"]
    test_data = data_dict["even_data"] if choose_data == "odd_data" else data_dict["odd_data"]
    train_data["max_node_pos"] = -1
    train_data["max_node_val"] = -1

    if global_settings['ml_method'] == 'lbn':
        if choose_data == 'odd_data':
            train_var = {
                'll': data_dict['ll_odd'],
                'hl': data_dict['hl_odd']
            }
            test_var = {
                'll': data_dict['ll_even'],
                'hl': data_dict['hl_even']
            }
        else:
            train_var = {
                'll': data_dict['ll_even'],
                'hl': data_dict['hl_even']
            }
            test_var = {
                'll': data_dict['ll_odd'],
                'hl': data_dict['hl_odd']\
            }
        train_predicted_probabilities = model.predict(
            [train_var["ll"], train_var["hl"]], batch_size=1024)
        test_predicted_probabilities = model.predict(
            [test_var["ll"], test_var["hl"]], batch_size=1024)
    else:
        train_predicted_probabilities = model.predict(
            train_data[trainvars].values)
        test_predicted_probabilities = model.predict(
            test_data[trainvars].values)
    cm = confusion_matrix(
        train_data["multitarget"].astype(int),
        np.argmax(train_predicted_probabilities, axis=1),
        sample_weight=train_data["evtWeight"].astype(float)
    )
    samples = []
    for i in sorted(set(train_data["multitarget"])):
        samples.append(list(set(train_data.loc[train_data["multitarget"] == i]["process"]))[0])
    samples = ['HH' if x.find('signal') != -1 else x for x in samples]
    plot_confusion_matrix(
        cm, samples, global_settings['output_dir'])

    if global_settings['ml_method'] != 'lbn':
        for process in set(train_data["process"]):
            data = train_data.loc[train_data["process"] == process]
            value = model.predict(data[trainvars].values)
            train_data.loc[train_data["process"] == process, "max_node_pos"]\
                = np.argmax(value, axis=1)
            train_data.loc[train_data["process"] == process, "max_node_val"] \
                = np.amax(value, axis=1)
    if global_settings['ml_method'] == 'lbn':
        for process in set(train_data["process"]):
            idx = np.where(np.array(train_data["process"]) == process)[0]
            value = model.predict(
                [train_var["ll"][idx], train_var["hl"][idx]], batch_size=1024)
            train_data.loc[train_data["process"] == process, "max_node_pos"] = np.argmax(value, axis=1)
            train_data.loc[train_data["process"] == process, "max_node_val"] = np.amax(value, axis=1)

    color = ['b', 'g', 'y', 'r', 'magenta', 'orange']
    for node in sorted(set(train_data["multitarget"])):
        fig1, ax = plt.subplots()
        values = []
        weights = []
        labels = []
        colors = []
        for i, process in enumerate(set(train_data["process"])):
            values.append(train_data.loc[((train_data["max_node_pos"] == node) & \
                                          (train_data["process"] == process)), ["max_node_val"]].values.tolist())
            weights.append(train_data.loc[((train_data["max_node_pos"] == node) & \
                                           (train_data["process"] == process)), ["evtWeight"]].values.tolist())
            labels.append(process) if process.find('signal') == -1 else labels.append("HH")
            colors.append(color[i])
        plt.hist(values, weights=weights,
                 label=labels, color=colors,
                 histtype='bar', stacked=True, range=(0, 1), bins=20)
        nodeName = list(set(train_data.loc[train_data["multitarget"] == node]["process"]))[0]
        plt.legend(loc='best', title=nodeName+"_node")
        plt.yscale('log')
        outfile = os.path.join(global_settings["output_dir"], 'DNNScore_'+nodeName+'_node_resolved.png')\
                  if global_settings["dataCuts"].find("resolved") != -1 \
                     else os.path.join(global_settings["output_dir"], 'DNNScore_'+nodeName+'_node_boosted.png')
        plt.savefig(outfile)
        plt.clf()
    test_fpr, test_tpr = mt.roc_curve(
        test_data['multitarget'].astype(int),
        test_predicted_probabilities,
        test_data['evtWeight'].astype(float)
    )
    train_fpr, train_tpr = mt.roc_curve(
        train_data['multitarget'].astype(int),
        train_predicted_probabilities,
        train_data['evtWeight'].astype(float)
    )
    train_auc = auc(train_fpr, train_tpr, reorder=True)
    test_auc = auc(test_fpr, test_tpr, reorder=True)
    test_info = {
        'fpr': test_fpr,
        'tpr': test_tpr,
        'auc': test_auc,
        'type': 'test',
        'prediction': test_predicted_probabilities
    }
    train_info = {
        'fpr': train_fpr,
        'tpr': train_tpr,
        'auc': train_auc,
        'type': 'train',
        'prediction': train_predicted_probabilities
    }
    return train_info, test_info


def define_trainvars(global_settings, preferences):
    if global_settings["ml_method"] == "lbn" :
        trainvars_path = os.path.join(info_dir, 'trainvars_resolved_lbn.json')
    if global_settings["dataCuts"].find("boosted") != -1 :
        trainvars_path = os.path.join(info_dir, 'trainvars_boosted.json')
    if global_settings["dataCuts"].find("boosted") != -1 and global_settings["ml_method"] == "lbn":
        trainvars_path = os.path.join(info_dir, 'trainvars_boosted_lbn.json')
    if global_settings["dataCuts"].find("boosted") == -1 and global_settings["ml_method"] == "lbn":
        trainvars_path = os.path.join(info_dir, 'trainvars_resolved_lbn.json')
    try:
        trainvar_info = dlt.read_trainvar_info(trainvars_path)
        preferences['trainvars'] = []
        with open(trainvars_path, 'rt') as infile:
            for line in infile:
                info = json.loads(line)
                preferences['trainvars'].append(str(info['key']))
    except:
        print("Using trainvars from trainvars.json")
    return preferences


if __name__ == '__main__':
    startTime = datetime.now()
    try:
        arguments = docopt.docopt(__doc__)
        save_model = bool(int(arguments['--save_model']))
        main('None', save_model)
    except docopt.DocoptExit as e:
        print(e)
    print(datetime.now() - startTime)
