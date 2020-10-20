import os
import numpy as np
import json
import pandas as pd
import itertools
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.utils.multiclass import type_of_target
from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_aux_tools as hhat
from machineLearning.machineLearning import nn_tools as nt
from machineLearning.machineLearning import multiclass_tools as mt
from machineLearning.machineLearning import hh_visualization_tools as hhvt


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap="summer")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=5, rotation=70)
    plt.yticks(tick_marks, class_names, fontsize=5)
    cm = np.moveaxis(
        np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
            decimals=2), 0, 1)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(i, j, cm[i, j], horizontalalignment="center", size=5)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.pdf")


def main(output_dir):
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
    preferences = hhat.get_hh_parameters(
        channel_dir,
        global_settings['tauID_training'],
        info_dir
    )
    data_dict = create_data_dict(preferences, global_settings)
    even_model = create_model(
        preferences, global_settings, data_dict, "even")
    odd_model = create_model(
        preferences, global_settings, data_dict, "odd")
    print(odd_model.summary())
    even_train_info, even_test_info = evaluate_model(
        even_model, data_dict, global_settings, "even")
    odd_train_info, odd_test_info = evaluate_model(
        odd_model, data_dict, global_settings, "odd")
    # hhvt.plotROC(
    #     [odd_train_info, odd_test_info],
    #     [even_train_info, even_test_info],
    #     global_settings
    # )
    # classes = set(data_dict["even_data"]["process"])
    # for class_ in classes:
    #     multitarget = list(set(
    #         data_dict["even_data"].loc[
    #             data_dict["even_data"]["process"] == class_, "multitarget"
    #         ]
    #     ))[0]
    #     print(str(class_) + '\t' + str(multitarget))
    #     hhvt.plot_sampleWise_bdtOutput(
    #         odd_model, data_dict["even_data"], preferences,
    #         global_settings, multitarget, class_, data_dict
    #     )

def create_data_dict(preferences, global_settings):
    data = dlt.load_data(
        preferences,
        global_settings,
        remove_neg_weights=True
    )
    hhat.normalize_hh_dataframe(
        data,
        preferences,
        global_settings
    )
    sumall = data.loc[data["process"] == "TT"]["totalWeight"].sum() \
        + data.loc[data["process"] == "W"]["totalWeight"].sum() \
        + data.loc[data["process"] == "DY"]["totalWeight"].sum() \
        + data.loc[data["target"] == 1]["totalWeight"].sum()
    print(
        "TT:W:DY \t" \
        + str(data.loc[data["process"] == "TT"]["totalWeight"].sum()/sumall) \
        + ":" + str(data.loc[data["process"] == "W"]["totalWeight"].sum()/sumall) \
        + ":" + str(data.loc[data["process"] == "DY"]["totalWeight"].sum()/sumall) \
        + "@" + str(data.loc[data["target"] == 1]["totalWeight"].sum()/sumall)
    )
    data = mt.multiclass_encoding(data)
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
        choose_data
):
    lbn = 1 if global_settings['ml_method'] == 'lbn' else 0
    trainvars = preferences['trainvars']
    nr_trainvars = len(trainvars)
    num_class = max((data_dict['odd_data']['multitarget'])) + 1
    number_samples = len(data_dict['odd_data']) if choose_data == "odd" else len(data_dict['even_data'])
    model_structure = nt.create_nn_model(
        nr_trainvars,
        num_class,
        lbn=lbn
    )
    if global_settings['ml_method'] == 'lbn':
        if choose_data == 'odd':
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
            epochs=2,
            batch_size=1024,
            sample_weight=train_data["train_data"]['totalWeight'].values,
            validation_data=(
                [val_data["ll"], val_data["hl"]],
                val_data["val_data"]["multitarget"],
                val_data["val_data"]["totalWeight"].values
            )
        )
    else:
        train_data = data_dict['odd_data'] if choose_data == "odd" else data_dict['even_data']
        val_data = data_dict['even_data']  if choose_data == "even" else data_dict['odd_data']
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
            )
        )
    fig1, ax = plt.subplots()
    pd.DataFrame(fitted_model.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.show()
    plt.yscale('log')
    plt.savefig("loss_sampleweight_%s.png" %choose_data)

    #feature_importance = nt.get_feature_importances(model_structure, data_dict, preferences["trainvars"], choose_data)
    #print feature_importance
    return model_structure


def evaluate_model(model, data_dict, global_settings, choose_data):
    trainvars = data_dict['trainvars']
    train_data = data_dict["odd_data"] if choose_data == "odd" else data_dict["even_data"]
    test_data = data_dict["even_data"] if choose_data == "odd" else data_dict["odd_data"]

    if global_settings['ml_method'] == 'lbn':
        if choose_data == 'odd':
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
        train_predicted_probabilities = model.predict_proba(
            train_data[trainvars].values)
        test_predicted_probabilities = model.predict_proba(
            test_data[trainvars].values)
    cm = confusion_matrix(
        train_data["multitarget"].astype(int),
        np.argmax(train_predicted_probabilities, axis=1)
    )
    plot_confusion_matrix(cm, ["TT","W", "HH", "DY"])
    test_fpr, test_tpr= mt.roc_curve(
        data_dict['even_data']['multitarget'].astype(int),
        test_predicted_probabilities,
        data_dict['even_data']['totalWeight'].astype(float)
    )
    train_fpr, train_tpr = mt.roc_curve(
        data_dict['odd_data']['multitarget'].astype(int),
        train_predicted_probabilities,
        data_dict['odd_data']['totalWeight'].astype(float)
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




if __name__ == '__main__':
    main('None')
