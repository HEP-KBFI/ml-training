""" Some helpful tools for plotting and data visualization
"""
import numpy as np
import os
import xgboost as xgb
from collections import OrderedDict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools

def plot_sampleWise_bdtOutput(
        model_odd,
        data_even,
        preferences,
        global_settings,
        weight='totalWeight',
):
    output_dir = global_settings['output_dir']
    data_even = data_even.copy()
    if 'nonres' in global_settings['scenario']:
        sig_name = 'HH_nonres_decay'
    else:
        sig_name = 'signal'
    data_even.loc[
        data_even['process'].str.contains('signal'), ['process']] = sig_name
    bkg_predictions = []
    bkg_labels = []
    bkg_weights = []
    bins = np.linspace(0., 1., 11)
    for process in set(data_even['process']):
        if process == sig_name:
            continue
        process_data = data_even.loc[data_even['process'] == process]
        process_prediction = np.array(model_odd.predict_proba(
            process_data[preferences['trainvars']]
        )[:, 1])
        weights = np.array(process_data[weight])
        bkg_weights.append(weights)
        bkg_predictions.append(process_prediction)
        bkg_labels.append(str(process))
    plt.hist(
        bkg_predictions, histtype='bar', label=bkg_labels, lw=2, bins=bins,
        weights=bkg_weights, alpha=1, stacked=True, normed=True
    )
    process_data = data_even.loc[data_even['process'] == sig_name]
    process_prediction = np.array(model_odd.predict_proba(
        process_data[preferences['trainvars']]
    )[:, 1])
    weights = np.array(process_data['totalWeight'])
    plt.hist(
        process_prediction, histtype='step', label=sig_name,
        lw=2, ec='k', alpha=1, normed=True, bins=bins, weights=weights
    )
    plt.legend()
    output_path = os.path.join(output_dir, 'sampleWise_bdtOutput.png')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')


def plot_feature_importances(model, global_settings, addition):
    fig, ax = plt.subplots()
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.tight_layout()
    plot_out = os.path.join(
        global_settings['output_dir'],
        addition + '_feature_importances.png'
    )
    fig.savefig(plot_out, bbox_inches='tight')
    plt.close('all')


def plotROC(odd_infos, even_infos, global_settings):
    output_dir = global_settings['output_dir']
    fig, ax = plt.subplots(figsize=(6, 6))
    linestyles = ['-', '--']
    for odd_info, linestyle in zip(odd_infos, linestyles):
        ax.plot(
            odd_info['fpr'], odd_info['tpr'], ls=linestyle, color='g',
            label='odd_' + odd_info['type'] + 'AUC = ' + str(
                round(odd_info['auc'], 4))
        )
    for even_info, linestyle in zip(even_infos, linestyles):
        ax.plot(
            even_info['fpr'], even_info['tpr'], ls=linestyle, color='r',
            label='even_' + even_info['type'] + 'AUC = ' + str(
                round(even_info['auc'], 4))
        )
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid()
    plot_out = os.path.join(output_dir, 'ROC_curve.png')
    plt.tight_layout()
    fig.savefig(plot_out, bbox_inches='tight')
    plt.close('all')


def plot_correlations(data, trainvars, global_settings):
    output_dir = global_settings['output_dir']
    classes = [('signal', 1), ('background', 0)]
    for mode in classes:
        mode_data = data.loc[data['target'] == mode[1]]
        plot_single_mode_correlation(mode_data, trainvars, output_dir, mode[0])
    plot_single_mode_correlation(data, trainvars, output_dir, 'total')


def plot_single_mode_correlation(data, trainvars, output_dir, addition):
    correlations = data[trainvars].corr()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap='viridis')
    ticks = np.arange(0, len(trainvars), 1)
    plt.rc('axes', labelsize=8)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(trainvars, rotation=-90)
    ax.set_yticklabels(trainvars)
    fig.colorbar(cax)
    fig.tight_layout()
    plot_out = os.path.join(output_dir, str(addition) + '_correlations.png')
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')


def plot_nodeWise_performance(
        global_settings, nodeWise_histo_dicts, mode
):
    output_dir = global_settings['output_dir']
    bins = np.linspace(0., 1., 11)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    for nodeWise_histo_dict in nodeWise_histo_dicts:
        node = nodeWise_histo_dict['node']
        plot_out = os.path.join(
            output_dir, mode + '_' + str(node) + '_nodeWisePredictions.png'
        )
        ###########################################
        values = plt.hist(
            nodeWise_histo_dict['sig_test'],
            weights=nodeWise_histo_dict['sig_test_w'], bins=bins,
            histtype='step', ec='orange', ls='--', normed=True, label='SIG_test'
        )[0]
        values_uw = np.histogram(
            np.array(nodeWise_histo_dict['sig_test'], dtype=float),
            bins=bins
        )[0]
        yerrors = [(np.sqrt(uw)/uw)*w for uw, w in zip(values_uw, values)]
        plt.errorbar(
            bin_centers, values, yerr=yerrors, fmt='none', color='orange', ec='orange',
            lw=2
        )
        ###########################################
        values = plt.hist(
            nodeWise_histo_dict['bkg_test'],
            weights=nodeWise_histo_dict['bkg_test_w'], bins=bins,
            histtype='step', ec='g', ls='--', normed=True, label='BKG_test'
        )[0]
        values_uw = np.histogram(
            np.array(nodeWise_histo_dict['bkg_test'], dtype=float),
            bins=bins
        )[0]
        yerrors = [(np.sqrt(uw)/uw)*w for uw, w in zip(values_uw, values)]
        plt.errorbar(
            bin_centers, values, yerr=yerrors, fmt='none', color='g', ec='g',
            lw=2
        )
        ###########################################
        values = plt.hist(
            nodeWise_histo_dict['sig_train'],
            weights=nodeWise_histo_dict['sig_train_w'], bins=bins,
            histtype='step', ec='r', ls='-', normed=True, label='SIG_train'
        )[0]
        values_uw = np.histogram(
            np.array(nodeWise_histo_dict['sig_train'], dtype=float),
            bins=bins
        )[0]
        yerrors = [(np.sqrt(uw)/uw)*w for uw, w in zip(values_uw, values)]
        plt.errorbar(
            bin_centers, values, yerr=yerrors, fmt='none', color='r', ec='r',
            lw=2
        )
        ###########################################
        values = plt.hist(
            nodeWise_histo_dict['bkg_train'],
            weights=nodeWise_histo_dict['bkg_train_w'], bins=bins,
            histtype='step', ec='b', ls='-', normed=True, label='BKG_train'
        )[0]
        values_uw = np.histogram(
            np.array(nodeWise_histo_dict['bkg_train'], dtype=float),
            bins=bins
        )[0]
        yerrors = [(np.sqrt(uw)/uw)*w for uw, w in zip(values_uw, values)]
        plt.errorbar(
            bin_centers, values, yerr=yerrors, fmt='none', color='b', ec='b',
            lw=2
        )
        ###########################################
        plt.legend(loc='upper center')
        plt.xlim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig(plot_out, bbox_inches='tight')
        plt.close('all')


def plot_nodeWise_roc(global_settings, roc_infos, mode):
    output_dir = global_settings['output_dir']
    colors = [('orange', 'r'), ('magenta', 'b'), ('k', 'g')]
    for color, roc_info in zip(colors, roc_infos):
        node = roc_info['node']
        plt.plot(
            roc_info['even_fpr_test'],
            roc_info['even_tpr_test'],
            lw=2, ls='--', color=color[0],
            label='node_' + str(node) + '_evenTrain_oddTest_' + \
            str('%0.3f' %roc_info['even_auc_test'])
        )
        plt.plot(
            roc_info['even_fpr_train'],
            roc_info['even_tpr_train'],
            lw=2, ls='-', color=color[0],
            label='node_' + str(node) + '_evenTrain_evenTest_' + \
            str('%0.3f' %roc_info['even_auc_train'])
        )
        plt.plot(
            roc_info['odd_fpr_test'],
            roc_info['odd_tpr_test'],
            lw=2, ls='--', color=color[1],
            label='node_' + str(node) + '_oddTrain_oddTest_' + \
            str('%0.3f' %roc_info['odd_auc_test'])
        )
        plt.plot(
            roc_info['odd_fpr_train'],
            roc_info['odd_tpr_train'],
            lw=2, ls='-', color=color[1],
            label='node_' + str(node) + '_oddTrain_evenTest_' + \
            str('%0.3f' %roc_info['odd_auc_train'])
        )
    plot_out = os.path.join(output_dir, 'nodeWiseROC_performance.png')
    plt.grid()
    plt.legend(loc='lower right')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')


def plot_feature_importances_from_dict(score_dict, output_dir):
    score_dict = OrderedDict(sorted(score_dict.items(), key=lambda x: -x[1]))
    plt.bar(range(len(score_dict)), score_dict.values(), align='center')
    plt.xticks(range(len(score_dict)), list(score_dict.keys()))
    file_name = os.path.join(output_dir, 'feature_importances.png')
    plt.xticks(rotation=90)
    plt.savefig(file_name, bbox_inches='tight')


def plot_trainvar_multi_distributions(data, trainvars, output_dir):
    plot_dir = os.path.join(output_dir, 'trainvar_distributions')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for trainvar in trainvars:
        trainvar_distribs = {}
        weight_distribs = {}
        all_data = data[trainvar]
        minimum_value = min(all_data)
        maximum_value = max(all_data)
        bins = np.linspace(minimum_value, maximum_value, 100)
        for process in set(data['process']):
            distrib = data.loc[data['process'] == process, trainvar]
            trainvar_distribs[process] = distrib
            weight_distribs[process] = data.loc[data['process'] == process, "totalWeight"]
        plot_single_distrib(trainvar_distribs, weight_distribs, plot_dir, trainvar, bins)


def plot_single_distrib(trainvar_distribs, weight_distribs, output_dir, trainvar, bins):
    keys = trainvar_distribs.keys()
    for key in keys:
        plt.hist(trainvar_distribs[key],  weights=weight_distribs[key], histtype='step', \
            label=key, bins=bins)
    plt.legend(loc='best', title=trainvar)
    plt.yscale('log')
    out_file = os.path.join(output_dir, trainvar + '_distribution.png')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close('all')


def plot_nn_sampleWise_bdtOutput(
        model_odd,
        data_even,
        preferences,
        global_settings,
        target=1,
        class_="",
        data_dict={},
        weight='totalWeight',
):
    output_dir = global_settings['output_dir']
    data_even = data_even.copy()
    if 'nonres' in global_settings['scenario']:
        sig_name = 'HH_nonres_decay'
    else:
        sig_name = 'signal'
    data_even.loc[
        data_even['process'].str.contains('signal'), ['process']] = sig_name  # 'signal'
    bkg_predictions = []
    bkg_labels = []
    bkg_weights = []
    bins = np.linspace(0., 1., 11)
    for process in set(data_even['process']):
        if process == sig_name:
            continue
        process_data = data_even.loc[data_even['process'] == process]
        idx = np.where(data_even['process'] == process)[0]
        process_prediction = np.array(model_odd.predict_proba(
            process_data[preferences['trainvars']]
        )[:, target]) if not global_settings["ml_method"] == 'lbn' else np.array(model_odd.predict(
            [data_dict["ll_even"][idx], data_dict["hl_even"][idx]], batch_size=1024
        )[:, target])
        weights = np.array(process_data[weight])
        bkg_weights.append(weights)
        bkg_predictions.append(process_prediction)
        bkg_labels.append(str(process))
    plt.hist(
        bkg_predictions, histtype='bar', label=bkg_labels, lw=2, bins=bins,
        weights=bkg_weights, alpha=1, stacked=True, normed=True
    )
    process_data = data_even.loc[data_even['process'] == sig_name]
    idx = np.where(data_even['process'] == sig_name)[0]
    process_prediction = np.array(model_odd.predict_proba(
        process_data[preferences['trainvars']]
    )[:, target]) if not global_settings["ml_method"] == 'lbn' else np.array(model_odd.predict(
        [data_dict["ll_even"][idx], data_dict["hl_even"][idx]], batch_size=1024
    )[:, target])
    weights = np.array(process_data['totalWeight'])
    plt.hist(
        process_prediction, histtype='step', label=sig_name,
        lw=2, ec='k', alpha=1, normed=True, bins=bins, weights=weights
    )
    plt.legend(loc='upper center')
    output_path = os.path.join(
        output_dir,
        'sampleWise_bdtOutput_node_%s.png' % class_
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.yscale('log')
    plt.close('all')

def  plot_loss_accuracy(fitted_model, output_dir, addition):
    fig1, ax = plt.subplots()
    epochs = range(1, len(fitted_model.history["loss"])+1)
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, fitted_model.history["loss"], "o-", label="Training")
    plt.plot(epochs, fitted_model.history["val_loss"], "o-", label="Validation")
    plt.xlabel("Epochs"), plt.ylabel("Loss")
    plt.ylim(0.0, 1.2*max(max(fitted_model.history["loss"]), max(fitted_model.history["val_loss"])))
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, fitted_model.history["accuracy"], "o-", label="Training")
    plt.plot(epochs, fitted_model.history["val_accuracy"], "o-", label="Validation")
    plt.xlabel("Epochs"), plt.ylabel("Accuracy")
    plt.ylim(0.0,1.0)
    plt.grid()
    plt.legend(loc="best");

    loss_vs_epoch = os.path.join(output_dir, "loss_vs_epoch_%s.png" \
                                 %(addition))
    plt.savefig(loss_vs_epoch)
    plt.close('all')

def plot_confusion_matrix(cm_original, class_names, output_dir, addition):
    figure = plt.figure(figsize=(4, 4))
    plt.imshow(cm_original, interpolation='nearest', cmap="summer")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=5, rotation=70)
    plt.yticks(tick_marks, class_names, fontsize=5)
    cm = np.moveaxis(
        np.around(
            cm_original.astype('float') / cm_original.sum(axis=1)[:, np.newaxis],
            decimals=2),
        0, 1
    )
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(i, j, cm[i, j], horizontalalignment="center", size=5)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    outfile = os.path.join(output_dir, 'confusion_matrix_rownorm_%s.png' %addition)
    plt.savefig(outfile, bbox_inches='tight')
    plt.close('all')

    figure = plt.figure(figsize=(4, 4))
    plt.imshow(cm_original, interpolation='nearest', cmap="summer")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=5, rotation=70)
    plt.yticks(tick_marks, class_names, fontsize=5)
    cm = np.moveaxis(
        np.around(
            cm_original.astype('float') / cm_original.sum(axis=0)[:, np.newaxis],
            decimals=2),
        0, 1
    )
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(i, j, cm[i, j], horizontalalignment="center", size=5)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    outfile = os.path.join(output_dir, 'confusion_matrix_columnnorm_%s.png' %addition)
    plt.savefig(outfile, bbox_inches='tight')
    plt.close('all')


def plot_DNNScore(data, output_dir, addition):
    color = ['b', 'g', 'y', 'r', 'magenta', 'orange', 'c']
    for node in sorted(set(data["multitarget"])):
        fig1, ax = plt.subplots()
        values = []
        weights = []
        labels = []
        colors = []
        for i, process in enumerate(set(data["process"])):
            if len(data.loc[((data["max_node_pos"] == node) & \
                (data["process"] == process))]):
                values.append(data.loc[((data["max_node_pos"] == node) & \
                    (data["process"] == process)), ["max_node_val"]].values.tolist())
                weights.append(data.loc[((data["max_node_pos"] == node) & \
                    (data["process"] == process)), ["evtWeight"]].values.tolist())
                labels.append('HH') if 'signal' in process else labels.append(process)
                colors.append(color[i])
        plt.hist(values, weights=weights,
            label=labels, color=colors,
            histtype='bar', stacked=True, range=(0, 1), bins=20)
        nodeName = list(set(data.loc[data["multitarget"] == node]["process"]))[0]
        plt.legend(loc='best', title=nodeName+"_node")
        plt.yscale('log')
        outfile = os.path.join(output_dir, 'DNNScore_'+nodeName+\
            '_node_'+addition+'.png')
        plt.savefig(outfile)
        plt.clf()

