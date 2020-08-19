''' Some helpful tools for plotting and data visualization
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import xgboost as xgb


def plot_sampleWise_bdtOutput(
        model_odd,
        data_even,
        preferences,
        global_settings,
        weight='totalWeight',
):
    output_dir = global_settings['output_dir']
    data_even = data_even.copy()
    data_even.loc[
        data_even['process'].str.contains('signal'), ['process']] = 'signal'
    bkg_predictions = []
    bkg_labels = []
    bkg_weights = []
    bins = np.linspace(0., 1., 11)
    for process in set(data_even['process']):
        if process == 'signal':
            continue
        process_data = data_even.loc[data_even['process'] == process]
        process_prediction = np.array(model_odd.predict_proba(
            process_data[preferences['trainvars']]
        )[:,1])
        weights = np.array(process_data[weight])
        bkg_weights.append(weights)
        bkg_predictions.append(process_prediction)
        bkg_labels.append(str(process))
    plt.hist(
        bkg_predictions, histtype='bar', label=bkg_labels, lw=2, bins=bins,
        weights=bkg_weights, alpha=1, stacked=True, normed=True
    )
    process_data = data_even.loc[data_even['process'] == 'signal']
    process_prediction = np.array(model_odd.predict_proba(
        process_data[preferences['trainvars']]
    )[:,1])
    weights = np.array(process_data['totalWeight'])
    plt.hist(
        process_prediction, histtype='step', label='signal',
        lw=2, ec='k', alpha=1, normed=True, bins=bins, weights=weights
    )
    plt.legend()
    output_path = os.path.join(output_dir, 'sampleWise_bdtOutput.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')


def plot_feature_importances(model, global_settings, addition):
    fig, ax = plt.subplots()
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plot_out = os.path.join(
        global_settings['output_dir'],
        addition + '_feature_importances.png'
    )
    fig.savefig(plot_out, bbox_inces='tight')


def plotROC(odd_infos, even_infos, global_settings):
    output_dir = global_settings['output_dir']
    fig, ax = plt.subplots(figsize=(6, 6))
    linestyles = ['-', '--']
    for odd_info, linestyle in zip(odd_infos, linestyles):
        ax.plot(
            odd_info['fpr'], odd_info['tpr'], ls=linestyle, color='g',
            label='odd_' + odd_info['type'] + 'AUC = ' + str(odd_info['auc'])
        )
    for even_info, linestyle in zip(even_infos, linestyles):
        ax.plot(
            even_info['fpr'], even_info['tpr'], ls=linestyle, color='r',
            label='even_' + even_info['type'] + 'AUC = ' + str(even_info['auc'])
        )
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid()
    plot_out = os.path.join(output_dir, 'ROC_curve.png')
    fig.savefig(plot_out, bbox_inces='tight')
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
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
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
        plt.legend()
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
            label='node_' + str(node) + '_evenTrain_oddTest'
        )
        plt.plot(
            roc_info['even_fpr_train'],
            roc_info['even_tpr_train'],
            lw=2, ls='-', color=color[0],
            label='node_' + str(node) + '_evenTrain_evenTest'
        )
        plt.plot(
            roc_info['odd_fpr_test'],
            roc_info['odd_tpr_test'],
            lw=2, ls='--', color=color[1],
            label='node_' + str(node) + '_oddTrain_oddTest'
        )
        plt.plot(
            roc_info['odd_fpr_train'],
            roc_info['odd_tpr_train'],
            lw=2, ls='-', color=color[1],
            label='node_' + str(node) + '_oddTrain_evenTest'
        )
    plot_out = os.path.join(output_dir, 'nodeWiseROC_performance.png')
    plt.grid()
    plt.legend()
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')
