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
    data_even.loc[data_even['process'].str.contains('signal'), ['process']] = 'signal'
    bkg_predictions = []
    bkg_labels = []
    bkg_weights = []
    bins = np.linspace(0., 1., 11)
    for process in set(data_even['process']):
        if process == 'signal':
            continue
        process_data = data_even.loc[data_even['process'] == process]
        process_DMatrix = xgb.DMatrix(
            process_data[preferences['trainvars']],
            nthread=global_settings['nthread'],
            feature_names=preferences['trainvars'],
        )
        process_prediction = model_odd.predict(process_DMatrix)
        bkg_predictions.append(process_prediction)
        bkg_labels.append(process)
        bkg_weights.append(np.array(process_data[weight]))
    plt.hist(
        bkg_predictions, histtype='bar', label=bkg_labels,
        lw=2, weights=bkg_weights, bins=bins,
        alpha=1, stacked=True, density=True
    )
    process_data = data_even.loc[data_even['process'] == process]
    process_DMatrix = xgb.DMatrix(
        process_data[preferences['trainvars']],
        nthread=global_settings['nthread'],
        feature_names=preferences['trainvars'],
    )
    process_prediction = model_odd.predict(process_DMatrix)
    plt.hist(
        process_prediction, histtype='step', label=str(process),
        lw=2, ec='k', weights=np.array(process_data['totalWeight']),
        alpha=1, density=True, bins=bins
    )
    plt.legend()
    output_path = os.path.join(output_dir, 'sampleWise_bdtOutput.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')


def plot_feature_importances(model, global_settings, addition):
    fig, ax = plt.subplots(figsize=(12, 18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plot_out = os.path.join(
        global_settings['output_dir'],
        addition,
        'feature_importances.png'
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
            even_info['fpr'], even_info['tpr'], ls=linestyle, color='g',
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
        mode_data = data.loc[data['target' == mode[1]]]
        plot_single_mode_correlation(mode_data, trainvars, output_dir, mode[0])
    plot_single_mode_correlation(data, trainvars, output_dir, 'total')


def plot_single_mode_correlation(data, trainvars, output_dir, addition):
    correlations = data[trainvars].corr()
    plt.matshow(correlations)
    ticks = np.arange(0, len(trainvars), 1)
    plt.rc('axes', labelsize=8)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xticklabels(trainvars, rotation=-90)
    plt.yticklabels(trainvars)
    plt.colorbar()
    plot_out = os.path.join(output_dir, str(addition) + '_correlations.png')
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')





def plot_nodeWise_performance(
        data, trainvars, prediction,
        label, global_settings, mode, savefig=False
):
    output_dir = global_settings['output_dir']
    plt.hist(
        prediction, histtype='step', weights=data['totalWeight'], label=label
    )
    if savefig:
        plot_out = os.path.join(output_dir, mode + '_nodeWisePredictions.png')
        plt.savefig(plot_out, bbox_inches='tight')
