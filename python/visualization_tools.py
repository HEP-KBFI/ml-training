''' Some helpful tools for plotting and data visualization
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os



def plot_sampleWise_bdtOutput(
        model_odd,
        data_even,
        preferences,
        global_settings,
        weight='totalWeight',
        standalone=True
):
    # output_dir = global_settings['output_dir']
    output_dir = '/home/laurits/'
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
    output_path = os.path.join(output_dir, 'sampleWise_bdtOutput.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')
