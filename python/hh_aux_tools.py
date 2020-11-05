from ROOT import TCanvas, TFile, TProfile
from ROOT import TH1D, THStack, TF1
from ROOT import gPad, TFitResultPtr
from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import universal_tools as ut
from sklearn.metrics import roc_curve, auc
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import xgboost as xgb
import math
import copy
import os
import ROOT
import matplotlib
import json
matplotlib.use('agg')
import matplotlib.pyplot as plt
ROOT.gROOT.SetBatch(True)


def normalize_hh_dataframe(
        data,
        preferences,
        global_settings,
        weight='totalWeight'
):
    '''Normalizes the weights for the HH data dataframe
    Parameters:
    ----------
    data : pandas Dataframe
        Dataframe containing all the data needed for the training.
    preferences : dict
        Preferences for the data choice and data manipulation
    global_settings : dict
        Preferences for the data, model creation and optimization
    [weight='totalWeight'] : str
        Type of weight to be normalized
    Returns:
    -------
    Nothing
    '''
    bdt_type = global_settings['bdtType']
    bkg_mass_rand = global_settings['bkg_mass_rand']
    ttbar_samples = ['TTToSemiLeptonic', 'TTTo2L2Nu']
    condition_sig = data['target'] == 1
    condition_bkg = data['target'] == 0
    if 'nonres' in bdt_type:
        data.loc[(data['target'] == 1), [weight]] *= 1./float(
           len(preferences['nonResScenarios']))
        data.loc[(data['target'] == 0), [weight]] *= 1./float(
           len(preferences['nonResScenarios']))
    elif 'oversampling' in bkg_mass_rand:
        data.loc[(data['target'] == 1), [weight]] *= 1./float(
            len(preferences['masses']))
        data.loc[(data['target'] == 0), [weight]] *= 1./float(
            len(preferences['masses']))
    if 'SUM_HH' in bdt_type:
        sample_normalizations = preferences['tauID_application']
        for sample in sample_normalizations.keys():
            sample_name = sample.replace('datacard', '')
            sample_weights = data.loc[data['process'] == sample_name, [weight]]
            sample_factor = sample_normalizations[sample]/sample_weights.sum()
            data.loc[data['process'] == sample, [weight]] *= sample_factor
        if 'nonres' in bdt_type:
            if global_settings["channel"] == "bb1l" :
                for node in range(len(preferences['nonResScenarios'])):
                    for process in set(data["process"]) :
                        condition_sig = data["process"].astype(str) == process
                        condition_node = data['nodeXname'].astype(str) == str(
                            preferences['nonResScenarios'][node])
                        node_sig_weight = data.loc[
                            condition_sig & condition_node, [weight]]
                        sig_node_factor = 100000./node_sig_weight.sum()
                        data.loc[
                            condition_sig & condition_node,
                            [weight]] *= sig_node_factor
            else :
                for node in range(len(preferences['nonResScenarios'])):
                    condition_node = data['nodeXname'].astype(str) == str(
                        preferences['nonResScenarios'][node])
                    node_sig_weight = data.loc[
                        condition_sig & condition_node, [weight]]
                    sig_node_factor = 100000./node_sig_weight.sum()
                    data.loc[
                        condition_sig & condition_node,
                        [weight]] *= sig_node_factor
                    node_bkg_weight = data.loc[
                        condition_bkg & condition_node, [weight]]
                    bkg_node_factor = 100000./node_bkg_weight.sum()
                    data.loc[
                        condition_bkg & condition_node,
                        [weight]] *= bkg_node_factor
        else:
            for mass in range(len(preferences['masses'])):
                condition_mass = data['gen_mHH'].astype(int) == int(
                    preferences['masses'][mass])
                mass_sig_weight = data.loc[
                    condition_sig & condition_mass, [weight]]
                sig_mass_factor = 100000./mass_sig_weight.sum()
                data.loc[
                    condition_sig & condition_mass,
                    [weight]] *= sig_mass_factor
                mass_bkg_weight = data.loc[
                    condition_bkg & condition_mass, [weight]]
                bkg_mass_factor = 100000./mass_bkg_weight.sum()
                data.loc[
                    condition_bkg & condition_mass,
                    [weight]] *= bkg_mass_factor
    else:
        sig_factor = 100000./data.loc[condition_sig, [weight]].sum()
        data.loc[condition_sig, [weight]] *= sig_factor
        bkg_factor = 100000./data.loc[condition_bkg, [weight]].sum()
        data.loc[condition_bkg, [weight]] *= bkg_factor


def load_hh_data(preferences, global_settings):
    data = dlt.load_data(
        preferences,
        global_settings
    )
    for trainvar in preferences['trainvars']:
        if str(data[trainvar].dtype) == 'object':
            try:
                data[trainvar] = data[trainvar].astype(int)
            except:
                continue
    if "nonres" not in global_settings['bdtType']:
        reweigh_dataframe(
            data,
            preferences['weight_dir'],
            preferences['trainvar_info'],
            ['gen_mHH'],
            preferences['masses'],
            preferences['trainvars']
        )
    normalize_hh_dataframe(data, preferences, global_settings)
    return data


def BkgLabelMaker(
        global_settings
):
    '''Makes Label for Bkg legend in plots
    Parameters:
    -----------
    global_settings : dict
                   Preferences for the data, model creation and optimization
    Returns
    ----------
    str
    '''
    labelBKG = ""
    bdtType = global_settings['bdtType']
    channel = global_settings['channel']
    if 'evtLevelSUM_HH_2l_2tau_res' in bdtType:
        labelBKG = "TT+DY+VV"
    elif '3l_1tau' in bdtType:
        labelBKG = "ZZ+WZ+TT+DY+ttZ+single higgs"
    elif 'evtLevelSUM' in bdtType:
        labelBKG = "SUM BKG"
        if channel in ["3l_0tau_HH"]:
            labelBKG = "WZ+tt+ttV"
    elif 'evtLevelWZ' in bdtType:
        labelBKG = "WZ"
    elif 'evtLevelDY' in bdtType:
        labelBKG = "DY"
    elif 'evtLevelTT' in bdtType:
        labelBKG = "TT"
    print('labelBKG: ', labelBKG)
    return labelBKG


def make_plots(
        featuresToPlot, nbin,
        data1, label1, color1,
        data2, label2, color2,
        plotname,
        printmin,
        plotResiduals,
        nodes_test=[],
        nodes_all=[],
        weights="totalWeight",
        mode="byNode"
):
    '''Makes plots of the Input Training Variables
    Parameters:
    -----------
    featuresToPlot : list
                  List of input training variables
    nbin : integer
        No of bins for the histograms
    data1 : pandas Dataframe
         Dataframe of training var.s for Bkg.
    label1 : str
          Label for the Bkg. Histogram legend
    color1 : str
          MatPlot lib label for the Bkg. Histogram color
    data2 : pandas Dataframe
         Dataframe of training var.s for Signal
    label2 : str
          Label for the Signal Histogram legend
    color2 : str
          MatPlot lib label for the Signal Histogram color
    plotname : str
            Name of the plot
    printmin : bool
            Handle to minimize verbosity of the plotting function
    plotResiduals : bool
                 Handle to plot Residuals = (Sig - Bkg)/Bkg
    nodes_test : list
          list of input signal test nodes to plot
    nodes_all : list
          list of all input signal nodes (mass or BM)
          (Needed to plot gen_mHH/nodeX histo.)
    mode: string
          evaluation mode to distinguish between Res training featuring
          gen_mHH and nonRes training featuring nodeX
    Returns:
    -------
    Plot of input training variables in .pdf format
    '''
    print('length of features to plot and features to plot',
          (len(featuresToPlot), featuresToPlot))
    hist_params = {
        'normed': True,
        'histtype': 'bar',
        'fill': True,
        'lw': 3,
        'alpha': 0.4
    }
    sizeArray = 0
    if(math.sqrt(len(featuresToPlot)) %
       int(math.sqrt(len(featuresToPlot))) == 0):
        sizeArray = int(math.sqrt(len(featuresToPlot)))
    else:
        sizeArray = int(math.sqrt(len(featuresToPlot)))+1
    drawStatErr = True
    residuals = []
    plt.figure(figsize=(5*sizeArray, 5*sizeArray))
    to_ymax = 10.
    to_ymin = 0.0001
    for n, feature in enumerate(featuresToPlot):
        # add sub plot on our figure
        plt.subplot(sizeArray, sizeArray, n+1)
        # define range for histograms by cutting 1% of data from both ends
        min_value, max_value = np.percentile(data1[feature], [0.0, 99])
        min_value2, max_value2 = np.percentile(data2[feature], [0.0, 99])
        print('----------------------')
        print('Feature: %s' % (feature))
        print('min_value: %s' % (min_value))
        print('max_value: %s' % (max_value))
        print('min_value2: %s' % (min_value2))
        print('max_value2: %s' % (max_value2))
        if feature == "gen_mHH":
            nbin_local = 10*len(nodes_all)
            range_local = [
                nodes_all[0]-20,
                nodes_all[len(nodes_all)-1]+20
            ]
        else:
            nbin_local = nbin
            range_local = (
                min(min_value, min_value2),
                max(max_value, max_value2)
            )
        if printmin:
            print('printing min and max value for feature: ', feature)
            print('min_value: ', min_value)
            print('max_value: ', max_value)
        values1, bins, _ = plt.hist(
                                    data1[feature].values.astype(float),
                                    weights=data1[weights].values.astype(
                                        np.float64),
                                    range=range_local,
                                    bins=nbin_local,
                                    edgecolor=color1,
                                    color=color1,
                                    label=label1,
                                    **hist_params
        )
        to_ymax = max(values1)
        to_ymin = min(values1)
        if drawStatErr:
            normed = sum(data1[feature].values)
            mid = 0.5*(bins[1:] + bins[:-1])
            err = np.sqrt(values1*normed)/normed  # deno. as plot is norm.
            plt.errorbar(
                mid, values1, yerr=err, fmt='none', color=color1,
                ecolor=color1, edgecolor=color1, lw=2
            )
        if len(nodes_test) == 0:  # 'gen_mHH' not in feature
            values2, bins, _ = plt.hist(
                data2[feature].values.astype(float),
                weights=data2[weights].values.astype(
                    np.float64),
                range=range_local,
                bins=nbin_local,
                edgecolor=color2,
                color=color2,
                label=label2,
                **hist_params
            )
            to_ymax2 = max(values2)
            to_ymax = max([to_ymax2, to_ymax])
            to_ymin2 = min(values2)
            to_ymin = max([to_ymin2, to_ymin])
            if drawStatErr:
                normed = sum(data2[feature].values)
                mid = 0.5*(bins[1:] + bins[:-1])
                err = np.sqrt(values2*normed)/normed  # deno. as plot is norm.
        else:
            hist_params2 = {
                'normed': True,
                'histtype': 'step',
                'fill': False,
                'lw': 3
            }
            colors_node = [
                'm', 'b', 'k', 'r', 'g',  'y', 'c',
                'chocolate', 'teal', 'pink', 'darkkhaki',
                'maroon', 'slategray', 'orange', 'silver',
                'aquamarine', 'lavender', 'goldenrod', 'salmon',
                'tan', 'lime', 'lightcoral'
            ]
            for nn, node in enumerate(nodes_test):
                plot_features = []
                plot_weights = []
                plot_label = ''
                if mode == 'byMass':
                    gen_mHH_mass = data2["gen_mHH"].astype(np.int)
                    plot_features = data2.loc[
                        (data2["gen_mHH"].astype(np.int) == int(node)),
                        feature].values
                    plot_weights = data2.loc[
                        (data2["gen_mHH"].astype(np.int) == int(node)),
                        weights].values
                    plot_label = label2 + "gen_mHH = " + str(node)
                elif mode == 'byNode':
                    plot_features = data2.loc[
                        (data2['nodeXname'].astype(str) == str(node)),
                        feature].values
                    plot_weights = data2.loc[
                        (data2['nodeXname'].astype(str) == str(node)),
                        weights].values
                    plot_label = label2 + "node = " + str(node)
                else:
                    raise ValueError(
                        'Please use a valid mode!')
                plot_wt_float = plot_weights.astype(np.float64)
                values2, bins, _ = plt.hist(
                                       plot_features.astype(float),
                                       weights=plot_wt_float,
                                       range=range_local,
                                       bins=nbin_local,
                                       edgecolor=colors_node[nn],
                                       color=colors_node[nn],
                                       label=plot_label,
                                       **hist_params
                )
                to_ymax2 = max(values2)
                to_ymax = max([to_ymax2, to_ymax])
                to_ymin2 = min(values2)
                to_ymin = max([to_ymin2, to_ymin])
                if drawStatErr:
                    normed = sum(data2[feature].values)
                    mid = 0.5*(bins[1:] + bins[:-1])
                    err = np.sqrt(values2*normed)/normed  # deno. as norm. plot
                    plt.errorbar(
                        mid, values2, yerr=err, fmt='none',
                        color=colors_node[nn], ecolor=colors_node[nn],
                        edgecolor=colors_node[nn], lw=2
                    )
        if(plotResiduals):
            residuals = residuals + [(plot1[0] - plot2[0])/(plot1[0])]
        plt.ylim(ymin=to_ymin*0.1, ymax=to_ymax*1.2)
        if(feature == "avg_dr_jet"):
            plt.yscale('log')
        else:
            plt.yscale('linear')
        if(n == (len(featuresToPlot) - 1)):
            plt.legend(loc='best')
        plt.xlabel(feature)
        # plt.xscale('log')
    plt.ylim(ymin=0)
    plt.savefig(plotname)
    plt.clf()
    if plotResiduals:
        residuals = np.nan_to_num(residuals)
        for n, feature in enumerate(trainVars(True)):
            (mu, sigma) = norm.fit(residualsSignal[n])
            plt.subplot(8, 8, n+1)
            residualsSignal[n] = np.nan_to_num(residualsSignal[n])
            n, bins, patches = plt.hist(residualsSignal[n],
                                        label='Residuals ' +
                                        label1 + '/' + label2)
            # add a 'best fit' line
            y = mlab.normpdf(bins, mu, sigma)
            # l = plt.plot(bins, y, 'r--', linewidth=2)
            plt.ylim(ymin=0)
            plt.title(feature + ' ' + r'mu=%.3f, sig=%.3f$' % (mu, sigma))
            print(feature + ' ' + r'mu=%.3f, sig=%.3f$' % (mu, sigma))
        plt.savefig(
            channel + "/" + bdtType + "_" + trainvar +
            "_Variables_Signal_fullsim_residuals.pdf"
        )
        plt.clf()


def num_to_str(
        num
):
    '''Converting floats to strings
    Parameters:
    -----------
    num : input floating point number
    Returns:
    --------
    string
    '''
    temp_str = str(num)
    final_str = temp_str.replace('.', 'o')
    return final_str


def numpyarrayHisto1DFill(
        arr,
        weight,
        histo1D
):
    '''Fills 1-Dim. Histogram with numpy arrays
    Parameters:
    -----------
    arr : numpy array
        quantity to be plotted
    weight : numpy array
        event level weight to be applied while plotting
    histo1D : TH1D Histogram
       histogram to be filled
    Returns:
    --------
    Nothing
    '''
    for x, w in zip(arr, weight):
        # print("x: {},  w: {}".format(x, w))
        histo1D.Fill(x, w)


def AddHistToStack(
        data,
        var_name,
        hstack,
        nbins,
        X_min,
        X_max,
        FillColor,
        processName,
        weights="totalWeight"
):
    '''Function for Adding histograms to THStack
    Parameters:
    -----------
    data : pandas dataframe
        Input dataframe
    var_name : str
            Label of the dataframe column/Variable to plot
    hstack : THStack object
          Histogram stack to be updated
    nbins : int
         No. of bins of histogram to be added to the Stack
    X_min : float
         Min. value of X-axis of histogram to be added to the Stack
    X_max : float
         Max. value of X-axis of histogram to be added to the Stack
    FillColor : int
             ROOT Color index of histogram to be added to the Stack
    processName : str
               Name of process whose histogram is being added to the Stack
    weights : str
           Name of dataframe column used for weighing all the Stack histograms
    Returns:
    --------
    Nothing
    '''
    histo1D = TH1D('histo1D', processName, nbins, X_min, X_max)
    data_X_array = np.array(data[var_name].values, dtype=np.float)
    data_wt_array = np.array(data[weights].values, dtype=np.float)
    numpyarrayHisto1DFill(data_X_array, data_wt_array, histo1D)
    histo1D.SetFillColor(FillColor)
    hstack.Add(histo1D)


def BuildTHstack(
        channel,
        hstack,
        data,
        var_name,
        nbins,
        X_min,
        X_max,
        weights="totalWeight"
):
    '''Building the channel specific THStack
    Parameters:
    -----------
    channel : str
           Channel label
    hstack : THStack object
          Histogram stack to be built
    data : pandas dataframe
        Input dataframe
    var_name : str
            Label of the dataframe column/Variable to plot
    nbins : int
         No. of bins of histogram to be added to the Stack
    X_min : float
         Min. value of X-axis of histogram to be added to the Stack
    X_max : float
         Max. value of X-axis of histogram to be added to the Stack
    weights : str
           Name of dataframe column used for weighing all the Stack histograms
    Returns:
    --------
    Nothing
    '''
    if(channel == "2l_2tau"):
        ttbar_samples = ['TTTo2L2Nu', 'TTToSemiLeptonic']
        vv_samples = ['ZZ', 'WZ', 'WW']
        ttv_samples = ['TTZJets', 'TTWJets']
        data_copy_TT = data.loc[
            (data['key'].isin(ttbar_samples))]  # TTbar
        data_copy_DY = data.loc[
            (data['key'] == 'DY')]  # DY
        data_copy_VV = data.loc[
            (data['key'].isin(vv_samples))]  # VV
        data_copy_TTV = data.loc[
            (data['key'].isin(ttv_samples))]  # TTV
        data_copy_TTH = data.loc[
            (data['key'] == 'TTH')]  # TTH
        data_copy_VH = data.loc[
            (data['key'] == 'VH')]  # VH

        if not(data_copy_TTH.empty):
            AddHistToStack(
                data_copy_TTH, var_name,
                hstack, nbins,
                X_min, X_max,
                5, 'TTH',
                weights
            )  # Yellow
        if not(data_copy_TTV.empty):
            AddHistToStack(
                data_copy_TTV, var_name,
                hstack, nbins,
                X_min, X_max,
                1, 'TTV',
                weights
            )  # Black
        if not(data_copy_VH.empty):
            AddHistToStack(
                data_copy_VH, var_name,
                hstack, nbins,
                X_min, X_max,
                6, 'VH',
                weights
            )  # Magenta
        if not(data_copy_VV.empty):
            AddHistToStack(
                data_copy_VV, var_name,
                hstack, nbins,
                X_min, X_max,
                3, 'VV',
                weights
            )  # Green
        if not(data_copy_DY.empty):
            AddHistToStack(
                data_copy_DY, var_name,
                hstack, nbins,
                X_min, X_max,
                2, 'DY',
                weights
            )  # Red
        if not(data_copy_TT.empty):
            AddHistToStack(
                data_copy_TT, var_name,
                hstack, nbins,
                X_min, X_max,
                4, 'TTbar',
                weights
            )  # TT
    if(channel == "0l_4tau" or channel == "0l_4tau_nonRes"):
        data_copy_ZZ = data.loc[
            (data['key'] == 'ZZ')]  # ZZ
        data_copy_WZ = data.loc[
            (data['key'] == 'WZ')]  # WZ
        data_copy_TT = data.loc[
            (data['key'] == 'TTT')]  # TTbar
        data_copy_DY = data.loc[
            (data['key'] == 'DY')]  # DY
        data_copy_ttH = data.loc[
            (data['key'] == 'ttH')]
        data_copy_VH = data.loc[
            (data['key'] == 'VH')]
        data_copy_TTZ = data.loc[
            (data['key'] == 'TTZJets')]
        if not(data_copy_DY.empty):
            AddHistToStack(
                data_copy_DY, var_name,
                hstack, nbins,
                X_min, X_max,
                2, 'DY',
                weights
            )  # Red
        if not(data_copy_TT.empty):
            AddHistToStack(
                data_copy_TT, var_name,
                hstack, nbins,
                X_min, X_max,
                4, 'TTbar',
                weights
            )  # TT
        if not(data_copy_ZZ.empty):
            AddHistToStack(
                data_copy_ZZ, var_name,
                hstack, nbins,
                X_min, X_max,
                4, 'ZZ',
                weights
            )  # Blue
        if not(data_copy_WZ.empty):
            AddHistToStack(
                data_copy_WZ, var_name,
                hstack, nbins,
                X_min, X_max,
                2, 'WZ',
                weights
            )  # Red
        if not(data_copy_ttH.empty):
            AddHistToStack(
                data_copy_ttH, var_name,
                hstack, nbins,
                X_min, X_max,
                2, 'ttH',
                weights
            )  # Red
        if not(data_copy_VH.empty):
            AddHistToStack(
                data_copy_VH, var_name,
                hstack, nbins,
                X_min, X_max,
                2, 'VH',
                weights
            )  # Red
        if not(data_copy_TTZ.empty):
            AddHistToStack(
                data_copy_TTZ, var_name,
                hstack, nbins,
                X_min, X_max,
                2, 'TTZ',
                weights
            )  # Red
    if(channel == "3l_1tau" or channel == "3l_1tau_nonRes"):
        zz_samples = ['ZZ']
        data_copy_ZZ = data.loc[
            (data['key'].isin(zz_samples))]  # ZZ
        data_copy_WZ = data.loc[
            (data['key'] == 'WZ')]  # WZ
        ttbar_samples = ['TTTo']
        data_copy_TT = data.loc[
            (data['key'].isin(ttbar_samples))]  # TTbar
        data_copy_DY = data.loc[
            (data['key'] == 'DY')]  # DY
        singleHiggs_samples = ["VH","qqH","ggH","ttH"]
        data_copy_singleHiggs = data.loc[
            (data['key'].isin(singleHiggs_samples))]  # DY
        data_copy_TTZ = data.loc[
            (data['key'] == 'TTZ')]
        if not(data_copy_DY.empty):
            AddHistToStack(
                data_copy_DY, var_name,
                hstack, nbins,
                X_min, X_max,
                2, 'DY',
                weights
            )  # Red
        if not(data_copy_TT.empty):
            AddHistToStack(
                data_copy_TT, var_name,
                hstack, nbins,
                X_min, X_max,
                4, 'TTbar',
                weights
            )  # TT
        if not(data_copy_ZZ.empty):
            AddHistToStack(
                data_copy_ZZ, var_name,
                hstack, nbins,
                X_min, X_max,
                4, 'ZZ',
                weights
            )  # Blue
        if not(data_copy_WZ.empty):
            AddHistToStack(
                data_copy_WZ, var_name,
                hstack, nbins,
                X_min, X_max,
                2, 'WZ',
                weights
            )  # Red
        if not(data_copy_TTZ.empty):
            AddHistToStack(
                data_copy_ZZ, var_name,
                hstack, nbins,
                X_min, X_max,
                4, 'TTZ',
                weights
            )  # Blue
        if not(data_copy_singleHiggs.empty):
            AddHistToStack(
                data_copy_WZ, var_name,
                hstack, nbins,
                X_min, X_max,
                2, 'single higgs',
                weights
            )  # Red
    else:
        print('Please implement settings for your own channel')


def MakeHisto(
        output_dir,
        channel,
        data,
        var_name_list,
        histo_dicts,
        label="",
        weights="totalWeight"
):
    '''Makes Histograms of list of Variables in .root format
    Parameters:
    -----------
    output_dir : str
           Path to the output directory
    channel : str
           Label denoting the channel
    data : pandas Dataframe
        Dataframe of training var.s for Bkg.
    var_name_list : list
                 List of strings of Dataframe column names to plot
    histo_dicts : list
               List of python dicts containing info about plot settings
    label : str
         Label to append to output histogram file names
    weights : str
           Name of dataframe column used for weighing the histograms
    Returns:
    -------
    Nothing
    '''
    assert os.path.isdir(output_dir), "Directory doesn't exist"

    data_copy = data.copy(deep=True)  # Making a deep copy of dataframe
    data_copy = data_copy.loc[(data_copy['target'] == 0)]  # df backgrounds
    for var_name in var_name_list:
        print('Variable Name: {}'.format(var_name))
        data_X = np.array(data_copy[var_name].values, dtype=np.float)
        data_wt = np.array(data_copy[weights].values, dtype=np.float)

        N_x = len(data_X)
        N_wt = len(data_wt)

        if(N_x == N_wt):
            print('Plotting Histogram: {}'.format(var_name))

            # Create a new canvas, and customize it.
            c1 = TCanvas('c1', 'Histogram', 200, 10, 700, 500)
            # c1.SetFillColor(42)
            # c1.GetFrame().SetFillColor(21)
            c1.GetFrame().SetBorderSize(6)
            c1.GetFrame().SetBorderMode(-1)

            PlotTitle = var_name
            Histo_Dict = dlt.find_correct_dict(
                'Variable', str(var_name), histo_dicts)
            if Histo_Dict == {}:
                continue
            print('Histo_Dict :', Histo_Dict)
            histo1D = TH1D(
                'histo1D', PlotTitle,
                Histo_Dict["nbins"],
                Histo_Dict["min"],
                Histo_Dict["max"]
            )
            histo1D.GetYaxis().SetTitle("Events")
            histo1D.GetXaxis().SetTitle(str(var_name))
            numpyarrayHisto1DFill(data_X, data_wt, histo1D)
            histo1D.Draw()
            c1.Modified()
            c1.Update()
            FileName = "{}/{}_{}_{}_{}.root".format(
                output_dir, channel,
                "Histo1D", str(var_name), label
            )
            c1.SaveAs(FileName)
        else:
            print('Arrays not of same length')
            print('N_x: {}, N_wt: {}'.format(N_x, N_wt))


def MakeTHStack(
        output_dir,
        channel,
        data,
        var_name_list,
        histo_dicts,
        label="",
        weights="totalWeight"
):
    '''Makes Stack plots of list of Variables in .root format
    Parameters:
    -----------
    output_dir : str
           Path to the output directory
    channel : str
           Label denoting the channel
    data : pandas Dataframe
        Dataframe of training var.s for Bkg.
    var_name_list : list
                 List of strings of Dataframe column names to plot
    histo_dicts : list
               List of python dicts containing info about plot settings
    label : str
         Label to append to output .root file name
    weights : str
           Name of dataframe column used for weighing all the Stack histograms
    Returns:
    -------
    Nothing
    '''
    assert os.path.isdir(output_dir), "Directory doesn't exist"

    data_copy = data.copy(deep=True)  # Making a deep copy of dataframe
    data_copy = data_copy.loc[(data_copy['target'] == 0)]  # df backgrounds

    for var_name in var_name_list:
        print('Variable Name: {}'.format(var_name))
        data_X = np.array(data_copy[var_name].values, dtype=np.float)
        data_wt = np.array(data_copy[weights].values, dtype=np.float)

        N_x = len(data_X)
        N_wt = len(data_wt)

        if(N_x == N_wt):
            print('Plotting Histogram: {}'.format(var_name))

            # Create a new canvas, and customize it.
            c1 = TCanvas('c1', 'Stack plot', 200, 10, 700, 500)
            # c1.SetFillColor(42)
            # c1.GetFrame().SetFillColor(21)
            c1.GetFrame().SetBorderSize(6)
            c1.GetFrame().SetBorderMode(-1)
            PlotTitle = var_name
            hstack = THStack('hstack', PlotTitle)
            Histo_Dict = dlt.find_correct_dict(
                'Variable', str(var_name), histo_dicts)
            BuildTHstack(
                channel,
                hstack,
                data_copy,
                var_name,
                Histo_Dict["nbins"],
                Histo_Dict["min"],
                Histo_Dict["max"],
                weights
            )
            hstack.Draw("hist")
            hstack.GetYaxis().SetTitle("Events")
            hstack.GetXaxis().SetTitle(var_name)
            c1.Modified()
            c1.Update()
            gPad.BuildLegend(0.75, 0.75, 0.95, 0.95, "")
            FileName = "{}/{}_{}_{}_{}.root".format(
                output_dir,
                channel,
                "THStack",
                str(var_name),
                label
            )
            c1.SaveAs(FileName)
        else:
            print('Arrays not of same length')
            print('N_x: {}, N_wt: {}'.format(N_x, N_wt))


def PlotFeaturesImportance(
        output_dir,
        global_settings,
        preferences,
        model,
        label=""
):
    '''Plot Importance of Input Variables
    Parameters:
    -----------
    output_dir : str
             Path to the output directory
    channel : str
             Label for the channel
    model : XGB Booster
        Booster obtained by training on the train dMatrix
    label : str
         Label for the output plot name
    Returns:
    -----------
    Nothing
    '''
    channel = global_settings['channel']
    nameout = "{}/{}_{}_InputVar_Importance.pdf".format(
        output_dir, channel, label)
    if 'nonres' in global_settings['bdtType']:
        booster = model.get_booster()
        feature_importances = booster.feature_importances()
        keys = feature_importances.keys()
        nonResScenarios = preferences['nonResScenarios']
        elements_BM = 0
        for nonRes_scenario in preferences['nonResScenarios']:
            elements_BM += feature_importances[nonRes_scenario]
        feature_importances['sumBM'] = sumBM
        for key in feature_immportances.keys():
            if key in nonResScenarios:
                feature_importances.pop(key)
        # ORDER THE DICT!!
        plt.bar(
            range(len(feature_importances)),
            list(feature_importances.values()),
            align='center'
        )
        plt.xticks(
            range(len(feature_importances)),
            list(feature_importances.keys())
        )
        plt.savefig(nameout, bbox_inces='tight')
    else:
        xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
        fig, ax = plt.subplots(figsize=(12, 18))
        fig.savefig(nameout)


def getPred(data, trainvars, nthread, targetName, weightsName, estimator):
    dMatrix_pred = PDfToDMatConverter(
        data,
        trainvars,
        nthread,
        target=targetName,
        weights=weightsName
    )
    return estimator.predict(dMatrix_pred)


def PlotROC(
        output_dir,
        channel,
        roc_train,
        roc_test,
        label=""
):
    '''Plot ROC curve for the overall training
    Parameters:
    -----------
    output_dir : str
             Path to the output directory
    channel : str
             Label for the channel
    roc_train : list
             List containing info like FPR,TPR and AUC for train dataset
    roc_test : list
             List containing info like FPR,TPR and AUC for test dataset
    label : str
         Label for the output plot name
    Returns:
    -----------
    Nothing
    '''
    PlotLabel = label
    styleline = ['-', '--']
    colorline = ['g', 'r']
    fig, ax = plt.subplots(figsize=(6, 6))
    for tt, rocs in enumerate(roc_test):
        ax.plot(
            roc_train[tt]['fpr'], roc_train[tt]['tpr'],
            color=colorline[tt], lw=2, linestyle='-',
            label=PlotLabel + ' train (area = %0.3f)' %
            (roc_train[tt]['train_auc'])
        )
        ax.plot(
            roc_test[tt]['fprt'], roc_test[tt]['tprt'],
            color=colorline[tt], lw=2, linestyle='--',
            label=PlotLabel + ' test (area = %0.3f)' %
            (roc_test[tt]['test_auc'])
        )
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid()
    nameout = "{}/{}_{}_ROC.pdf".format(output_dir, channel, PlotLabel)
    fig.savefig(nameout)


def PlotClassifier(
        output_dir,
        global_settings,
        model,
        train,
        test,
        trainvars,
        label='',
        target='target',
        weights='totalWeight'
):
    '''Makes plot for Classifier output for train and test data
    Parameters:
    -----------
    output_dir : str
             Path to the output directory
    global_settings : dict
              Preferences for the data, model creation and optimization
    model : XGB Booster
        Booster object obtained by training on the train dMatrix
    train : pandas dataframe
         Training dataset
    test : pandas dataframe
        Testing dataset
    trainvars : list
             List of names of training variables
    label : str
         Label for the output plot name
    target: string
           name of the target column in the data frame
    weights: string
           name of the weights column in the data frame

    Returns:
    -----------
    Nothing
    '''
    labelBKG = BkgLabelMaker(global_settings)

    channel = global_settings["channel"]
    bdtType = global_settings["bdtType"]
    nthread = global_settings["nthread"]

    hist_params = {'normed': True, 'bins': 10,
                   'histtype': 'step', 'lw': 2, 'range': [0, 1]}

    pred_test_BKG = getPred(test.loc[(test[target].values == 0)],
                            trainvars, nthread, target, weights, model)
    weights_test_BKG = test.loc[(test[target].values == 0), [weights]]

    pred_test_SIG = getPred(test.loc[(test[target].values == 1)],
                            trainvars, nthread, target, weights, model)
    weights_test_SIG = test.loc[(test[target].values == 1), [weights]]

    pred_train_BKG = getPred(train.loc[(train[target].values == 0)],
                             trainvars, nthread, target, weights, model)
    weights_train_BKG = train.loc[(train[target].values == 0), [weights]]

    pred_train_SIG = getPred(train.loc[(train[target].values == 1)],
                             trainvars, nthread, target, weights, model)
    weights_train_SIG = train.loc[(train[target].values == 1), [weights]]

    fig, ax = plt.subplots(figsize=(6, 6))
    PlotLabel_BKG_train = labelBKG+' (train)'
    PlotLabel_SIG_train = 'signal (train)'
    PlotLabel_BKG_test = labelBKG+' (test)'
    PlotLabel_SIG_test = 'signal (test)'

    dict_plot = [
        [pred_train_BKG, weights_train_BKG, "-", 'g',
         PlotLabel_BKG_train],
        [pred_train_SIG, weights_train_SIG, "-", 'r',
         PlotLabel_SIG_train],
        [pred_test_BKG, weights_test_BKG, "-", 'b',
         PlotLabel_BKG_test],
        [pred_test_SIG, weights_test_SIG, "-", 'magenta',
         PlotLabel_SIG_test],
    ]
    for item in dict_plot:
        values, bins, _ = ax.hist(
            np.array(item[0], dtype=float),
            weights=np.array(item[1], dtype=float),
            ls=item[2], color=item[3],
            label=item[4],
            **hist_params
        )
        #  create unweighted and non normalized hist
        #  to calculate proper per bin errors
        values_unweighted, bins_unweighted = np.histogram(np.array(item[0]),
                                                          bins=bins)
        yerrs = []
        for vv, value in enumerate(values_unweighted):
            if value > 0:
                bin_err = (math.sqrt(value)/value)*values[vv]
            else:
                bin_err = 0
            yerrs.append(bin_err)
        mid = 0.5*(bins[1:] + bins[:-1])
        plt.errorbar(
            mid, values,
            yerr=yerrs, fmt='none',
            color=item[3], ecolor=item[3],
            edgecolor=item[3], lw=2
        )
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=2, fancybox=True, shadow=True)
    ax.set_xlabel('classifier output')
    ax.yaxis.set_label_text('normalized entries')
    plt.savefig(
        output_dir + '/' + channel + '_' + bdtType +
        '_InputVars_' + str(len(trainvars)) + '_' +
        label + '_XGBclassifier.pdf'
    )
    plt.close()


def PlotCorrelation(
        output_dir,
        global_settings,
        data,
        trainvars,
        label=""
):
    '''Makes plots for Input Var. Correl.s for both
       signal and background for given dataframe
    Parameters:
    -----------
    output_dir : str
             Path to the output directory
    global_settings : dict
                  Preferences for the data, model creation and optimization
    data : pandas dataframe
         Input dataframe
    trainvars : list
             List of names of training variables
    label : str
         Label for the output plot name
    Returns:
    -----------
    Nothing
    '''
    channel = global_settings["channel"]
    bdtType = global_settings["bdtType"]
    PlotLabel = label
    for ii in [1, 2]:
        if ii == 1:
            datad = data.loc[data['target'].values == 1]
            proc_label = "signal"
        else:
            datad = data.loc[data['target'].values == 0]
            proc_label = "BKG"
        datacorr = datad[trainvars].astype(float)
        correlations = datacorr.corr()
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
        plt.savefig("{}/{}_{}_InputVars{}_corr_{}_{}.pdf".format(
            output_dir, channel,
            bdtType, str(len(trainvars)),
            proc_label, PlotLabel)
        )


def PlotROCByX(
        output_dir,
        global_settings,
        preferences,
        model_list,
        df_list,
        trainvars,
        label_list=['', ''],
        weights='totalWeight',
        target='target',
        mode='byNode'
):
    '''Make ROC curves as function of gen_mHH for test masses
    Parameters:
    -----------
    output_dir : str
             Path to the output directory
    global_settings : dict
                   Preferences for the data, model creation and optimization
    preferences : dict
               Preferences for the data choice and data manipulation
    model_list : list
            List of XGB Boosters obtained by training the Odd-Even dMatrices
            pair and vice-versa
    df_list : list
           List of pandas dataframes for evaluation
    trainvars : list
             List of names of training variables
    label_list : list
              List of labels for the output plot name
    weights: string
           name of the weights column in the data frame
    target: string
           name of the target column in the data frame
    mode: string
          evaluation mode to distinguish between Res training featuring
          gen_mHH and nonRes training featuring nodeX
    Returns:
    -----------
    Nothing
    '''
    bdtType = global_settings['bdtType']
    channel = global_settings['channel']
    nthread = global_settings['nthread']
    nodes = []
    if mode == 'byMass':
        nodes = preferences["masses_test"]
        if('nonres' in global_settings['bdtType']):
            raise ValueError(
                'Please use this mode only with Res!')
    elif mode == 'byNode':
        nodes = preferences['nonResScenarios_test']
        if('nonres' not in global_settings['bdtType']):
            raise ValueError(
                'Please use this mode only with nonRes!')
    else:
        raise ValueError(
            'Please use a valid mode!')

    styleline = ['-', '--', '-.', ':']
    colors_nodes = ['m', 'b', 'k', 'r', 'g',  'y', 'c', ]
    fig, ax = plt.subplots(figsize=(12, 6))
    sl = 0
    for nn, node in enumerate(nodes):
        for dd in range(len(df_list)):
            val_idx = int(abs(dd-1))
            if mode == 'byMass':
                val_data = df_list[val_idx].loc[
                    df_list[val_idx]["gen_mHH"].astype(np.int) == int(node)]
                train_data = df_list[dd].loc[
                    df_list[dd]["gen_mHH"].astype(np.int) == int(node)]
            else:
                val_data = df_list[val_idx].loc[
                    (df_list[val_idx]['nodeXname'].astype(str) == str(node))]
                train_data = df_list[dd].loc[
                    (df_list[dd]['nodeXname'].astype(str) == str(node))]

            pred_train = getPred(train_data, trainvars,
                                 nthread, target, weights, model_list[dd])
            weights_train = train_data[weights].astype(np.float64)
            targets_train = train_data[target].astype(np.bool)

            fpr_train, tpr_train, thresholds_train = roc_curve(
                targets_train, pred_train, sample_weight=weights_train
            )
            train_auc = auc(fpr_train, tpr_train, reorder=True)

            if mode == 'byMass':
                print('train set auc ' + str(train_auc) +
                      ' (mass = ' + str(node) + ')')
            else:
                print('train set auc ' + str(train_auc) +
                      ' (node = ' + str(node) + ")")

            pred_val = getPred(val_data, trainvars,
                               nthread, target, weights, model_list[dd])
            weights_val = val_data[weights].astype(np.float64)
            targets_val = val_data[target].astype(np.bool)

            fpr_val, tpr_val, thresholds_val = roc_curve(
                targets_val, pred_val, sample_weight=weights_val
            )
            val_auc = auc(fpr_val, tpr_val, reorder=True)

            if mode == 'byMass':
                print('val set auc ' + str(val_auc) +
                      ' (mass = ' + str(node) + ")")
            else:
                print('val set auc ' + str(val_auc) +
                      ' (node = ' + str(node) + ')')
            label_train, label_val = ['', '']

            if mode == 'byMass':
                label_train = label_list[dd] + ' training: ' + label_list[dd] + ' events (area = %0.3f)' % (train_auc) + ' (mass = ' + str(node) + ')'
                label_val = label_list[dd] + ' training: ' + label_list[val_idx] + ' events (area = %0.3f)' % (val_auc) + ' (mass = ' + str(node) + ')'
            else:
                label_train = label_list[dd] + ' training: ' + label_list[dd] + ' events (area = %0.3f)' % (train_auc) + ' (node = ' + str(node) + ')'
                label_val = label_list[dd] + ' training: ' + label_list[val_idx] + ' events (area = %0.3f)' % (val_auc) + ' (node = ' + str(node) + ')'
            ax.plot(
                fpr_train, tpr_train,
                lw=2, linestyle=styleline[dd + dd*1],
                color=colors_nodes[nn],
                label=label_train
            )
            sl += 1
            ax.plot(
                fpr_val, tpr_val,
                lw=2, linestyle=styleline[dd + 1 + + dd*1],
                color=colors_nodes[nn],
                label=label_val
            )
            sl += 1

    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    ax.grid()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
    ax.legend(loc='center', bbox_to_anchor=(1.6, 0.5), fontsize='small')

    if mode == 'byMass':
        nameout = "{}/{}_{}_InputVars_{}_roc_by_mass.pdf".format(
            output_dir, channel, bdtType, str(len(trainvars)))
    else:
        nameout = "{}/{}_{}_InputVars_{}_roc_by_node.pdf".format(
            output_dir, channel, bdtType, str(len(trainvars)))
    fig.savefig(nameout)


def PlotClassifierByX(
        output_dir,
        global_settings,
        preferences,
        model_list,
        df_list,
        trainvars,
        label_list=['', ''],
        weights='totalWeight',
        target='target',
        mode='byNode'
):
    '''Make ROC curves as function of gen_mHH for test masses
    Parameters:
    -----------
    output_dir : str
             Path to the output directory
    global_settings : dict
                   Preferences for the data, model creation and optimization
    preferences : dict
               Preferences for the data choice and data manipulation
    model_list : list
            List of XGB Boosters obtained by training on the Odd-Even dMatrices
            pair and vice-versa
    df_list : list
           List of pandas dataframes for evaluation
    trainvars : list
             List of names of training variables
    label_list : list
              List of labels for the output plot nam
    weights: string
           name of the weights column in the data frame
    target: string
           name of the target column in the data frame
    mode: string
          evaluation mode to distinguish between Res training featuring
          gen_mHH and nonRes training featuring nodeX
    Returns:
    -----------
    Nothing
    '''
    bdtType = global_settings['bdtType']
    channel = global_settings['channel']
    nthread = global_settings['nthread']
    labelBKG = BkgLabelMaker(global_settings)
    nodes = []
    if mode == 'byMass':
        nodes = preferences['masses_test']
        if('nonres' in global_settings['bdtType']):
            raise ValueError(
                'Please use this mode only with Res!')
    elif mode == 'byNode':
        nodes = preferences['nonResScenarios_test']
        if('nonres' not in global_settings['bdtType']):
            raise ValueError(
                'Please use this mode only with nonRes!')
    else:
        raise ValueError(
            'Please use a valid mode!')

    hist_params = {'normed': True, 'bins': 10, 'histtype': 'step',
                   "lw": 2, 'range': [0, 1]}
    for nn, node in enumerate(nodes):
        plt.clf()
        colorcold = ['g', 'b']
        colorhot = ['r', 'magenta']
        fig, ax = plt.subplots(figsize=(12, 6))
        for dd in range(len(df_list)):
            val_idx = int(abs(dd-1))
            if mode == 'byMass':
                val_data = df_list[val_idx].loc[
                    df_list[val_idx]["gen_mHH"].astype(np.int) == int(node)]
                train_data = df_list[dd].loc[
                    df_list[dd]["gen_mHH"].astype(np.int) == int(node)]
            else:
                val_data = df_list[val_idx].loc[
                    (df_list[val_idx]['nodeXname'].astype(str) == str(node))]
                train_data = df_list[dd].loc[
                    (df_list[dd]['nodeXname'].astype(str) == str(node))]

            pred_BKG_test = getPred(val_data.loc[(val_data[target] == 0)],
                                    trainvars, nthread, target,
                                    weights, model_list[dd])
            BKG_test_weights = val_data.loc[
                (val_data[target] == 0), [weights]]

            pred_SIG_test = getPred(val_data.loc[(val_data[target] == 1)],
                                    trainvars, nthread, target,
                                    weights, model_list[dd])
            SIG_test_weights = val_data.loc[
                (val_data[target] == 1), [weights]]

            pred_BKG_train = getPred(train_data.loc[(train_data[target] == 0)],
                                     trainvars, nthread, target,
                                     weights, model_list[dd])
            BKG_train_weights = train_data.loc[
                (train_data[target] == 0), [weights]]

            pred_SIG_train = getPred(train_data.loc[(train_data[target] == 1)],
                                     trainvars, nthread, target,
                                     weights, model_list[dd])
            SIG_train_weights = train_data.loc[
                (train_data[target] == 1), [weights]]

            dict_plot = [
                [pred_BKG_test, BKG_test_weights, "-", colorcold[dd],
                 label_list[dd] + " training: " + label_list[val_idx] + " events (BKG: " + labelBKG + " )"],
                [pred_SIG_test, SIG_test_weights, "-", colorhot[dd],
                 label_list[dd] + " training: " + label_list[val_idx] + " events (SIG)"],
                [pred_BKG_train, BKG_train_weights, "--", colorcold[dd],
                 label_list[dd] + " training: " + label_list[dd] + " events (BKG: " + labelBKG + " )"],
                [pred_SIG_train, SIG_train_weights, "--", colorhot[dd],
                 label_list[dd] + " training: " + label_list[dd] + " events (SIG)"],
            ]
            for item in dict_plot:
                values, bins, _ = ax.hist(
                    np.array(item[0], dtype=float), weights=np.array(item[1], dtype=float),
                    ls=item[2], color=item[3],
                    label=item[4],
                    **hist_params
                )
                #  create unweighted and non normalized hist
                #  to calculate proper per bin errors
                values_unweighted, bins_unweighted = np.histogram(
                    np.array(item[0], dtype=float), bins=bins)
                yerrs = []
                for vv, value in enumerate(values_unweighted):
                    if value > 0:
                        bin_err = (math.sqrt(value)/value)*values[vv]
                    else:
                        bin_err = 0
                    yerrs.append(bin_err)
                mid = 0.5*(bins[1:] + bins[:-1])
                plt.errorbar(
                    mid, values,
                    yerr=yerrs, fmt='none',
                    color=item[3], ecolor=item[3],
                    edgecolor=item[3], lw=2
                )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
        ax.legend(loc='center', bbox_to_anchor=(1.5, 0.5),
                  title=str(node), fontsize='small')
        ax.set_xlabel('classifier output')
        ax.yaxis.set_label_text('normalized entries')
        if mode == 'byMass':
            nameout = "{}/{}_{}_{}InputVars_mass_{}_XGBClassifier.pdf".format(
                output_dir, channel,
                bdtType, str(len(trainvars)),
                str(node)
            )
        else:
            nameout = "{}/{}_{}_{}InputVars_node_{}_XGBClassifier.pdf".format(
                output_dir, channel,
                bdtType, str(len(trainvars)),
                str(node)
            )
        fig.savefig(nameout)


def get_hh_parameters(
        channel_dir,
        tau_id_training,
        info_dir,
):
    '''Reads the parameters for HH data loading

    Parameters:
    ----------
    channel_dir : str
        Path of the whole channel info direcotry
    tau_id_training : str
        Tau ID for training
    info_dir : str
        Path to the "info" firectory of the current run

    Returns:
    --------
    parameters : dict
        The necessary info for loading data
    '''
    info_path = os.path.join(info_dir, 'info.json')
    trainvars_path = os.path.join(info_dir, 'trainvars.json')
    info_dict = ut.read_json_cfg(info_path)
    default_tauID = info_dict['default_tauID_application']
    parameters = {}
    tau_id_application = info_dict.pop('tauID_application')
    parameters['tauID_application'] = tau_id_application[default_tauID]
    parameters.update(dlt.find_input_paths(info_dict, tau_id_training))
    keys = info_dict.pop('keys')
    parameters.update(dlt.load_era_keys(keys))
    trainvar_info = dlt.read_trainvar_info(trainvars_path)
    parameters['trainvars'] = []
    with open(trainvars_path, 'rt') as infile:
        for line in infile:
            info = json.loads(line)
            parameters['trainvars'].append(str(info['key']))
    all_trainvars_path = os.path.join(channel_dir, 'all_trainvars.json')
    all_trainvar_info = dlt.read_trainvar_info(all_trainvars_path)
    parameters['trainvars'] = []
    with open(trainvars_path, 'rt') as infile:
        for line in infile:
            info = json.loads(line)
            parameters['trainvars'].append(str(info['key']))
    all_trainvars_path = os.path.join(channel_dir, 'all_trainvars.json')
    all_trainvar_info = dlt.read_trainvar_info(all_trainvars_path)
    parameters['trainvar_info'] = all_trainvar_info
    parameters.update(info_dict)
    return parameters


def reweigh_dataframe(
        data,
        weight_files_dir,
        trainvar_info,
        cancelled_trainvars,
        masses,
        trainvars,
        skip_int_vars=True
):
    '''Reweighs the dataframe

    Parameters:
    ----------
    data : pandas Dataframe
        Data to be reweighed
    weighed_files_dir : str
        Path to the directory where the reweighing files are
    trainvar_info : dict
        Dictionary containing trainvar info (e.g is the trainvar supposed to
        be an integer or not)
    cancelled_trainvars :list
        list of trainvars not to include
    masses : list
        list of masses

    Returns:
    -------
    Nothing
    '''
    for trainvar in trainvars:
        if trainvar in cancelled_trainvars:
            continue
        filename = '_'.join(['TProfile_signal_fit_func', trainvar]) + '.root'
        file_path = str(os.path.join(weight_files_dir, filename))
        tfile = ROOT.TFile.Open(file_path)
        fit_function_name = str('_'.join(['fitFunction', trainvar]))
        function = tfile.Get(fit_function_name)
        if bool(trainvar_info[trainvar]) and skip_int_vars:
            data[trainvar] = data[trainvar].astype(int)
            continue
        for mass in masses:
            data.loc[
                data['gen_mHH'] == mass, [trainvar]] /= function.Eval(mass)
        tfile.Close()
