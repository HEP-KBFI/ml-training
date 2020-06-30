import matplotlib
matplotlib.use('agg')
from ROOT import TCanvas, TFile, TProfile
from ROOT import TH1D, THStack, TF1
from ROOT import gPad, TFitResultPtr
from machineLearning.machineLearning import data_loading_tools as dlt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import math
import copy
import os
import ROOT
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
    if 'oversampling' in bkg_mass_rand:
        data.loc[(data['target'] == 1), [weight]] *= 1./float(
            len(preferences['masses']))
        data.loc[(data['target'] == 0), [weight]] *= 1./float(
            len(preferences['masses']))
    if 'SUM_HH' in bdt_type:
        ttbar_weights = data.loc[data['key'].isin(ttbar_samples), [weight]]
        ttbar_factor = preferences['TTdatacard']/ttbar_weights.sum()
        data.loc[data['key'].isin(ttbar_samples), [weight]] *= ttbar_factor
        dy_weights = data.loc[data['key'] == 'DY', [weight]]
        dy_factor = preferences['DYdatacard']/dy_weights.sum()
        data.loc[data['key'] == 'DY', [weight]] *= dy_factor
        if "evtLevelSUM_HH_bb1l_res" in bdt_type:
            w_weights = data.loc[data['key'] == 'W', [weight]]
            w_factor = preferences['Wdatacard']/w_weights.sum()
            data.loc[data['key'] == 'W', [weight]] *= w_factor
        if "evtLevelSUM_HH_2l_2tau_res" in bdt_type:
            zz_weights = data.loc[data['key'] == 'ZZ', [weight]]
            zz_factor = preferences['ZZdatacard']/zz_weights.sum()
            data.loc[data['key'] == 'ZZ', [weight]] *= zz_factor
        if "evtLevelSUM_HH_3l_1tau_res" in bdt_type:
            zz_samples = ['ZZTo', 'ggZZTo']
            zz_weights = data.loc[data['key'].isin(zz_samples), [weight]]
            zz_factor = preferences['ZZdatacard']/zz_weights.sum()
            data.loc[data['key'].isin(zz_samples), [weight]] *= zz_factor
            wz_weights = data.loc[data['key'] == 'WZTo', [weight]]
            wz_factor = preferences['WZdatacard']/wz_weights.sum()
            data.loc[data['key'] == 'WZTo', [weight]] *= wz_factor
        for mass in range(len(preferences['masses'])):
            condition_mass = data['gen_mHH'].astype(int) == int(
                preferences['masses'][mass])
            mass_sig_weight = data.loc[
                condition_sig & condition_mass, [weight]]
            sig_mass_factor = 100000./mass_sig_weight.sum()
            data.loc[
                condition_sig & condition_mass, [weight]] *= sig_mass_factor
            mass_bkg_weight = data.loc[
                condition_bkg & condition_mass, [weight]]
            bkg_mass_factor = 100000./mass_bkg_weight.sum()
            data.loc[
                condition_bkg & condition_mass, [weight]] *= bkg_mass_factor
    else:
        sig_factor = 100000./data.loc[condition_sig, [weight]].sum()
        data.loc[condition_sig, [weight]] *= sig_factor
        bkg_factor = 100000./data.loc[condition_bkg, [weight]].sum()
        data.loc[condition_bkg, [weight]] *= bkg_factor


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
    print("labelBKG: ", labelBKG)
    return labelBKG


def make_plots(
        featuresToPlot, nbin,
        data1, label1, color1,
        data2, label2, color2,
        plotname,
        printmin,
        plotResiduals,
        masses=[],
        masses_all=[],
        weights="totalWeight"
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
    masses : list
          list of input signal masses to plot
    masses_all : list
          list of all input signal masses (Needed to plot gen_mHH histo.)
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
            nbin_local = 10*len(masses_all)
            range_local = [
                masses_all[0]-20,
                masses_all[len(masses_all)-1]+20
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
                                    data1[feature].values,
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
        if len(masses) == 0:  # 'gen_mHH' not in feature
            values2, bins, _ = plt.hist(
                data2[feature].values,
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
            colors_mass = [
                'm', 'b', 'k', 'r', 'g',  'y', 'c',
                'chocolate', 'teal', 'pink', 'darkkhaki',
                'maroon', 'slategray', 'orange', 'silver',
                'aquamarine', 'lavender', 'goldenrod', 'salmon',
                'tan', 'lime', 'lightcoral'
            ]
            for mm, mass in enumerate(masses):
                gen_mHH_mass = data2["gen_mHH"].astype(np.int)
                plot_features = data2.loc[(gen_mHH_mass == int(mass)),
                                          feature].values
                plot_weights = data2.loc[(gen_mHH_mass == int(mass)),
                                         weights].values
                plot_wt_float = plot_weights.astype(np.float64)
                plot_label = label2 + "gen_mHH = " + str(mass)
                values2, bins, _ = plt.hist(
                                       plot_features,
                                       weights=plot_wt_float,
                                       range=range_local,
                                       bins=nbin_local,
                                       edgecolor=colors_mass[mm],
                                       color=colors_mass[mm],
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
                        color=colors_mass[mm], ecolor=colors_mass[mm],
                        edgecolor=colors_mass[mm], lw=2
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


def numpyarrayTProfileFill(
        arr_X,
        arr_Y,
        arr_wt,
        tprof
):
    '''Filling a ROOT TProfile plot
    Parameters:
    -----------
    arr_X : numpy array
          array of X-axis values
    arr_Y : numpy array
          array of Y-axis values
    arr_wt : numpy array
          array of weight values
    tprof : ROOT TProfile object
          TProfile plot to be filled
    Returns:
    --------
    Nothing
    '''
    for x, y, w in np.nditer([arr_X, arr_Y, arr_wt]):
        # print("x: {}, y: {}, w: {}".format(x, y, w))
        tprof.Fill(x, y, w)


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
    if(channel == "3l_1tau"):
        zz_samples = ['ZZTo', 'ggZZTo']
        data_copy_ZZ = data.loc[
            (data['key'].isin(zz_samples))]  # ZZ
        data_copy_WZ = data.loc[
            (data['key'] == 'WZTo')]  # WZ
        ttbar_samples = ['TTTo2L2Nu', 'TTToSemiLeptonic']
        data_copy_TT = data.loc[
            (data['key'].isin(ttbar_samples))]  # TTbar
        data_copy_DY = data.loc[
            (data['key'] == 'DY')]  # DY
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
    else:
        print("Please implement settings for your own channel")


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
        print("Variable Name: {}".format(var_name))
        data_X = np.array(data_copy[var_name].values, dtype=np.float)
        data_wt = np.array(data_copy[weights].values, dtype=np.float)

        N_x = len(data_X)
        N_wt = len(data_wt)

        if(N_x == N_wt):
            print("Plotting Histogram: {}".format(var_name))

            # Create a new canvas, and customize it.
            c1 = TCanvas('c1', 'Histogram', 200, 10, 700, 500)
            # c1.SetFillColor(42)
            # c1.GetFrame().SetFillColor(21)
            c1.GetFrame().SetBorderSize(6)
            c1.GetFrame().SetBorderMode(-1)

            PlotTitle = var_name
            Histo_Dict = dlt.find_correct_dict(
                'Variable', str(var_name), histo_dicts)
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
            print("N_x: {}, N_wt: {}".format(N_x, N_wt))


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
        print("Variable Name: {}".format(var_name))
        data_X = np.array(data_copy[var_name].values, dtype=np.float)
        data_wt = np.array(data_copy[weights].values, dtype=np.float)

        N_x = len(data_X)
        N_wt = len(data_wt)

        if(N_x == N_wt):
            print("Plotting Histogram: {}".format(var_name))

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
            print("N_x: {}, N_wt: {}".format(N_x, N_wt))


def MakeTProfile(
        output_dir,
        mass_list,
        channel,
        data,
        var_name_list_wo_gen_mHH,
        histo_dicts,
        Target=0,
        doFit=False,
        label="",
        TrainMode=0,
        weights="totalWeight"
):
    '''Makes TProfile plots of input var.s as function of gen_mHH
    and (optionally) fits them with a polynomial function
    Parameters:
    -----------
    output_dir : str
              Path to the output directory
    channel : str
           Label denoting the channel
    data : pandas Dataframe
        Dataframe of training var.s for Bkg.
    var_name_list_wo_gen_mHH : list
                List of strings of Dataframe column names to plot
    histo_dicts : list
              List of python dicts containing info about plot settings
    Target : int
         Training label for signal (1) and Background (0)
    doFit : bool
        Handle to start doing the polynomial fit to the TProfile plot
    label : str
         Label to append to output .root file names
    TrainMode : int
            Index to specify signal mass range: 0/1/2 (All/Low/High Masses)
    mass_list : list
            List of signal mass points
    weights : str
           Name of dataframe column used for weighing all the histograms
    Returns:
    -------
    Nothing
    '''
    assert os.path.isdir(output_dir), "Directory doesn't exist"

    data_copy = data.copy(deep=True)
    data_copy = data_copy.loc[(data_copy['target'] == Target)]

    # Create a new canvas, and customize it.
    c1 = TCanvas('c1', 'TProfile plot', 200, 10, 700, 500)
    # c1.SetFillColor(42)
    # c1.GetFrame().SetFillColor(21)
    c1.GetFrame().SetBorderSize(6)
    c1.GetFrame().SetBorderMode(-1)

    for var_name in var_name_list_wo_gen_mHH:
        print("Variable Name: {}".format(var_name))
        FileName = ""
        Fit_Func_FileName = ""
        if(Target == 1):
            FileName = "{}/{}_{}_{}.root".format(
                output_dir, "TProfile_signal",
                str(var_name), label
            )
            if(doFit):
                Fit_Func_FileName = "{}/{}_{}.root".format(
                    output_dir,
                    "TProfile_signal_fit_func",
                    var_name
                )
        else:
            FileName = "{}/{}_{}_{}.root".format(
                output_dir, "TProfile",
                str(var_name), label
            )
        data_X = np.array(data_copy['gen_mHH'].values, dtype=np.float)
        data_Y = np.array(data_copy[var_name].values, dtype=np.float)
        data_wt = np.array(data_copy[weights].values, dtype=np.float)
        N_x = len(data_X)
        N_y = len(data_Y)
        N_wt = len(data_wt)

        if((N_x == N_y) and (N_y == N_wt)):
            print("N_x: {}, N_y: {}, N_wt: {}".format(N_x, N_y, N_wt))
            PlotTitle = 'Profile of '+str(var_name)+' vs gen_mHH'
            Histo_Dict = dlt.find_correct_dict(
                'Variable', str(var_name), histo_dicts)
            Nbins = (len(mass_list) - 1)
            xlow = mass_list[0]
            xhigh = (mass_list[(len(mass_list) - 1)] + 100.0)
            ylow = Histo_Dict["min"]
            yhigh = Histo_Dict["max"]
            hprof = TProfile(
                'hprof', PlotTitle, Nbins,
                xlow, xhigh, ylow, yhigh
            )
            new_mass_list = [float(i) for i in mass_list]
            xbins = np.array(new_mass_list)
            hprof.SetBins((len(xbins) - 1), xbins)
            hprof.GetXaxis().SetTitle("gen_mHH (GeV)")
            hprof.GetYaxis().SetTitle(str(var_name))
            numpyarrayTProfileFill(data_X, data_Y, data_wt, hprof)
            hprof.Draw()
            c1.Modified()
            c1.Update()

            if(doFit and (Target == 1)):  # do the fit for signal only
                fitFuncName = "fitFunction_" + str(var_name)
                mass_low = float(mass_list[0])
                mass_high = float(mass_list[(len(mass_list) - 1)])
                print("f_Name: %s, m_l: %f, m_h: %f",
                      (fitFuncName, mass_low, mass_high))
                if(TrainMode == 0):  # All masses used in the training
                    fit_poly_order = Histo_Dict["fitFunc_AllMassTraining"]
                    f_old = TF1(
                        fitFuncName, fit_poly_order,
                        mass_low, mass_high
                    )
                elif(TrainMode == 1):  # Only Low masses used in the training
                    fit_poly_order = Histo_Dict["fitFunc_LowMassTraining"]
                    f_old = TF1(
                        fitFuncName, fit_poly_order,
                        mass_low, mass_high
                    )
                elif(TrainMode == 2):  # Only High masses used in the training
                    fit_poly_order = Histo_Dict["fitFunc_HighMassTraining"]
                    f_old = TF1(
                        fitFuncName, fit_poly_order,
                        mass_low, mass_high
                    )
                else:
                    assert TrainMode == 0, "Invalid TrainMode: Choose 0/1/2"
                r_old = TFitResultPtr()
                r_old = hprof.Fit(f_old, "SF")  # Fit using Minuit
                f_old.Draw("same")
                c1.Modified()
                c1.Update()
                c1.SaveAs(FileName)
                FuncFile = TFile(Fit_Func_FileName, "RECREATE")
                f_old.Write()
                FuncFile.Close()
            else:
                print("No fit will be performed")
                c1.SaveAs(FileName)
        else:
            print('Arrays not of same length')
            print("N_x: {}, N_y: {}, N_wt: {}".format(N_x, N_y, N_wt))


def PlotFeaturesImportance(
        output_dir,
        channel,
        cls,
        trainvars,
        label=""
):
    '''Plot Importance of Input Variables
    Parameters:
    -----------
    output_dir : str
             Path to the output directory
    channel : str
             Label for the channel
    cls : XGBClassifier
        Classifier obtained by fitting to the train dataset
    trainvars : list
             List of names of training variables
    label : str
         Label for the output plot name
    Returns:
    -----------
    Nothing
    '''
    fig, ax = plt.subplots()
    f_score_dict = cls.get_booster().get_fscore()
    fig, ax = plt.subplots()
    f_score_dict = cls.get_booster().get_fscore()
    f_score_dict = {trainvars[int(k[1:])]: v for k, v in f_score_dict.items()}
    feat_imp = pd.Series(f_score_dict).sort_values(ascending=True)
    feat_imp.plot(kind='barh', title='Feature Importances')
    fig.tight_layout()
    nameout = "{}/{}_{}_InputVar_Importance.pdf".format(
        output_dir, channel, label)
    fig.savefig(nameout)


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
        cls,
        train,
        test,
        trainvars,
        label=""
):
    '''Makes plot for Classifier output for train and test data
    Parameters:
    -----------
    output_dir : str
             Path to the output directory
    global_settings : dict
              Preferences for the data, model creation and optimization
    cls : XGBClassifier
        Classifier obtained by fitting to the train dataset
    train : pandas dataframe
         Training dataset
    test : pandas dataframe
        Testing dataset
    trainvars : list
             List of names of training variables
    label : str
         Label for the output plot name
    Returns:
    -----------
    Nothing
    '''
    labelBKG = BkgLabelMaker(global_settings)

    channel = global_settings["channel"]
    bdtType = global_settings["bdtType"]

    hist_params = {'normed': True, 'bins': 10, 'histtype': 'step', "lw": 2}
    plt.clf()
    y_pred_train = cls.predict_proba(
        train.ix[train.target.values == 0,
                 trainvars].values)[:, 1]
    y_predS_train = cls.predict_proba(
        train.ix[train.target.values == 1,
                 trainvars].values)[:, 1]
    y_pred_test = cls.predict_proba(
        test.ix[test.target.values == 0,
                trainvars].values)[:, 1]
    y_predS_test = cls.predict_proba(
        test.ix[test.target.values == 1,
                trainvars].values)[:, 1]
    plt.figure('XGB', figsize=(6, 6))
    PlotLabel_Bkg_train = labelBKG+' (train)'
    PlotLabel_sig_train = 'signal (train)'
    PlotLabel_Bkg_test = labelBKG+' (test)'
    PlotLabel_sig_test = 'signal (test)'
    values, bins, _ = plt.hist(
        y_pred_train, ls="-", color='g',
        label=PlotLabel_Bkg_train, **hist_params)
    values, bins, _ = plt.hist(
        y_predS_train, ls="-", color='r',
        label=PlotLabel_sig_train, **hist_params)
    values, bins, _ = plt.hist(
        y_pred_test, ls="--", color='b',
        label=PlotLabel_Bkg_test, **hist_params)
    values, bins, _ = plt.hist(
        y_predS_test, ls="--", color='magenta',
        label=PlotLabel_sig_test, **hist_params)
    plt.legend(loc='best')
    plt.savefig(
        output_dir + '/' + channel + '_' + bdtType +
        '_InputVars_' + str(len(trainvars)) + '_' +
        label + '_XGBclassifier.pdf'
    )


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


def PlotROCByMass(
        output_dir,
        global_settings,
        preferences,
        cls_list,
        df_list,
        trainvars,
        label_list=["", ""],
        weights="totalWeight",
        target='target'
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
    cls_list : list
            List of XGBClassifiers obtained by fitting to Odd-Even dataframe
            pair and vice-versa
    df_list : list
           List of pandas dataframes for evaluation
    trainvars : list
             List of names of training variables
    label_list : list
              List of labels for the output plot name
    Returns:
    -----------
    Nothing
    '''
    bdtType = global_settings['bdtType']
    channel = global_settings['channel']
    test_masses = preferences["masses_test"]
    estimator = cls_list
    order_train = df_list
    order_train_name = label_list

    # by mass ROC
    styleline = ['-', '--', '-.', ':']
    colors_mass = ['m', 'b', 'k', 'r', 'g',  'y', 'c', ]
    fig, ax = plt.subplots(figsize=(6, 6))
    sl = 0
    for mm, mass in enumerate(test_masses):
        for dd, data_do in enumerate(order_train):
            if dd == 0:
                val_data = 1
            else:
                val_data = 0
            proba = estimator[dd].predict_proba(
                data_do.loc[(data_do["gen_mHH"].astype(np.int) == int(mass)),
                            trainvars].values)
            fpr, tpr, thresholds = roc_curve(
                data_do.loc[(data_do["gen_mHH"].astype(np.int) == int(mass)),
                            target].astype(np.bool), proba[:, 1],
                sample_weight=(data_do.loc[
                    (data_do["gen_mHH"].astype(np.int) == int(mass)),
                    weights].astype(np.float64))
            )
            train_auc = auc(fpr, tpr, reorder=True)
            print("train set auc " + str(train_auc) +
                  " (mass = " + str(mass) + ")")
            proba = estimator[dd].predict_proba(
                order_train[val_data].loc[
                    (order_train[val_data]["gen_mHH"].astype(np.int)
                     == int(mass)), trainvars].values)
            fprt, tprt, thresholds = roc_curve(
                order_train[val_data].loc[
                    (order_train[val_data]["gen_mHH"].astype(np.int)
                     == int(mass)), target].astype(np.bool), proba[:, 1],
                sample_weight=(order_train[val_data].loc[
                    (order_train[val_data]["gen_mHH"].astype(np.int)
                     == int(mass)), weights].astype(np.float64))
            )
            test_auct = auc(fprt, tprt, reorder=True)
            print("test set auc " + str(test_auct) +
                  " (mass = " + str(mass) + ")")
            ax.plot(
                fpr, tpr,
                lw=2, linestyle=styleline[dd + dd*1],
                color=colors_mass[mm],
                label=order_train_name[dd] +
                ' train (area = %0.3f)' % (train_auc) +
                " (mass = " + str(mass) + ")"
            )
            sl += 1
            ax.plot(
                fprt, tprt,
                lw=2, linestyle=styleline[dd + 1 + + dd*1],
                color=colors_mass[mm],
                label=order_train_name[dd] +
                ' test (area = %0.3f)' % (test_auct) +
                " (mass = " + str(mass) + ")"
            )
            sl += 1
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right", fontsize='small')
    ax.grid()
    nameout = "{}/{}_{}_InputVars_{}_roc_by_mass.pdf".format(
        output_dir, channel, bdtType, str(len(trainvars)))
    fig.savefig(nameout)


def PlotClassifierByMass(
        output_dir,
        global_settings,
        preferences,
        cls_list,
        df_list,
        trainvars,
        label_list=["", ""],
        weights="totalWeight",
        target='target'
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
    cls_list : list
            List of XGBClassifiers obtained by fitting to Odd-Even dataframe
            pair and vice-versa
    df_list : list
           List of pandas dataframes for evaluation
    trainvars : list
             List of names of training variables
    label_list : list
              List of labels for the output plot name
    Returns:
    -----------
    Nothing
    '''
    bdtType = global_settings['bdtType']
    channel = global_settings['channel']
    test_masses = preferences["masses_test"]
    labelBKG = BkgLabelMaker(global_settings)
    estimator = cls_list
    order_train = df_list
    order_train_name = label_list

    hist_params = {'normed': True, 'bins': 10, 'histtype': 'step', "lw": 2}
    for mm, mass in enumerate(test_masses):
        plt.clf()
        colorcold = ['g', 'b']
        colorhot = ['r', 'magenta']
        fig, ax = plt.subplots(figsize=(6, 6))
        for dd, data_do in enumerate(order_train):
            if dd == 0:
                val_data = 1
            else:
                val_data = 0
            y_pred = estimator[dd].predict_proba(
                order_train[val_data].loc[
                    (order_train[val_data][target].values == 0) &
                    (order_train[val_data]["gen_mHH"].astype(np.int)
                     == int(mass)), trainvars].values)[:, 1]
            y_predS = estimator[dd].predict_proba(
                order_train[val_data].loc[
                    (order_train[val_data][target].values == 1) &
                    (order_train[val_data]["gen_mHH"].astype(np.int)
                     == int(mass)), trainvars].values)[:, 1]
            y_pred_train = estimator[dd].predict_proba(
                data_do.ix[(data_do[target].values == 0) &
                           (data_do["gen_mHH"].astype(np.int)
                            == int(mass)), trainvars].values)[:, 1]
            y_predS_train = estimator[dd].predict_proba(
                data_do.ix[(data_do[target].values == 1) &
                           (data_do["gen_mHH"].astype(np.int)
                            == int(mass)), trainvars].values)[:, 1]
            dict_plot = [
                [y_pred, "-", colorcold[dd],
                 order_train_name[dd] + " test " + labelBKG],
                [y_predS, "-", colorhot[dd],
                 order_train_name[dd] + " test signal"],
                [y_pred_train, "--", colorcold[dd],
                 order_train_name[dd] + " train " + labelBKG],
                [y_predS_train, "--", colorhot[dd],
                 order_train_name[dd] + " train signal"]
            ]
            for item in dict_plot:
                values1, bins, _ = ax.hist(
                    item[0],
                    ls=item[1], color=item[2],
                    label=item[3],
                    **hist_params
                )
                normed = sum(y_pred)
                mid = 0.5*(bins[1:] + bins[:-1])
                err = np.sqrt(values1*normed)/normed
                plt.errorbar(
                    mid, values1,
                    yerr=err, fmt='none',
                    color=item[2], ecolor=item[2],
                    edgecolor=item[2], lw=2
                )
        ax.legend(loc='upper center',
                  title="mass = "+str(mass)+" GeV", fontsize='small')
        nameout = "{}/{}_{}_{}InputVars_mass_{}GeV_XGBClassifier.pdf".format(
            output_dir, channel,
            bdtType, str(len(trainvars)),
            str(mass)
        )
        fig.savefig(nameout)
