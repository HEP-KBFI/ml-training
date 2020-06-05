#!/usr/bin/env python
'''
Final Training script for the HH analysis
using the Odd-Even method taking PSO
optmized hyper-para.s as input
Call with 'python3'

Usage:
    xgb_training_script_OddEvenSplit.py \
        --best_hyper_paras_file_path=BEST_HYP_DIR_PTH   \
        --output_dir=$HOME/TrainDir \
        --skipInterpolStudy=<BOOL>

Options:
  -p --best_hyper_paras_file_path=BEST_HYP_DIR_PTH
       Path to the directory containing the
       "best_hyperparameters.json" file
       containing the best hyper-para.s given by PSO
  -o --output_dir=$HOME/TrainDir
       output Directory for storing final training results
  -s --skipInterpolStudy=BOOL  [default: True]
       Skip mass interpolation studies ? True or False
'''
from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_aux_tools as hhat
from sklearn.metrics import roc_curve, auc, accuracy_score
import xgboost as xgb
import numpy as np
import os
import docopt
import json
import copy
import pickle
import shutil


def main(best_hyper_paras_file_path, output_dir, skipInterpolStudy):
    print("skipInterpolStudy: ", StrToBool(skipInterpolStudy))
    if not StrToBool(skipInterpolStudy):
        print("DOING MASS INTERPOL. STUDIES FOR THE HALVES")
    else:
        print("RUNNING SEPERATE BDT TRAINING FOR THE HALVES")

    assert os.path.isdir(best_hyper_paras_file_path), "Directory doesn't exist"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)  # delete the old directory
        os.makedirs(output_dir)
    hyperparameter_file = os.path.join(
        best_hyper_paras_file_path, 'best_hyperparameters.json')
    hyperparameters = ut.read_parameters(hyperparameter_file)
    print("Best Hyper-Para.s: ", hyperparameters)
    settings_dir = os.path.join(best_hyper_paras_file_path, 'run_settings')
    global_settings = ut.read_settings(settings_dir, 'global')
    num_classes = global_settings['num_classes']
    nthread = global_settings['nthread']
    save_dir = str(output_dir)
    ut.save_run_settings(output_dir)
    ut.save_info_settings(output_dir, global_settings)
    channel_dir = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/info',
        global_settings['process'],
        global_settings['channel']
    )
    histo_dicts_json = os.path.join(channel_dir, 'histo_dict.json')
    histo_dicts = ut.read_parameters(histo_dicts_json)
    print("global_settings[tauID_training] = ", global_settings['tauID_training'])
    preferences = dlt.get_hh_parameters(
        global_settings['channel'],
        global_settings['tauID_training']
    )
    data = dlt.load_data(
        preferences['inputPath'],
        preferences['channelInTree'],
        preferences['trainvars'],
        global_settings['bdtType'],
        global_settings['channel'],
        preferences['keys'],
        preferences['masses'],
        global_settings['bkg_mass_rand']
    )
    BDTvariables = preferences['trainvars']
    print("BDTvariables: ", BDTvariables)

    # Removing gen_mHH from the list of input variables
    BDTvariables_wo_gen_mHH = copy.deepcopy(BDTvariables)
    BDTvariables_wo_gen_mHH.remove("gen_mHH")
    print("BDTvariables_wo_gen_mHH", BDTvariables_wo_gen_mHH)

    print("MAKING PRE-REWEIGHING PLOTS")
    hhat.MakeHisto(
        save_dir,
        global_settings['channel'],
        data,
        BDTvariables,
        histo_dicts,
        "bef_rewt",
        weights="totalWeight"
    )
    hhat.MakeTHStack(
        save_dir,
        global_settings['channel'],
        data,
        BDTvariables,
        histo_dicts,
        label="bef_rewt",
        weights="totalWeight"
    )
    hhat.MakeTProfile(
        save_dir,
        preferences['masses'],
        global_settings['channel'],
        data,
        BDTvariables_wo_gen_mHH,
        histo_dicts,
        Target=0,
        doFit=False,
        label="bef_rewt",
        TrainMode=0,
        weights="totalWeight"
    )
    PlotInputVar(
        data,
        preferences,
        global_settings,
        save_dir,
        label="bef_rewt_BDT",
        weights="totalWeight"
    )

    print("MAKING THE FITS FOR REWEIGHING")
    hhat.MakeTProfile(
        save_dir,
        preferences['masses'],
        global_settings['channel'],
        data,
        BDTvariables_wo_gen_mHH,
        histo_dicts,
        Target=1,
        doFit=True,
        label="bef_rewt",
        TrainMode=0,
        weights="totalWeight"
    )

    
    print("REWEIGHING ENTIRE DATAFRAME")
    dlt.reweigh_dataframe(
        data,
        preferences['weight_dir'],
        preferences['trainvars'],
        ['gen_mHH'],
        preferences['masses']
    )

    print("MAKING POST-REWEIGHING PLOTS")
    hhat.MakeHisto(
        save_dir,
        global_settings['channel'],
        data,
        BDTvariables,
        histo_dicts,
        "aft_rewt",
        weights="totalWeight"
    )
    hhat.MakeTHStack(
        save_dir,
        global_settings['channel'],
        data,
        BDTvariables,
        histo_dicts,
        label="aft_rewt",
        weights="totalWeight"
    )
    hhat.MakeTProfile(
        save_dir,
        preferences['masses'],
        global_settings['channel'],
        data,
        BDTvariables_wo_gen_mHH,
        histo_dicts,
        Target=1,
        doFit=False,
        label="aft_rewt",
        TrainMode=0,
        weights="totalWeight"
    )
    hhat.MakeTProfile(
        save_dir,
        preferences['masses'],
        global_settings['channel'],
        data,
        BDTvariables_wo_gen_mHH,
        histo_dicts,
        Target=0,
        doFit=False,
        label="aft_rewt",
        TrainMode=0,
        weights="totalWeight"
    )
    PlotInputVar(
        data,
        preferences,
        global_settings,
        save_dir,
        label="aft_rewt_BDT",
        weights="totalWeight"
    )
    print("NORMALIZING DATAFRAME")
    hhat.normalize_hh_dataframe(
        data, preferences,
        global_settings, weight='totalWeight'
    )
    print("SPLITTING DATAFRAME INTO ODD-EVEN HALVES")
    # exclude background samples from splitting
    # if the background is estimated from data
    # to gain more training statistics
    keysNotToSplit = []
    if ("3l_1tau" in global_settings['channel']):
        keysNotToSplit = ['WZTo', 'DY']
        print "These keys are excluded from splitting: ", keysNotToSplit
    else:
        print "PLEASE IMPLEMENT SETTINGS FOR YOUR CHANNEL"
    evtNotToSplit = (data['key'].isin(keysNotToSplit))
    evtEven = (data["event"].values % 2 == 0)
    evtOdd = ~(data["event"].values % 2 == 0)
    Even_df = data.loc[np.logical_or(evtEven, evtNotToSplit)]
    Odd_df = data.loc[np.logical_or(evtOdd, evtNotToSplit)]
    df_list = [Odd_df, Even_df]

    if not StrToBool(skipInterpolStudy):
        print("DOING MASS INTERPOL. STUDIES FOR THE HALVES")
        InterpolTest(
             output_dir,
             global_settings,
             preferences,
             hyperparameters,
             df_list,
             preferences['masses'],
             preferences['masses_test'],
             label_list=["Odd", "Even"],
             weights='totalWeight',
             target='target'
        )
    else:
        print("RUNNING SEPERATE BDT TRAINING FOR THE HALVES")
        model_Even_train_Odd_test = Evaluate(
                                    output_dir,
                                    Even_df,
                                    Odd_df,
                                    BDTvariables,
                                    global_settings,
                                    hyperparameters,
                                    savePKL=True,
                                    makePlots=True,
                                    label="Even_train_Odd_test",
                                    weights='totalWeight'
        )
        model_Odd_train_Even_test = Evaluate(
                                    output_dir,
                                    Odd_df,
                                    Even_df,
                                    BDTvariables,
                                    global_settings,
                                    hyperparameters,
                                    savePKL=True,
                                    makePlots=True,
                                    label="Odd_train_Even_test",
                                    weights='totalWeight'
        )
        model_list = [model_Odd_train_Even_test, model_Even_train_Odd_test]
        model_label_list = ["Odd_train_Even_test", "Even_train_Odd_test"]
        hhat.PlotROCByMass(
            output_dir,
            global_settings,
            preferences,
            model_list,
            df_list,
            BDTvariables,
            label_list=model_label_list
        )
        hhat.PlotClassifierByMass(
            output_dir,
            global_settings,
            preferences,
            model_list,
            df_list,
            BDTvariables,
            label_list=model_label_list
        )
    

def PlotInputVar(
        data,
        preferences,
        global_settings,
        output_dir,
        label="",
        weights='totalWeight'
):
    '''Make Input Var plots for a given dataframe
    Parameters:
    -----------
    data : pandas Dataframe
        Dataframe containing all the data needed for the training.
    preferences : dict
               Preferences for the data choice and data manipulation
    global_settings : dict
                   Preferences for the data, model creation and optimization
    output_dir : str
              Path to the directory where plots will be stored
    label : str
         Label for plot name
    weights : str
           Name of the column in the dataframe to be used as event weight
    Returns
    --------
    Nothing
    '''
    bdtType = global_settings['bdtType']
    channel = global_settings['channel']
    trainvar = global_settings['trainvar']
    test_masses = preferences["masses_test"]
    mass_list = preferences['masses']
    labelBKG = hhat.BkgLabelMaker(global_settings)
    BDTvariables = preferences['trainvars']
    # --- PLOTING OPTIONS ---#
    printmin = True
    plotResiduals = False
    nbins = 15
    colorFast = 'g'
    colorFastT = 'b'
    plotName = "{}/{}_{}_{}_{}.pdf".format(
        str(output_dir), channel,
        bdtType, trainvar, label
    )
    hhat.make_plots(
        BDTvariables, nbins,
        data.ix[data.target.values == 0], labelBKG, colorFast,
        data.ix[data.target.values == 1], 'Signal', colorFastT,
        plotName, printmin,
        plotResiduals, test_masses,
        mass_list, weights
    )


def Evaluate(
        output_dir,
        train,
        test,
        trainvars,
        global_settings,
        hyperparameters,
        savePKL=False,
        makePlots=True,
        label="",
        weights='totalWeight'
):
    '''Perform BDT training for a given train, test dataframe pair
    Parameters:
    -----------
    output_dir : str
              Path to store the output files
    train : pandas dataframe
         Training dataset
    test : pandas dataframe
         Testing dataset
    global_settings : dict
        Preferences for the data, model creation and optimization
    hyperparameters : dict
                   Dictionary containing info of hyper-parameters
    savePKL : bool
           Handle to save the training as a .pkl file
    makePlots : bool
             Handle to make and plots
    label : str
         Label for the output file names
    weights : str
           Name of the column in the dataframe to be used for evt weights
    Returns
    --------
    XGBClassifier object
    '''
    PlotLabel = label
    model = xgb.XGBClassifier(
               n_estimators=hyperparameters[0]["num_boost_round"],
               max_depth=hyperparameters[0]["max_depth"],
               min_child_weight=hyperparameters[0]["min_child_weight"],
               learning_rate=hyperparameters[0]["learning_rate"],
               gamma=hyperparameters[0]["gamma"],
               subsample=hyperparameters[0]["subsample"],
               colsample_bytree=hyperparameters[0]["colsample_bytree"],
               nthread=global_settings["nthread"],
               num_class=global_settings["num_classes"],
               objective="multi:softprob"
    )
    model.fit(
        train[trainvars].values,
        train['target'].astype(np.bool),
        sample_weight=(train[weights].astype(np.float64))
    )
    if(savePKL):
        channel = global_settings['channel']
        trainvar = global_settings['trainvar']
        bdtType = global_settings['bdtType']
        VarNos = str(len(trainvars))
        pklFileName = "{}/{}_XGB_{}_{}_InputVars{}_train_{}".format(
            output_dir, channel, trainvar,
            bdtType, VarNos, label
        )
        pickle.dump(model, open(pklFileName+".pkl", 'wb'))
        file = open(pklFileName + "pkl.log", "w")
        file.write(str(trainvars) + "\n")
        file.close()
        print("Saved ", pklFileName+".pkl")
        print("Variables are: ", pklFileName + "_pkl.log")
    else:
        print("No .pkl file will be saved")

    if(makePlots):
        proba_train = model.predict_proba(train[trainvars].values)
        fpr, tpr, thresholds_train = roc_curve(
            train['target'].astype(np.bool),
            proba_train[:, 1],
            sample_weight=(train[weights].astype(np.float64))
        )
        train_auc = auc(fpr, tpr, reorder=True)
        roc_train = []
        roc_train = roc_train + [{
                     "fpr": fpr,
                     "tpr": tpr,
                     "train_auc": train_auc
        }]
        print("XGBoost train set auc - {}".format(train_auc))
        proba_test = model.predict_proba(test[trainvars].values)
        fprt, tprt, thresholds_test = roc_curve(
            test['target'].astype(np.bool),
            proba_test[:, 1],
            sample_weight=(test[weights].astype(np.float64))
        )
        test_auc = auc(fprt, tprt, reorder=True)
        roc_test = []
        roc_test = roc_test + [{
                    "fprt": fprt,
                    "tprt": tprt,
                    "test_auc": test_auc
        }]
        print("XGBoost test set auc - {}".format(test_auc))
        # --- PLOTTING FEATURE IMPORTANCES AND ROCs ---#
        hhat.PlotFeaturesImportance(
            output_dir,
            global_settings['channel'],
            model,
            trainvars,
            label=PlotLabel
        )
        hhat.PlotROC(
            output_dir,
            global_settings['channel'],
            roc_train,
            roc_test,
            label=PlotLabel
        )
        hhat.PlotClassifier(
            output_dir,
            global_settings,
            model,
            train,
            test,
            trainvars,
            label=PlotLabel
        )
        hhat.PlotCorrelation(
            output_dir,
            global_settings,
            train,
            trainvars,
            label=PlotLabel
        )
    else:
        print("No plots will be made")
    return model


def StrToBool(
        string
):
    '''Function to convert string to bool
    Parameters:
    ----------
    file_name : str
              input string
    Returns
    --------
    boolean
    '''
    if(string == 'False'):
        return False
    elif(string == 'True'):
        return True


def InterpolTest(
        output_dir,
        global_settings,
        preferences,
        hyperparameters,
        df_list,
        mass_list,
        test_masses,
        label_list=["Odd", "Even"],
        weights='totalWeight',
        target='target'
):
    '''Builds all files necessary to perform MVA interpolation studies
    Parameters:
    -----------
    output_dir : str
              Path to store the output files
    global_settings : dict
        Preferences for the data, model creation and optimization
    preferences : dict
               Preferences for the data choice and data manipulation
    hyperparameters : dict
                   Dictionary containing info of hyper-parameters
    df_list : list
           List of pandas dataframes for evaluation
    mass_list : List
             List of masses to train
    test_masses : list
               List of masses to test (must be a subset of mass_list)
    label_list : str list
         List of Labels for the output file names
    weights : str
           Name of the column in the dataframe to be used for evt weights
    target : str
           Name of the column in the dataframe used for Signal/Bkg label
    Returns
    --------
    Nothing
    '''
    BDTvariables = preferences['trainvars']
    print("BDTvariables: ", BDTvariables)

    # Removing gen_mHH from the list of input variables
    BDTvariables_wo_gen_mHH = copy.deepcopy(BDTvariables)
    BDTvariables_wo_gen_mHH.remove("gen_mHH")
    print("BDTvariables_wo_gen_mHH", BDTvariables_wo_gen_mHH)

    channel = global_settings['channel']
    Bkg_mass_rand = global_settings['bkg_mass_rand']
    for mm, mass in enumerate(test_masses):  # Loop over the test masses
        print("Performing interpol. test for mass: ", mass)
        MASS_STR = "{}".format(int(mass))
        logFile = "{}/{}_InterpolMass_{}.json".format(
            output_dir, channel, MASS_STR
        )
        for dd, data_do in enumerate(df_list):  # Loop over Odd-Even dfs
            if(dd == 0):
                val_data = 1
            else:
                val_data = 0
            print("Bkg_mass_rand: ", Bkg_mass_rand)
            if(Bkg_mass_rand == "default"):
                traindataset1 = data_do.loc[
                    ~((data_do["gen_mHH"] == mass) &
                      (data_do[target] == 1))
                ]  # Training for all signal masses except the test mass
                valdataset1 = df_list[val_data].loc[
                    ~((df_list[val_data]["gen_mHH"] != mass) &
                      (df_list[val_data][target] == 1))
                ]  # Testing on only the test mass as signal
                traindataset2 = data_do.loc[
                    ~((data_do["gen_mHH"] != mass) &
                      (data_do[target] == 1))
                ]  # Training on only the test mass as signal
                valdataset2 = valdataset1.copy(deep=True)
            else:
                traindataset1 = data_do.loc[
                    ~(data_do["gen_mHH"] == mass)
                ]  # Training for all signal masses except the test mass
                valdataset1 = df_list[val_data].loc[
                    (df_list[val_data]["gen_mHH"] == mass)
                ]  # Testing on only the test mass as signal
                traindataset2 = data_do.loc[
                    (data_do["gen_mHH"] == mass)
                ]  # Training on only the test mass as signal
                valdataset2 = valdataset1.copy(deep=True)
            mass_list_copy = list(mass_list)
            mass_list_copy.remove(mass)
            masses1 = mass_list_copy  # list of all but the test mass
            masses_test1 = [masses1[0]]  # taking first element of masses1
            masses2 = [mass]  # mass_list
            masses_test2 = [mass]
            labelBKG = hhat.BkgLabelMaker(global_settings)
            Label = label_list[dd]+"_train_"+label_list[val_data]+"_test"
            plot_name_train1 = "{}/{}_{}_{}.pdf".format(
                output_dir,
                Label,
                "_InputVars_traindataset1_BDT_",
                MASS_STR
            )
            plot_name_val1 = "{}/{}_{}_{}.pdf".format(
                output_dir,
                Label,
                "_InputVars_valdataset1_BDT_",
                MASS_STR
            )
            plot_name_train2 = "{}/{}_{}_{}.pdf".format(
                output_dir,
                Label,
                "_InputVars_traindataset2_BDT_",
                MASS_STR
            )
            plot_name_val2 = "{}/{}_{}_{}.pdf".format(
                output_dir,
                Label,
                "_InputVars_valdataset2_BDT_",
                MASS_STR
            )
            # --- PLOTING OPTIONS ---#
            printmin = True
            plotResiduals = False
            nbins = 15
            colorFast = 'g'
            colorFastT = 'b'
            hhat.make_plots(
                BDTvariables,
                nbins,
                traindataset1.ix[traindataset1[target].values == 0],
                labelBKG,
                colorFast,
                traindataset1.ix[traindataset1[target].values == 1],
                'Signal',
                colorFastT,
                plot_name_train1,
                printmin,
                plotResiduals,
                masses_test1,
                masses1,
                weights
            )
            hhat.make_plots(
                BDTvariables,
                nbins,
                valdataset1.ix[valdataset1[target].values == 0],
                labelBKG,
                colorFast,
                valdataset1.ix[valdataset1[target].values == 1],
                'Signal',
                colorFastT,
                plot_name_val1,
                printmin,
                plotResiduals,
                masses_test2,
                masses2,
                weights
            )
            hhat.make_plots(
                BDTvariables,
                nbins,
                traindataset2.ix[traindataset2[target].values == 0],
                labelBKG,
                colorFast,
                traindataset2.ix[traindataset2[target].values == 1],
                'Signal',
                colorFastT,
                plot_name_train2,
                printmin,
                plotResiduals,
                masses_test1,
                masses1,
                weights
            )
            hhat.make_plots(
                BDTvariables,
                nbins,
                valdataset2.ix[valdataset2[target].values == 0],
                labelBKG,
                colorFast,
                valdataset2.ix[valdataset2[target].values == 1],
                'Signal',
                colorFastT,
                plot_name_val2,
                printmin,
                plotResiduals,
                masses_test2,
                masses2,
                weights
            )
            print("EVALUATION WITH gen_mHH AS INPUT VAR.")
            print("TRAIN ALL MASSES EXCEPT THE TEST MASS")
            print("VALIDATE USING ONLY THE TEST MASS")
            print('Train all masses except %0.1f GeV' % mass)
            print('Test mass %0.1f GeV (%s)' % (mass, Label))
            WriteInterpolLogFile(
                traindataset1,
                valdataset1,
                BDTvariables,
                global_settings,
                hyperparameters,
                test_mass=mass,
                train_df_label=label_list[dd],
                test_df_label=label_list[val_data],
                train_mode_label="train_all_test_one_w_genmHH",
                target='target',
                weights='totalWeight',
                LogFile=logFile
            )
            print("EVALUATION WITH gen_mHH AS INPUT VAR.")
            print("TRAIN USING ONLY THE TEST MASS")
            print("VALIDATE USING ONLY THE TEST MASS")
            print('Train mass %0.1f GeV' % mass)
            print('Test mass %0.1f GeV (%s)' % (mass, Label))
            WriteInterpolLogFile(
                traindataset2,
                valdataset2,
                BDTvariables,
                global_settings,
                hyperparameters,
                test_mass=mass,
                train_df_label=label_list[dd],
                test_df_label=label_list[val_data],
                train_mode_label="train_one_test_one_w_genmHH",
                target='target',
                weights='totalWeight',
                LogFile=logFile
            )
            print("EVALUATION W/O gen_mHH AS INPUT VAR.")
            print("TRAIN ALL MASSES EXCEPT THE TEST MASS")
            print("VALIDATE USING ONLY THE TEST MASS")
            print('Train masses excpt %0.1f GeV (w/o gen_mHH)' % mass)
            print('Test mass %0.1f GeV (w/o gen_mHH) (%s)' % (mass, Label))
            WriteInterpolLogFile(
                traindataset1,
                valdataset1,
                BDTvariables_wo_gen_mHH,
                global_settings,
                hyperparameters,
                test_mass=mass,
                train_df_label=label_list[dd],
                test_df_label=label_list[val_data],
                train_mode_label="train_all_test_one_wo_genmHH",
                target='target',
                weights='totalWeight',
                LogFile=logFile
            )
            print("EVALUATION W/O gen_mHH AS INPUT VAR.")
            print("TRAIN USING ONLY THE TEST MASS")
            print("VALIDATE USING ONLY THE TEST MASS")
            print('Train mass %0.1f GeV (w/o gen_mHH)' % mass)
            print('Test mass %0.1f GeV (%s) (w/o gen_mHH)' % (mass, Label))
            WriteInterpolLogFile(
                traindataset2,
                valdataset2,
                BDTvariables_wo_gen_mHH,
                global_settings,
                hyperparameters,
                test_mass=mass,
                train_df_label=label_list[dd],
                test_df_label=label_list[val_data],
                train_mode_label="train_one_test_one_wo_genmHH",
                target='target',
                weights='totalWeight',
                LogFile=logFile
            )


def WriteJSONFile(
        out_dicts,
        LogFile='file.json'
):
    '''Function to write .json files
    Parameters:
    -----------
    out_dicts : python dict
            dictionary to dump
    LogFile : str
           Full path to the output .json file
    Returns
    --------
    Nothing
    '''
    with open(LogFile, 'at') as out_file:
        json.dump(out_dicts, out_file)
        out_file.write('\n')


def WriteInterpolLogFile(
        traindataset,
        valdataset,
        trainvars,
        global_settings,
        hyperparameters,
        test_mass=300,
        train_df_label="ODD",
        test_df_label="EVEN",
        train_mode_label="train_all_test_one",
        target='target',
        weights='totalWeight',
        LogFile='file.json'
):
    '''Function to create interpol info
       and write it to a .json file
    Parameters:
    -----------
    traindataset : pandas dataframe
                dataset used to train XGBClassifier object
    valdataset : pandas dataframe
              dataset used to test XGBClassifier object
    trainvars : list
             List of input training variables
    global_settings : dict
        Preferences for the data, model creation and optimization
    hyperparameters : dict
                   Dictionary containing info of hyper-parameters
    test_mass : float
             test mass value
    train_df_label : str
         Label to distinguish training rows in log file
    test_df_label : str
         Label to distinguish test rows in log file
    train_mode_label : str
              Label to distinuguish training mode in interpol.
    target : str
           Name of the column in the dataframe used for Signal/Bkg label
    weights : str
           Name of the column in the dataframe to be used for evt weights
    LogFile : str
           Full path to the .json file used for writing interpol info
    Returns
    --------
    Nothing
    '''
    Mass_str = "{}".format(int(test_mass))
    Train_Mode_str = train_mode_label
    Train_df_str = train_df_label
    Test_df_str = test_df_label
    model = xgb.XGBClassifier(
               n_estimators=hyperparameters[0]["num_boost_round"],
               max_depth=hyperparameters[0]["max_depth"],
               min_child_weight=hyperparameters[0]["min_child_weight"],
               learning_rate=hyperparameters[0]["learning_rate"],
               gamma=hyperparameters[0]["gamma"],
               subsample=hyperparameters[0]["subsample"],
               colsample_bytree=hyperparameters[0]["colsample_bytree"],
               nthread=global_settings["nthread"],
               num_class=global_settings["num_classes"],
               objective="multi:softprob"
    )
    print("FITTING")
    model.fit(
        traindataset[trainvars].values,
        traindataset[target].astype(np.bool),
        sample_weight=(traindataset[weights].astype(np.float64))
    )
    print("CALCULATING predict_proba() FOR TRAIN DF")
    proba_train = model.predict_proba(traindataset[trainvars].values)
    fpr, tpr, thresholds = roc_curve(
        traindataset[target].astype(np.bool),
        proba_train[:, 1],
        sample_weight=(traindataset[weights].astype(np.float64))
    )
    train_auc = auc(fpr, tpr, reorder=True)
    print("CALCULATING predict_proba() FOR TEST DF")
    proba_val = model.predict_proba(valdataset[trainvars].values)
    fprt, tprt, thresholdst = roc_curve(
        valdataset[target].astype(np.bool),
        proba_val[:, 1],
        sample_weight=(valdataset[weights].astype(np.float64))
    )
    test_auc = auc(fprt, tprt, reorder=True)
    # ---- BDT Output distributions ----#
    y_pred_test = model.predict_proba(
        valdataset.loc[
            (valdataset[target].values == 0)
        ][trainvars].values
    )[:, 1]
    y_pred_test_weights = valdataset.loc[
        (valdataset[target].values == 0)
    ][weights]
    y_predS_test = model.predict_proba(
        valdataset.loc[
            (valdataset[target].values == 1)
        ][trainvars].values
    )[:, 1]
    y_predS_test_weights = valdataset.loc[
        (valdataset[target].values == 1)
    ][weights]
    y_pred_train = model.predict_proba(
        traindataset.loc[
            (traindataset[target].values == 0)
        ][trainvars].values
    )[:, 1]
    y_pred_train_weights = traindataset.loc[
        (traindataset[target].values == 0)
    ][weights]
    y_predS_train = model.predict_proba(
        traindataset.loc[
            (traindataset[target].values == 1)
        ][trainvars].values
    )[:, 1]
    y_predS_train_weights = traindataset.loc[
        (traindataset[target].values == 1)
    ][weights]

    # ------ Labels -----#
    train_auc_label = ('train_auc_%s_%s_%s' %
                       (Train_df_str, Train_Mode_str, Mass_str))
    test_auc_label = ('test_auc_%s_%s_%s' %
                      (Test_df_str, Train_Mode_str, Mass_str))
    xtrain_label = ('xtrain_%s_%s_%s' %
                    (Train_df_str, Train_Mode_str, Mass_str))
    ytrain_label = ('ytrain_%s_%s_%s' %
                    (Train_df_str, Train_Mode_str, Mass_str))
    xtest_label = ('xval_%s_%s_%s' %
                   (Test_df_str, Train_Mode_str, Mass_str))
    ytest_label = ('yval_%s_%s_%s' %
                   (Test_df_str, Train_Mode_str, Mass_str))
    y_pred_train_label = ('y_pred_train_%s_%s_%s' %
                          (Train_df_str, Train_Mode_str, Mass_str))
    y_pred_train_weights_label = ('y_pred_train_weights_%s_%s_%s' %
                                  (Train_df_str, Train_Mode_str, Mass_str))
    y_pred_test_label = ('y_pred_test_%s_%s_%s' %
                         (Test_df_str, Train_Mode_str, Mass_str))
    y_pred_test_weights_label = ('y_pred_test_weights_%s_%s_%s' %
                                 (Test_df_str, Train_Mode_str, Mass_str))
    y_predS_train_label = ('y_predS_train_%s_%s_%s' %
                           (Train_df_str, Train_Mode_str, Mass_str))
    y_predS_train_weights_label = ('y_predS_train_weights_%s_%s_%s' %
                                   (Train_df_str, Train_Mode_str, Mass_str))
    y_predS_test_label = ('y_predS_test_%s_%s_%s' %
                          (Test_df_str, Train_Mode_str, Mass_str))
    y_predS_test_weights_label = ('y_predS_test_weights_%s_%s_%s' %
                                  (Test_df_str, Train_Mode_str, Mass_str))
    out_dicts = [
        {train_auc_label: [train_auc]},
        {test_auc_label: [test_auc]},
        {xtrain_label: fpr.tolist()},
        {ytrain_label: tpr.tolist()},
        {xtest_label: fprt.tolist()},
        {ytest_label: tprt.tolist()},
        {y_pred_train_label: y_pred_train.tolist()},
        {y_pred_train_weights_label: y_pred_train_weights.tolist()},
        {y_pred_test_label: y_pred_test.tolist()},
        {y_pred_test_weights_label: y_pred_test_weights.tolist()},
        {y_predS_train_label: y_predS_train.tolist()},
        {y_predS_train_weights_label: y_predS_train_weights.tolist()},
        {y_predS_test_label: y_predS_test.tolist()},
        {y_predS_test_weights_label: y_predS_test_weights.tolist()}
    ]
    final_dict = ut.to_one_dict(out_dicts)
    WriteJSONFile(final_dict, LogFile)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        print(arguments)
        best_hyper_paras_file_path = arguments['--best_hyper_paras_file_path']
        output_dir = arguments['--output_dir']
        skipInterpolStudy = arguments['--skipInterpolStudy']
        if arguments['--skipInterpolStudy'] == 'True':
            print("Perfoming the final training")
        else:
            print("Performing mass interpolation studies")
        main(best_hyper_paras_file_path, output_dir, skipInterpolStudy)
    except docopt.DocoptExit as e:
        print(e)
