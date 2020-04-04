'''
Call with 'python3'

Usage:
    xgb_training_script_OddEvenSplit.py \
        --best_hyper_paras_file_path=BEST_HYP_DIR_PTH   \
        --output_dir=$HOME/TrainDir


Options:
  -p --best_hyper_paras_file_path=BEST_HYP_DIR_PTH
       Path to the directory containing the
       "best_hyperparameters.json" file
       containing the best hyper-para.s given by PSO
  --output_dir=$HOME/TrainDir
       output Directory for storing final training results
'''
from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import evaluation_tools as et
from machineLearning.machineLearning import xgb_tools as xt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_aux_tools as df
from sklearn.metrics import roc_curve, auc, accuracy_score
from pathlib import Path
import xgboost as xgb
import numpy as np
import os
import subprocess
import sys
import csv
import docopt
import json
import copy
import pickle


def main(best_hyper_paras_file_path, output_dir):
    assert os.path.isdir(best_hyper_paras_file_path), "Directory doesn't exist"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        run_cmd('rm -rf %s/*' % output_dir)  # clean the directory
    hyperparameter_file = os.path.join(
        best_hyper_paras_file_path, 'best_hyperparameters.json')
    hyperparameters = ut.read_parameters(hyperparameter_file)
    print("Best Hyper-Para.s: ", hyperparameters)
    settings_dir = os.path.join(best_hyper_paras_file_path, 'run_settings')
    global_settings = ut.read_settings(settings_dir, 'global')
    num_classes = global_settings['num_classes']
    nthread = global_settings['nthread']
    save_dir = str(output_dir)
    channel_dir = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/info',
        global_settings['process'],
        global_settings['channel']
    )
    histo_dicts_json = os.path.join(channel_dir, 'histo_dict.json')
    histo_dicts = ut.read_parameters(histo_dicts_json)
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

    # --- MAKING PRE-REWEIGHING PLOTS --- #
    df.MakeHisto(
        save_dir,
        global_settings['channel'],
        data,
        BDTvariables,
        histo_dicts,
        "bef_rewt",
        weights="totalWeight"
    )
    df.MakeTHStack(
        save_dir,
        global_settings['channel'],
        data,
        BDTvariables,
        histo_dicts,
        label="bef_rewt",
        weights="totalWeight"
    )
    df.MakeTProfile(
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

    # ----MAKING THE FITS FOR REWEIGHING--- #
    df.MakeTProfile(
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

    # ---- REWEIGHING ENTIRE DATAFRAME --- #
    dlt.reweigh_dataframe(
        data,
        preferences['weight_dir'],
        preferences['trainvars'],
        ['gen_mHH'],
        preferences['masses']
    )

    # --- MAKING POST-REWEIGHING PLOTS --- #
    df.MakeHisto(
        save_dir,
        global_settings['channel'],
        data,
        BDTvariables,
        histo_dicts,
        "aft_rewt",
        weights="totalWeight"
    )
    df.MakeTHStack(
        save_dir,
        global_settings['channel'],
        data,
        BDTvariables,
        histo_dicts,
        label="aft_rewt",
        weights="totalWeight"
    )
    df.MakeTProfile(
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
    df.MakeTProfile(
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

    # ----SPLITTING DATAFRAME INTO ODD-EVEN HALVES--#
    Even_df = data.loc[(data["event"].values % 2 == 0)]
    Odd_df = data.loc[~(data["event"].values % 2 == 0)]
    df_list = [Odd_df, Even_df]
    # print("Even DataFrame: ", Even_df)
    # print("Odd DataFrame: ",  Odd_df)

    # ----NORMALIZING EACH HALF SEPARATELY----#
    df.normalize_hh_dataframe(
        Even_df, preferences,
        global_settings, weight='totalWeight'
    )
    df.normalize_hh_dataframe(
        Odd_df, preferences,
        global_settings, weight='totalWeight'
    )

    # ----RUNNING SEPERATE BDT TRAINING FOR THE HALVES--#
    cls_Even_train_Odd_test = Evaluate(
                                output_dir,
                                Even_df,
                                Odd_df,
                                BDTvariables,
                                global_settings,
                                hyperparameters,
                                savePKL=True,
                                label="Even_train_Odd_test",
                                weights='totalWeight'
    )
    cls_Odd_train_Even_test = Evaluate(
                                output_dir,
                                Odd_df,
                                Even_df,
                                BDTvariables,
                                global_settings,
                                hyperparameters,
                                savePKL=True,
                                label="Odd_train_Even_test",
                                weights='totalWeight'
    )

    cls_list = [cls_Odd_train_Even_test, cls_Even_train_Odd_test]
    cls_label_list = ["Odd_train_Even_test", "Even_train_Odd_test"]
    df.PlotROCByMass(
          output_dir,
          global_settings,
          preferences,
          cls_list,
          df_list,
          BDTvariables,
          label_list=cls_label_list
    )
    df.PlotClassifierByMass(
          output_dir,
          global_settings,
          preferences,
          cls_list,
          df_list,
          BDTvariables,
          label_list=cls_label_list
    )


def run_cmd(command):
    print("executing command = '%s'" % command)
    p = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = p.communicate()
    return stdout


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
    labelBKG = df.BkgLabelMaker(global_settings)
    BDTvariables = preferences['trainvars']
    # --- PLOTING OPTIONS ---#
    printmin = True
    plotResiduals = False
    plotAll = False
    nbins = 15
    colorFast = 'g'
    colorFastT = 'b'
    plotName = "{}/{}_{}_{}_{}.pdf".format(
        str(output_dir), channel,
        bdtType, trainvar, label
    )
    df.make_plots(
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
    label : str
         Label for the output file names
    weights : str
           Name of the column in the dataframe to be used for evt weights
    Returns
    --------
    XGBClassifier object
    '''
    PlotLabel = label
    cls = xgb.XGBClassifier(
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
    cls.fit(
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
        pickle.dump(cls, open(pklFileName+".pkl", 'wb'))
        file = open(pklFileName + "pkl.log", "w")
        file.write(str(trainvars) + "\n")
        file.close()
        print ("Saved ", pklFileName+".pkl")
        print ("Variables are: ", pklFileName + "_pkl.log")
    else:
        print("No .pkl file will be saved")

    proba_train = cls.predict_proba(train[trainvars].values)
    fpr, tpr, thresholds_train = roc_curve(
        train['target'].astype(np.bool),
        proba_train[:, 1],
        sample_weight=(train[weights].astype(np.float64))
    )
    train_auc = auc(fpr, tpr, reorder=True)
    roc_train = []
    roc_train = roc_train + [{"fpr": fpr, "tpr": tpr, "train_auc": train_auc}]
    print("XGBoost train set auc - {}".format(train_auc))

    proba_test = cls.predict_proba(test[trainvars].values)
    fprt, tprt, thresholds_test = roc_curve(
        test['target'].astype(np.bool),
        proba_test[:, 1],
        sample_weight=(test[weights].astype(np.float64))
    )
    test_auc = auc(fprt, tprt, reorder=True)
    roc_test = []
    roc_test = roc_test + [{"fprt": fprt, "tprt": tprt, "test_auc": test_auc}]
    print("XGBoost test set auc - {}".format(test_auc))

    # --- PLOTTING FEATURE IMPORTANCES AND ROCs ---#
    df.PlotFeaturesImportance(
        output_dir,
        global_settings['channel'],
        cls,
        trainvars,
        label=PlotLabel
    )
    df.PlotROC(
        output_dir,
        global_settings['channel'],
        roc_train,
        roc_test,
        label=PlotLabel
    )
    df.PlotClassifier(
        output_dir,
        global_settings,
        cls,
        train,
        test,
        trainvars,
        label=PlotLabel
    )
    df.PlotCorrelation(
        output_dir,
        global_settings,
        train,
        trainvars,
        label=PlotLabel
    )
    return cls


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        best_hyper_paras_file_path = arguments['--best_hyper_paras_file_path']
        output_dir = arguments['--output_dir']
        main(best_hyper_paras_file_path, output_dir)
    except docopt.DocoptExit as e:
        print(e)
