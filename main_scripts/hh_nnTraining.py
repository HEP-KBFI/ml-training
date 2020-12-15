'''
Call with 'python'

Usage: 
    hh_nnTraining.py
    hh_nnTraining.py [--save_model=INT --mode=INT --bdtType=INT --ml_method=INT --spinCase=INT --era=INT]

Options:
    -s --save_model=INT             Whether or not to save the model [default: 0]
    -m --mode=INT                   whhether resolved or boosted catgory to be considered [default: resolved]
    -bdt --bdtType=INT              type of bdt  [default: evtLevelSUM_HH_bb2l_nonres]
    -ml_method --ml_method=INT      name of ml_method  [default: lbn]
    -spinCase --spinCase=INT        which spin to be considered [default: 0]
    -era   --era=INT                era to be processed [default: 2016]
'''
import os
import docopt
import numpy as np
import json
import pandas as pd
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
from machineLearning.machineLearning import hh_aux_tools as hhat
from machineLearning.machineLearning import nn_tools as nt
from machineLearning.machineLearning import multiclass_tools as mt
from machineLearning.machineLearning import hh_visualization_tools as hhvt
import cmsml

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

execfile('python/particle_list.py')
particles_list = ll_objects_list()

def plot_confusion_matrix(data, probabilities, output_dir, addition):
    cm = confusion_matrix(
        data["multitarget"].astype(int),
        np.argmax(probabilities, axis=1),
        sample_weight=data["totalWeight"].astype(float)
    )
    samples = []
    for i in sorted(set(data["multitarget"])):
        samples.append(list(set(data.loc[data["multitarget"] == i]["process"]))[0])
    samples = ['HH' if x.find('signal') != -1 else x for x in samples]
    hhvt.plot_confusion_matrix(
        cm, samples, output_dir, addition)

def plot_DNNScore(data, model, lbn, output_dir, trainvars, particles, addition):
    data["max_node_pos"] = -1
    data["max_node_val"] = -1
    if lbn != 'lbn':
        for process in set(data["process"]):
            data = data.loc[data["process"] == process]
            value = model.predict(data[trainvars].values)
            data.loc[data["process"] == process, "max_node_pos"]\
                = np.argmax(value, axis=1)
            data.loc[data["process"] == process, "max_node_val"]\
                = np.amax(value, axis=1)
    else:
        for process in set(data["process"]):
            process_only_data = data.loc[data["process"] == process]
            value = model.predict(
                [dlt.get_low_level(process_only_data, particles),
                 dlt.get_high_level(process_only_data, particles, trainvars)],
                batch_size=1024)
            data.loc[data['process'] == process, "max_node_pos"] = np.argmax(value, axis=1)
            data.loc[data['process'] == process, "max_node_val"] = np.amax(value, axis=1)
    hhvt.plot_DNNScore(data, output_dir, addition)

def update_global_settings(global_settings, bdtType, spinCase, mode, ml_method, era):
    global_settings['bdtType'] = bdtType
    channel = 'bb2l' if 'bb2l' in bdtType else 'bb1l'
    channel = channel+'_bdt' if 'xgb' in ml_method else channel
    global_settings['channel'] = channel
    global_settings['spinCase'] = spinCase
    global_settings['dataCuts'] = 'cuts_%s.json' %mode
    global_settings['ml_method'] = ml_method
    global_settings['mode'] = mode
    era = era.replace('20', '')
    global_settings['era'] = era
    return global_settings

def main(output_dir, save_model, bdtType, spinCase, mode, ml_method, era):
    settings_dir = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/settings'
    )
    global_settings = ut.read_settings(settings_dir, 'global')
    global_settings = update_global_settings(global_settings, bdtType, spinCase, mode, ml_method, era)
    if output_dir == 'None':
        res_nonres = 'res' if 'nonres' not in global_settings['bdtType']\
                     else 'nonres'
        res_nonres = res_nonres+"_"+global_settings['spinCase'] if 'nonres' not in global_settings['bdtType']\
                     else res_nonres
        output_dir = global_settings['channel']+'/'+global_settings['ml_method']+'/'+\
                     res_nonres + '/' + global_settings['mode'] +'/' + era
        global_settings['output_dir'] = output_dir
    else:
        global_settings['output_dir'] = output_dir
    global_settings['output_dir'] = os.path.expandvars(
        global_settings['output_dir'])
    global_settings['debug'] = False
    if not os.path.exists(global_settings['output_dir']):
        os.makedirs(global_settings['output_dir'])
    channel_dir, info_dir, _ = ut.find_settings(global_settings)
    preferences = hhat.get_hh_parameters(
        channel_dir,
        global_settings['tauID_training'],
        info_dir,
    )
    preferences = define_trainvars(global_settings, preferences, info_dir)
    particles = particles_list['bb1l'] if global_settings['channel'] == 'bb1l'\
                  else particles_list['bb2l']
    data_dict = create_data_dict(preferences, global_settings, particles)
    classes = set(data_dict["even_data"]["process"])
    for class_ in classes:
        multitarget = list(set(
            data_dict["even_data"].loc[
                data_dict["even_data"]["process"] == class_, "multitarget"
            ]
        ))[0]
        print(str(class_) + '\t' + str(multitarget))
    even_model = create_model(
        preferences, global_settings, data_dict, "even_data", save_model, particles)
    if global_settings['feature_importance'] == 1:
        trainvars = preferences['trainvars']
        data = data_dict['odd_data']
        score_dict = nt.custom_permutation_importance(
            even_model, data[trainvars], data['evtWeight'],
            trainvars, data['multitarget'], global_settings['ml_method'], particles
        )
        hhvt.plot_feature_importances_from_dict(
         score_dict, global_settings['output_dir'])
    odd_model = create_model(
        preferences, global_settings, data_dict, "odd_data", save_model, particles)
    print(odd_model.summary())
    nodewise_performance(data_dict['odd_data'], data_dict['even_data'],\
        odd_model, even_model, data_dict['trainvars'], particles, \
        global_settings, preferences)
    even_train_info, even_test_info = evaluate_model(
        even_model, data_dict['even_data'], data_dict['odd_data'],\
        data_dict['trainvars'], global_settings, "even_data", particles)
    odd_train_info, odd_test_info = evaluate_model(
        odd_model, data_dict['odd_data'], data_dict['even_data'], \
        data_dict['trainvars'], global_settings, "odd_data", particles)
    hhvt.plotROC(
        [odd_train_info, odd_test_info],
        [even_train_info, even_test_info],
        global_settings
    )

def create_data_dict(preferences, global_settings, particles):
    data = dlt.load_data(
        preferences,
        global_settings,
        eras=global_settings['era'],
        remove_neg_weights=True
    )
    for trainvar in preferences['trainvars']:
        if str(data[trainvar].dtype) == 'object':
            try:
                data[trainvar] = data[trainvar].astype(int)
            except:
                continue

    hhat.normalize_hh_dataframe(
        data,
        preferences,
        global_settings
    )
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
        "TT:W:DY:ST:HH \t" \
        + str(data.loc[data["process"] == "TT"]["totalWeight"].sum()/sumall) \
        + ":" + str(data.loc[data["process"] == "W"]["totalWeight"].sum()/sumall) \
        + ":" + str(data.loc[data["process"] == "DY"]["totalWeight"].sum()/sumall) \
        + ":" + str(data.loc[data["process"] == 'ST']["totalWeight"].sum()/sumall) \
        + ":" + str(data.loc[data["target"] == 1]["totalWeight"].sum()/sumall)
    )
    use_Wjet = True
    if 'bb2l' in global_settings['channel']: use_Wjet = False
    data = mt.multiclass_encoding(data, use_Wjet)
    hhvt.plot_correlations(data, preferences["trainvars"], global_settings)
    even_data = data.loc[(data['event'].values % 2 == 0)]
    odd_data = data.loc[~(data['event'].values % 2 == 0)]
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
        save_model,
        particles
):
    train_data = data_dict['odd_data'] if choose_data == "odd_data" else data_dict['even_data']
    val_data = data_dict['even_data']  if choose_data == "odd_data" else data_dict['odd_data']
    lbn = 1 if global_settings['ml_method'] == 'lbn' else 0
    low_level_var = ["%s_%s" %(part, var) for part in particles
                    for var in ["e", "px", "py", "pz"]]
    trainvars = preferences['trainvars']
    num_class = max((data_dict['odd_data']['multitarget'])) + 1
    number_samples = len(data_dict[choose_data])
    n_particles = len(particles)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, min_delta=0.001,
        restore_best_weights=True)

    categorical_vars = ["SM", "BM1","BM2","BM3","BM4","BM5","BM6","BM7","BM8","BM9","BM10","BM11","BM12"]
    if lbn:
        nr_trainvars = len(trainvars) - len(low_level_var)
        hl_var = [var for var in trainvars if var not in low_level_var]
        input_var = np.array([data_dict[choose_data][var] for var in hl_var])
        categorical_var_index = [hl_var.index(categorical_var) for categorical_var in categorical_vars \
                                 if categorical_var in hl_var]
    else :
        nr_trainvars = len(trainvars)
        input_var = np.array([data_dict[choose_data][var] for var in trainvars])
        categorical_var_index = [trainvars.index(categorical_var) for categorical_var in \
                             categorical_vars if categorical_var in trainvars]
    if len(categorical_var_index) == 0: categorical_var_index = None
    model_structure = nt.create_nn_model(
        nr_trainvars,
        num_class,
        input_var,
        categorical_var_index,
        n_particles,
        lbn=lbn
    )
    if global_settings['ml_method'] == 'lbn':
        fitted_model = model_structure.fit(
            [dlt.get_low_level(train_data, particles),
             dlt.get_high_level(train_data, particles, trainvars)],
            train_data['multitarget'].values,
            epochs=25,
            batch_size=1024,
            sample_weight=train_data['totalWeight'].values,
            validation_data=(
                [dlt.get_low_level(val_data, particles),
                 dlt.get_high_level(val_data, particles, trainvars)],
                val_data["multitarget"].values,
                val_data["totalWeight"].values
            ),
            callbacks=[reduce_lr, early_stopping]
        )
    else:
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
            callbacks=[reduce_lr, early_stopping]
        )
    if save_model:
        savemodel(model_structure, global_settings)
    hhvt.plot_loss_accuracy(fitted_model, global_settings['output_dir'], choose_data)

    return model_structure

def nodewise_performance(odd_data, even_data,
                         odd_model, even_model, trainvars, particles,\
                         global_settings, preferences):
    if 'nonres' in global_settings['bdtType']:
        nodes = preferences['nonResScenarios_test']
        mode = 'nodeXname'
    else:
        nodes = preferences['masses_test']
        mode = 'gen_mHH'
    roc_infos = []
    for node in nodes:
        split_odd_data = odd_data.loc[odd_data[mode] == node]
        split_even_data = even_data.loc[even_data[mode] == node]
        odd_total_infos = list(evaluate_model(
            odd_model, split_odd_data, split_even_data, trainvars,
            global_settings, 'odd_data', particles, True
        ))
        even_total_infos = list(evaluate_model(
            even_model, split_even_data, split_odd_data, trainvars,
            global_settings, 'even_data', particles, True
        ))
        roc_info = {
            'even_auc_test': even_total_infos[1]['auc'],
            'odd_auc_test': odd_total_infos[1]['auc'],
            'even_auc_train': even_total_infos[0]['auc'],
            'odd_auc_train': odd_total_infos[0]['auc'],
            'even_fpr_test': even_total_infos[1]['fpr'],
            'odd_fpr_test': odd_total_infos[1]['fpr'],
            'even_fpr_train': even_total_infos[0]['fpr'],
            'odd_fpr_train': odd_total_infos[0]['fpr'],
            'even_tpr_test': even_total_infos[1]['tpr'],
            'odd_tpr_test': odd_total_infos[1]['tpr'],
            'even_tpr_train': even_total_infos[0]['tpr'],
            'odd_tpr_train': odd_total_infos[0]['tpr'],
            'node': node
        }
        #nodeWise_performances.append(nodeWise_histo_dict)
        roc_infos.append(roc_info)
    #hhvt.plot_nodeWise_performance(
    #global_settings, nodeWise_performances, mode)
    hhvt.plot_nodeWise_roc(global_settings, roc_infos, mode)

def evaluate_model(model, train_data, test_data, trainvars, \
                   global_settings, choose_data, particles, nodeWise=False):

    if global_settings['ml_method'] == 'lbn':
        train_predicted_probabilities = model.predict(
            [dlt.get_low_level(train_data, particles),
             dlt.get_high_level(train_data, particles, trainvars)],
            batch_size=1024)
        test_predicted_probabilities = model.predict(
            [dlt.get_low_level(test_data, particles),
             dlt.get_high_level(test_data, particles, trainvars)],
            batch_size=1024)
        #print test_var["ll"][0], 'hl==', test_var["hl"][0]
        #print 'proba===', test_predicted_probabilities[0]
    else:
        train_predicted_probabilities = model.predict(
            train_data[trainvars].values)
        test_predicted_probabilities = model.predict(
            test_data[trainvars].values)
    if not nodeWise:
        plot_confusion_matrix(test_data, test_predicted_probabilities, \
           global_settings["output_dir"], choose_data+'_test')
        plot_confusion_matrix(train_data, train_predicted_probabilities, \
           global_settings["output_dir"], choose_data+'_train')
        plot_DNNScore(train_data, model, global_settings['ml_method'], \
           global_settings['output_dir'], trainvars, particles, choose_data+'_train')
        plot_DNNScore(test_data, model, global_settings['ml_method'], \
           global_settings['output_dir'], trainvars, particles, choose_data+'_test')

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

def savemodel(model_structure, global_settings):
     if 'nonres' in global_settings["bdtType"]:
         res_nonres = 'nonres'
     else:
        res_nonres = 'res_%s' %global_settings['spinCase']
     pb_filename = os.path.join(global_settings["output_dir"], "multiclass_DNN_w%s_for_%s_%s_%s_%s.pb"\
                                %(global_settings["ml_method"], global_settings["channel"], \
                                  global_settings["mode"], choose_data, res_nonres))
     log_filename = os.path.join(global_settings["output_dir"], "multiclass_DNN_w%s_for_%s_%s_%s_%s.log"\
                                %(global_settings["ml_method"], global_settings["channel"], \
                                  global_settings["mode"], choose_data, res_nonres))
     cmsml.tensorflow.save_graph(pb_filename, model_structure, variables_to_constants=True)
     file = open(log_filename, "w")
     file.write(str(hl_var))
     file.close()

def define_trainvars(global_settings, preferences, info_dir):
    if global_settings["ml_method"] == "lbn" :
        trainvars_path = os.path.join(info_dir, 'trainvars.json')
    if global_settings["dataCuts"].find("boosted") != -1 :
        trainvars_path = os.path.join(info_dir, 'trainvars_boosted.json')
    if global_settings["dataCuts"].find("boosted") != -1 and global_settings["ml_method"] == "lbn":
        trainvars_path = os.path.join(info_dir, 'trainvars_boosted.json')
    if global_settings["dataCuts"].find("boosted") == -1 and global_settings["ml_method"] == "lbn":
        trainvars_path = os.path.join(info_dir, 'trainvars.json')
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
        bdtType = arguments['--bdtType']
        spinCase = arguments['--spinCase']
        mode = arguments['--mode']
        ml_method = arguments['--ml_method']
        era = arguments['--era']
        main('onlySMwovbf', save_model, bdtType, spinCase, mode, ml_method, era)
    except docopt.DocoptExit as e:
        print(e)
    print(datetime.now() - startTime)
