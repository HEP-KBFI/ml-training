'''
Call with 'python'

Usage:
    hh_nnTraining.py
    hh_nnTraining.py [--channel=STR --res_nonres=STR --mode=STR --era=INT --BM=INT --split_ggf_vbf=INT --sig_weight=INT --spin=INT --mass_region=STR]

Options:
    -channel --channel=INT          which channel to be considered [default: bb1l]
    -res_nonres --res_nonres=STR        res or nonres to be considered [default: nonres]
    -m --mode=INT                   whhether resolved or boosted catgory to be considered [default: resolved]
    -era --era=INT                era to be processed [default: 2016]
    -BM --BM=STR                    BM point to be considered  [default: None]
    -split --split_ggf_vbf=INT    whether want to split ggf and vbf [default: 0]
    -sig_weight --sig_weight=INT  total signal weight to be consifered [default: 1]
    -sp --spin=STR              wgich spin to be used [default: None]
    -mr --mass_region=STR       which mass region to be considered [default: None]
'''
import os
import json
from datetime import datetime
import docopt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve as roc
from sklearn.metrics import auc
import matplotlib
matplotlib.use('agg')
import cmsml
from machineLearning.machineLearning import data_loading_tools as dlt
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import hh_parameter_reader as hpr
from machineLearning.machineLearning import bbWW_tools as bbwwt
from machineLearning.machineLearning import nn_tools as nt
from machineLearning.machineLearning import multiclass_tools as mt
from machineLearning.machineLearning.visualization import hh_visualization_tools as hhvt

tf.config.threading.set_intra_op_parallelism_threads(3)
tf.config.threading.set_inter_op_parallelism_threads(3)

PARTICLE_INFO = low_level_object = {
    'bb1l': ["bjet1", "bjet2", "wjet1", "wjet2", "lep"],
    'bb2l': ["bjet1", "bjet2", "lep1", "lep2"]
}

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

def main(output_dir, channel, mode, era, BM, split_ggf_vbf, sig_weight, mass_region):
    settings_dir = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/settings'
    )
    global_settings = ut.read_settings(settings_dir, 'global')
    create_global_settings(global_settings, channel, res_nonres, mode)
    print('global settings ' + str(global_settings))
    if output_dir == 'None':
        mode_BM = mode
        if spin != 'None':
            assert(mass_region != 'None')
            mode_BM += '/' + spin + '_' + mass_region
        else:
            mode_BM += '/' + BM
        output_dir = global_settings['channel'] + '/split_ggf_vbf_' + str(split_ggf_vbf) + \
                '_sig_weight_' + str(sig_weight) + '/' + global_settings['ml_method'] \
                +'/'+ res_nonres + '/' + mode_BM + '/' + era
        global_settings['output_dir'] = output_dir
    else:
        global_settings['output_dir'] = output_dir
    global_settings['output_dir'] = os.path.expandvars(
        global_settings['output_dir'])
    if not os.path.exists(global_settings['output_dir']):
        os.makedirs(global_settings['output_dir'])
    print ('output dir :' + global_settings['output_dir'])
    channel_dir, info_dir, _ = ut.find_settings(global_settings)
    scenario = global_settings['scenario']
    reader = hpr.HHParameterReader(channel_dir, scenario)
    preferences = reader.parameters
    if not BM == 'None':
        preferences["nonResScenarios"] = [BM]
    if BM == 'all':
        preferences["nonResScenarios"] = ["SM", "BM1","BM2","BM3","BM4","BM5","BM6","BM7","BM8","BM9","BM10","BM11","BM12"]
    print('BM point to be considered: ' + str(preferences["nonResScenarios"]))
    if not era == '0':
        preferences['included_eras'] = [era.replace('20', '')]
    print('era: ' + str(preferences['included_eras']))
    preferences['signal_weight'] = sig_weight
    preferences['masses'] = preferences['masses_%s' %mass_region]
    preferences = define_trainvars(global_settings, preferences, info_dir)
    if split_ggf_vbf:
        preferences['trainvars'].append('vbf_m_jj')
        preferences['trainvars'].append('vbf_dEta_jj')
    check_trainvar(preferences)
    particles = PARTICLE_INFO[global_settings['channel']]
    data_dict = create_data_dict(preferences, global_settings, split_ggf_vbf)
    classes = set(data_dict["even_data"]["process"])
    for class_ in classes:
        multitarget = list(set(
            data_dict["even_data"].loc[
                data_dict["even_data"]["process"] == class_, "multitarget"
            ]
        ))[0]
        print(str(class_) + '\t' + str(multitarget))
    even_model = create_model(
        preferences, global_settings,\
        data_dict['even_data_train'], data_dict['even_data_val'],\
        "even_data", split_ggf_vbf)
    odd_model = create_model(
        preferences, global_settings,\
        data_dict['odd_data_train'], data_dict['odd_data_val'],\
        "odd_data", split_ggf_vbf)
    print(odd_model.summary())
    nodewise_performance(data_dict['odd_data_train'], data_dict['even_data_train'],\
        data_dict['odd_data'], data_dict['even_data'],\
        odd_model, even_model, data_dict['trainvars'], particles, \
        global_settings, preferences)
    even_train_info, even_test_info = evaluate_model(
        even_model, data_dict['even_data_train'], data_dict['odd_data'],\
        data_dict['trainvars'], global_settings, "even_data", particles)
    odd_train_info, odd_test_info = evaluate_model(
        odd_model, data_dict['odd_data_train'], data_dict['even_data'], \
        data_dict['trainvars'], global_settings, "odd_data", particles)
    hhvt.plotROC(
        [odd_train_info, odd_test_info],
        [even_train_info, even_test_info],
        global_settings
    )
    if global_settings['feature_importance'] == 1:
        trainvars = preferences['trainvars']
        LBNFeatureImportance = nt.LBNFeatureImportances(even_model, data_dict['odd_data'],\
            trainvars, global_settings['channel'])
        score_dict = LBNFeatureImportance.custom_permutation_importance()
        hhvt.plot_feature_importances_from_dict(
            score_dict, global_settings['output_dir'])

def create_data_dict(preferences, global_settings, split_ggf_vbf):
    normalizer = bbwwt.bbWWDataNormalizer
    mergeWjets = 'bb2l' in global_settings["channel"]
    loader = bbwwt.bbWWLoader(
        normalizer,
        preferences,
        global_settings,
        split_ggf_vbf,
        mergeWjets,
        False,
        False
    )
    data = loader.data
    hhvt.plot_single_mode_correlation(
        data, preferences['trainvars'],
        global_settings['output_dir'], 'trainvar'
    )
    hhvt.plot_trainvar_multi_distributions(
        data, preferences['trainvars'],
        global_settings['output_dir']
    )
    sumall = 0
    for process in list(set(data['process'])):
        sumall += data.loc[data["process"] == process]["totalWeight"].sum()
    print('totalWeight of each process:')
    for process in list(set(data['process'])):
        print(process + ': ' +str("%0.3f" %(data.loc[data["process"] == process]["totalWeight"].sum()/sumall)))
    use_Wjet = True
    if 'bb2l' in global_settings['channel']:
        use_Wjet = False
    data = mt.multiclass_encoding(data, use_Wjet, split_ggf_vbf)
    hhvt.plot_correlations(data, preferences["trainvars"], global_settings)
    even_data = data.loc[(data['event'].values % 2 == 0)]
    odd_data = data.loc[~(data['event'].values % 2 == 0)]
    even_data_train = even_data.sample(frac=0.80)
    even_data_val = even_data.drop(even_data_train.index)
    odd_data_train = odd_data.sample(frac=0.80)
    odd_data_val = odd_data.drop(odd_data_train.index)
    data_dict = {
        'trainvars': preferences['trainvars'],
        'odd_data':  odd_data,
        'even_data': even_data,
        'odd_data_train': odd_data_train,
        'odd_data_val': odd_data_val,
        'even_data_train': even_data_train,
        'even_data_val': even_data_val,
    }
    return data_dict

def create_model(
        preferences,
        global_settings,
        train_data,
        val_data,
        choose_data,
        split_ggf_vbf,
        save_model=True
):
    lbn = global_settings['ml_method'] == 'lbn'
    parameters = {'epoch':25, 'batch_size':601, 'lr':0.00781838825015861, 'l2':0.0, 'dropout':0, 'layer':5, 'node':212}
    if lbn:
        modeltype = nt.LBNmodel(
            train_data,
            val_data,
            preferences['trainvars'],
            global_settings['channel'],
            parameters,
            True,
            split_ggf_vbf,
            global_settings['output_dir'],
            choose_data
        )
    model = modeltype.create_model()
    BMpoint = preferences["nonResScenarios"][0] if len(preferences["nonResScenarios"]) ==1\
              else 'all'
    if save_model:
        savemodel(model, preferences['trainvars'], global_settings, choose_data, \
                  BMpoint, '20%s' %preferences['included_eras'][0])
    return model

def nodewise_performance(odd_data_train, even_data_train,
                         odd_data_test, even_data_test,
                         odd_model, even_model, trainvars, particles,\
                         global_settings, preferences):
    if 'nonres' in global_settings['scenario']:
        nodes = preferences['nonResScenarios_test']
        mode = 'nodeXname'
    else:
        nodes = preferences['masses_test_%s' %mass_region]
        mode = 'gen_mHH'
    roc_infos = []
    for node in nodes:
        split_odd_data_train = odd_data_train.loc[odd_data_train[mode] == node]
        split_even_data_train = even_data_train.loc[even_data_train[mode] == node]
        split_odd_data_test = odd_data_test.loc[odd_data_test[mode] == node]
        split_even_data_test = even_data_test.loc[even_data_test[mode] == node]
        odd_total_infos = list(evaluate_model(
            odd_model, split_odd_data_train, split_even_data_test, trainvars,
            global_settings, 'odd_data', particles, True
        ))
        even_total_infos = list(evaluate_model(
            even_model, split_even_data_train, split_odd_data_test, trainvars,
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
     #   global_settings, nodeWise_performances, mode)
    hhvt.plot_nodeWise_roc(global_settings, roc_infos, mode)

def evaluate_model(model, train_data, test_data, trainvars, \
                   global_settings, choose_data, particles, nodeWise=False):

    train_data["max_node_pos"] = -1
    train_data["max_node_val"] = -1
    test_data["max_node_pos"] = -1
    test_data["max_node_val"] = -1
    if global_settings['ml_method'] == 'lbn':
        train_predicted_probabilities = model.predict(
            [dlt.get_low_level(train_data, particles),
             dlt.get_high_level(train_data, particles, trainvars)],
            batch_size=1024)
        test_predicted_probabilities = model.predict(
            [dlt.get_low_level(test_data, particles),
             dlt.get_high_level(test_data, particles, trainvars)],
            batch_size=1024)
    else:
        train_predicted_probabilities = model.predict(
            train_data[trainvars].values)
        test_predicted_probabilities = model.predict(
            test_data[trainvars].values)
    if not nodeWise:
        train_data["max_node_val"] = np.amax(train_predicted_probabilities, axis=1)
        train_data["max_node_pos"] = np.argmax(train_predicted_probabilities, axis=1)
        test_data["max_node_val"] = np.amax(test_predicted_probabilities, axis=1)
        test_data["max_node_pos"] = np.argmax(test_predicted_probabilities, axis=1)
        plot_confusion_matrix(test_data, test_predicted_probabilities, \
            global_settings["output_dir"], choose_data+'_test')
        plot_confusion_matrix(train_data, train_predicted_probabilities, \
            global_settings["output_dir"], choose_data+'_train')
        hhvt.plot_DNNScore(train_data, global_settings['output_dir'], choose_data+'_train')
        hhvt.plot_DNNScore(test_data, global_settings['output_dir'], choose_data+'_test')

    '''test_fpr, test_tpr = mt.roc_curve(
        test_data['multitarget'].astype(int),
        test_predicted_probabilities,
        test_data['totalWeight'].astype(float)
    )
    train_fpr, train_tpr = mt.roc_curve(
        train_data['multitarget'].astype(int),
        train_predicted_probabilities,
        train_data['totalWeight'].astype(float)
    )'''
    test_fpr, test_tpr, _ = roc(
        test_data['multitarget'].astype(int),
        test_predicted_probabilities[:,0],
        sample_weight=(test_data['totalWeight'].astype(float)),
        pos_label=0
    )
    train_fpr, train_tpr, _ = roc(
        train_data['multitarget'].astype(int),
        train_predicted_probabilities[:,0],
        sample_weight=(train_data['totalWeight'].astype(float)),
        pos_label=0
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

def savemodel(model_structure, trainvars, global_settings, addition, BM, era):
    if 'spin' not in global_settings["scenario"]:
        res_nonres = 'nonres_' + BM
        mode = global_settings['scenario'].split('/')[2]
    else:
        res_nonres = 'res_%s_%s' %(global_settings['scenario'].split('/')[0], mass_region)
        mode = global_settings['scenario'].split('/')[1]
    pb_filename = os.path.join(global_settings["output_dir"], \
         "multiclass_DNN_w%s_for_%s_%s_%s_%s_%s.pb"\
         %(global_settings["ml_method"], global_settings["channel"], \
          mode, addition, res_nonres, era))
    log_filename = os.path.join(global_settings["output_dir"],\
          "multiclass_DNN_w%s_for_%s_%s_%s_%s.log"\
           %(global_settings["ml_method"], global_settings["channel"], \
           mode, addition, res_nonres))
    cmsml.tensorflow.save_graph(pb_filename, model_structure, variables_to_constants=True)
    ll_var = ['%s_%s' %(part, var) for part in PARTICLE_INFO[global_settings['channel']]\
              for var in ['e', 'px', 'py', 'pz']
              ]
    hl_var = [trainvar for trainvar in trainvars if trainvar not in ll_var]
    file = open(log_filename, "w")
    file.write(str(hl_var))
    file.close()

def define_trainvars(global_settings, preferences, info_dir):
    if global_settings["ml_method"] == "lbn":
        trainvars_path = os.path.join(info_dir, 'trainvars.json')
    if global_settings["dataCuts"].find("boosted") != -1:
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

def create_global_settings(global_settings, channel, res_nonres, mode):
    channel = channel.replace('_bdt', '')
    global_settings['channel'] = channel
    global_settings['ml_method'] = 'lbn'
    if 'nonres' in res_nonres:
        global_settings['scenario'] = '%s/base/%s' %(res_nonres, mode)
    else:
        global_settings['scenario'] = '%s/%s' %(spin, mode)
    global_settings['dataCuts'] = 'cuts_%s.json' %mode
    global_settings['feature_importance'] = 1

def check_trainvar(preferences):
    BMpoints = ['SM', 'BM1', 'BM2', 'BM3', 'BM4', 'BM5', 'BM6', 'BM7', 'BM8', 'BM9', 'BM10', 'BM11', 'BM12']
    for BMpoint in BMpoints:
        if BMpoint in preferences['trainvars']:
            preferences['trainvars'].remove(BMpoint)

if __name__ == '__main__':
    startTime = datetime.now()
    try:
        arguments = docopt.docopt(__doc__)
        print arguments
        channel = arguments['--channel']
        mode = arguments['--mode']
        res_nonres = arguments['--res_nonres']
        era = arguments['--era']
        BM = arguments['--BM']
        split_ggf_vbf = bool(int(arguments['--split_ggf_vbf']))
        sig_weight = float(arguments['--sig_weight'])
        spin = arguments['--spin']
        mass_region = arguments['--mass_region']
        main('None', channel, mode, era, BM, split_ggf_vbf, sig_weight, mass_region)
    except docopt.DocoptExit as e:
        print(e)
    print(datetime.now() - startTime)
