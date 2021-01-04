'''
Call with 'python'

Usage:
    hh_nnTraining.py
    hh_nnTraining.py [--save_model=INT --channel=STR --res_nonres=STR --mode=STR --era=INT --BM=INT]

Options:
    -s --save_model=INT             Whether or not to save the model [default: 0]
    -channel --channel=INT          which channel to be considered [default: bb1l]
    -res_nonres --res_nonres=STR        res or nonres to be considered [default: nonres]
    -m --mode=INT                   whhether resolved or boosted catgory to be considered [default: resolved]
    -era --era=INT                era to be processed [default: 2016]
    -BM --BM=STR                    BM point to be considered  [default: None]
'''
import os
import json
from datetime import datetime
import docopt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
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

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

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

def main(output_dir, save_model, channel, mode, era, BM):
    settings_dir = os.path.join(
        os.path.expandvars('$CMSSW_BASE'),
        'src/machineLearning/machineLearning/settings'
    )
    global_settings = ut.read_settings(settings_dir, 'global')
    global_settings['ml_method'] = 'lbn'
    global_settings['channel'] = 'bb1l'
    if output_dir == 'None':
        output_dir = global_settings['channel']+'/'+global_settings['ml_method']+'/'+\
                     res_nonres + '/' + mode +'/' + era
        global_settings['output_dir'] = output_dir
    else:
        global_settings['output_dir'] = output_dir
    global_settings['output_dir'] = os.path.expandvars(
        global_settings['output_dir'])
    if not os.path.exists(global_settings['output_dir']):
        os.makedirs(global_settings['output_dir'])
    channel_dir, info_dir, _ = ut.find_settings()
    scenario = global_settings['scenario']
    reader = hpr.HHParameterReader(channel_dir, scenario)
    preferences = reader.parameters
    if not BM == 'None':
        preferences["nonResScenarios"] = [BM]
    print('BM point to be considered: ' + str(preferences["nonResScenarios"]))
    if not era == '0':
        preferences['included_eras'] = [era.replace('20', '')]
    print('era: ' + str(preferences['included_eras']))
    preferences = define_trainvars(global_settings, preferences, info_dir)
    particles = PARTICLE_INFO[global_settings['channel']]
    data_dict = create_data_dict(preferences, global_settings)
    classes = set(data_dict["even_data"]["process"])
    for class_ in classes:
        multitarget = list(set(
            data_dict["even_data"].loc[
                data_dict["even_data"]["process"] == class_, "multitarget"
            ]
        ))[0]
        print(str(class_) + '\t' + str(multitarget))
    even_model = create_model(
        preferences, global_settings, data_dict, "even_data", save_model)
    if global_settings['feature_importance'] == 1:
        trainvars = preferences['trainvars']
        data = data_dict['odd_data']
        LBNFeatureImportance = nt.LBNFeatureImportances(even_model, data,\
            trainvars, global_settings['channel'])
        score_dict = LBNFeatureImportance.custom_permutation_importance()
        hhvt.plot_feature_importances_from_dict(
            score_dict, global_settings['output_dir'])
    odd_model = create_model(
        preferences, global_settings, data_dict, "odd_data", save_model)
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

def create_data_dict(preferences, global_settings):
    normalizer = bbwwt.bbWWDataNormalizer
    loader = bbwwt.bbWWLoader(
        normalizer,
        preferences,
        global_settings
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
    sumall = data.loc[data["process"] == "TT"]["totalWeight"].sum() \
        + data.loc[data["process"] == "W"]["totalWeight"].sum() \
        + data.loc[data["process"] == "DY"]["totalWeight"].sum() \
        + data.loc[data["process"] == "ST"]["totalWeight"].sum() \
        + data.loc[data["process"] == "Other"]["totalWeight"].sum() \
        + data.loc[data["target"] == 1]["totalWeight"].sum()
    print(
        "TT:W:DY:ST:Other:HH \t" \
        + str("%0.3f" %(data.loc[data["process"] == "TT"]["totalWeight"].sum()/sumall)) \
        + ":" + str("%0.3f" %(data.loc[data["process"] == "W"]["totalWeight"].sum()/sumall)) \
        + ":" + str("%0.3f" %(data.loc[data["process"] == "DY"]["totalWeight"].sum()/sumall)) \
        + ":" + str("%0.3f" %(data.loc[data["process"] == 'ST']["totalWeight"].sum()/sumall)) \
        + ":" + str("%0.3f" %(data.loc[data["process"] == 'Other']["totalWeight"].sum()/sumall)) \
        + ":" + str("%0.3f" %(data.loc[data["target"] == 1]["totalWeight"].sum()/sumall))
    )
    use_Wjet = True
    if 'bb2l' in global_settings['channel']:
        use_Wjet = False
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
        save_model
):
    train_data = data_dict['odd_data'] if choose_data == "odd_data" else data_dict['even_data']
    val_data = data_dict['even_data']  if choose_data == "odd_data" else data_dict['odd_data']
    lbn = global_settings['ml_method'] == 'lbn'
    parameters = {'epoch':20, 'batch_size':1024, 'lr':0.0003, 'l2':0.0003, 'dropout':0, 'layer':3, 'node':256}
    if lbn:
        modeltype = nt.LBNmodel(
            train_data,
            val_data,
            preferences['trainvars'],
            global_settings['channel'],
            parameters,
            True,
            global_settings['output_dir'],
            choose_data
        )
    model = modeltype.create_model()
    if save_model:
        savemodel(model, preferences['trainvars'], global_settings)
    return model

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

def savemodel(model_structure, trainvars, global_settings):
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


if __name__ == '__main__':
    startTime = datetime.now()
    try:
        arguments = docopt.docopt(__doc__)
        print arguments
        save_model = bool(int(arguments['--save_model']))
        channel = arguments['--channel']
        mode = arguments['--mode']
        res_nonres = arguments['--res_nonres']
        era = arguments['--era']
        BM = arguments['--BM']
        main('None', save_model, channel, mode, era, BM)
    except docopt.DocoptExit as e:
        print(e)
    print(datetime.now() - startTime)
