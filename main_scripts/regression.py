import os
import sys
import pandas
import ROOT
import argparse
import keras
import collections
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Nadam
from sklearn.metrics import r2_score
from sklearn.metrics import auc
import numpy as np
import xgboost as xgb
import tensorflow as tf
import math
import getpass
import cmsml
from sklearn.metrics import roc_curve as roc
from machineLearning.machineLearning.visualization import hh_visualization_tools_regression as hhvtre
from machineLearning.machineLearning.visualization import hh_visualization_tools as hhvt
from machineLearning.machineLearning import converter_tools as ct
from machineLearning.machineLearning import nn_tools as nt
from machineLearning.machineLearning import regression_auxiliary as ra
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

tf, tf1, tf_version = cmsml.tensorflow.import_tf()
np.printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument("-m ", dest="method", help="type of training", default='bdt', choices=['nn', 'bdt'])
parser.add_argument("-t ", dest="target", help="type of target", default='memProbS', choices=['memProbB', 'memProbS', 'z'])
parser.add_argument("-nbjet ", dest="nbjet", type=int, help="number of bjet", default=2, choices=[1,2])
parser.add_argument("-nwjet ", dest="nwjet", type=int, help="number of wjet", default=2, choices=[1,2])
parser.add_argument("-nmedium ", dest="nbjets_medium", type=int, help="number of medium bjet", default=2, choices=[1,2])
parser.add_argument("-o ", dest="output_dir", help="name of output dir")
parser.add_argument("-c ", dest="channel", help="which channel", default='bb2l', choices=['bb1l','bb2l'])
parser.add_argument("-e ", dest="evaluate", action='store_true',help="want to evaluate", default=False)
parser.add_argument("-inp ", dest="input_dir",help="name of input directory of model", default='test_nn')

options = parser.parse_args()
method = options.method
target = options.target
nbjets = options.nbjet
nbjets_medium = options.nbjets_medium
nwjets = options.nwjet
output_dir = options.output_dir
channel = options.channel
evaluate = options.evaluate
input_dir = options.input_dir

if not evaluate:
    output_dir = os.path.join("/home", getpass.getuser(), output_dir, method, channel, target, 'nbjet_'+ str(nbjets),\
                              'nbjet_medium_'+str(nbjets_medium))
else:
    output_dir = os.path.join("/home", getpass.getuser(), output_dir, method, channel, 'evaluate', 'nbjet_'+ str(nbjets),\
                              'nbjet_medium_'+str(nbjets_medium))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
parameters = collections.OrderedDict()
parameters = {
    'nn' : {
        'memProbB' : {'epoch':56, 'batch_size':549, 'lr':0.006737946999085467, 'l2':0.0048385278040772375, 'drop_out': 0., 
                      'layer':6, 'node':37},
        'memProbS' : {'epoch':50, 'batch_size':1471, 'lr':0.006737946999085467, 'l2':0.05358655530086876, 'drop_out': 0.,
                      'layer':4, 'node':34}
    },
    'bdt': {
        'memProbS' : {"n_estimators": 797, "subsample": 0.8002377033977673, "colsample_bytree": 1,
                      "gamma": 3.984940910974453, "learning_rate": 0.0070305239398164385, "max_depth": 5,
                      "min_child_weight":496.02471411591904, 'eval_metric': 'rmse', 'silent': 0, 'nthread': 3
                  },
        'memProbB' : {"n_estimators": 392, "subsample": 0.8053470479755194, "colsample_bytree": 1.0,
                      "gamma": 5.742057131774252, "learning_rate": 0.04111776357968012, "max_depth": 6,
                      "min_child_weight":499.861219021751, 'eval_metric': 'rmse', 'silent': 0, 'nthread': 3
                  }
    }
}

def create_data_dict(data, trainvars):
    even_data = data.loc[(data['event'].values % 2 == 0)]
    odd_data = data.loc[~(data['event'].values % 2 == 0)]
    even_data_train = even_data.sample(frac=0.80)
    even_data_val = even_data.drop(even_data_train.index)
    odd_data_train = odd_data.sample(frac=0.80)
    odd_data_val = odd_data.drop(odd_data_train.index)
    trainvars = trainvars
    data_dict = {
        'trainvars': trainvars,
        'odd_data':  odd_data,
        'even_data': even_data,
        'odd_data_train': odd_data_train,
        'odd_data_val': odd_data_val,
        'even_data_train': even_data_train,
        'even_data_val': even_data_val,
    }
    return data_dict

def evaluate_model(model, event):
    session = tf1.Session(
                graph=model,
                config=tf1.ConfigProto(
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1,
                ),
            )
    with model.as_default():
        x1 = model.get_tensor_by_name("input_var:0")
        y = model.get_tensor_by_name("Identity:0")
        pred = session.run(y, {x1: event[trainvars]})
    return pred

def save_xmlFile(output_dir, model, addition, addition2=target):
    model_name = '%s_train_%s.xml' % (addition, addition2)
    xmlFile = os.path.join(output_dir, model_name)
    bst = model.get_booster()
    features = bst.feature_names
    bdtModel = ct.BDTxgboost(model, features, ['%s' %target])
    bdtModel.to_tmva(xmlFile)
    print('.xml BDT model saved to ' + str(xmlFile))

def save_pklFile(output_dir, model, addition, addition2=target):
    model_name = '%s_train_%s.pkl' % (addition, addition2)
    pklFile_path = os.path.join(output_dir, model_name)
    with open(pklFile_path, 'wb') as pklFile:
        pickle.dump(model, pklFile)
    print('.pkl file saved to: ' + str(pklFile_path))

def create_model(data, choose_data):

    train_data = data['%s_train' %choose_data]
    val_data = data['%s_val' %choose_data]
    trainvars = data['trainvars']
    print trainvars
    if method == 'bdt':
        classifier = xgb.XGBRegressor(**parameters[method][target])
        model = classifier.fit(
            train_data[trainvars],
            train_data['logtarget'],
        eval_metric="rmse",
        sample_weight=train_data['genWeight'],
        sample_weight_eval_set= [train_data['genWeight'], val_data['genWeight']],
        eval_set= [(train_data[trainvars], train_data['logtarget']),\
        (val_data[trainvars], val_data['logtarget'])],
        early_stopping_rounds=10
        )
        results = model.evals_result()
        save_pklFile(output_dir, model, choose_data, 'on_%s' %target)
        save_xmlFile(output_dir, model, choose_data, 'on_%s' %target)
        hhvtre.plot_bdt_history(results, output_dir)
        score = model.score(train_data[trainvars], train_data['logtarget'], sample_weight=train_data['genWeight'])
        print("Training score: ", score)
        score = model.score(val_data[trainvars], val_data['logtarget'],sample_weight=val_data['genWeight'])
        print("Testing score: ", score)
    else:
        model = nt.NNmodel(
            train_data,
            val_data,
            trainvars,
            parameters[method][target],
            True,
            model = 'regression',
            output_dir = output_dir,
            weight='genWeight'
        )
        model = model.create_model()
        print model.summary()
        cmsml.tensorflow.save_graph('%s/%s_data_train.pb' %(output_dir, choose_data), model, variables_to_constants=True)
        val_pred = model.predict(val_data[trainvars].values.astype(np.float))
        train_pred = model.predict(train_data[trainvars].values.astype(np.float))
        score = r2_score(train_data['logtarget'], train_pred, sample_weight=train_data['genWeight'])
        print("Training score: ", score)
        score = r2_score(val_data['logtarget'], val_pred, sample_weight=val_data['genWeight'])
        print("Testing score: ", score)
    return model

def plot_ROC(true, regression, issignal, addition):
    reg_fpr, reg_tpr, _ = roc(
        issignal,
        regression,
    )
    true_fpr, true_tpr, _ = roc(
        issignal,
        true
    )
    true_auc = auc(true_fpr, true_tpr, reorder=True)
    reg_auc = auc(reg_fpr, reg_tpr, reorder=True)
    reg_info = {
        'fpr': reg_fpr,
        'tpr': reg_tpr,
        'auc': reg_auc,
        'type': 'reg',
    }
    true_info = {
        'fpr': true_fpr,
        'tpr': true_tpr,
        'auc': true_auc,
        'type': 'true',
    }
    hhvtre.plotROC(
        [true_info, reg_info],
        output_dir, addition
    )

def isgenmatched(gen_eta, gen_phi, rec_eta, rec_phi):
    match_index =-1
    count = 0
    for rec_index in range(len(rec_eta)):
        for gen_index in range(len(gen_eta)):
            if match_index == gen_index: continue
            delta_eta = rec_eta[rec_index] - gen_eta[gen_index]
            delta_phi = rec_phi[rec_index] - gen_phi[gen_index]
            deltaR = math.sqrt(delta_eta**2 + delta_phi**2)
            if deltaR <0.3:
                match_index = gen_index
                count +=1
                break
    return count

def do_genmatch(event):
    event_dict = {}
    for row in range(len(event)):
        rec_bjet_eta = event[row:row+1]['bjet1_eta'].to_list()
        rec_bjet_eta.extend(event[row:row+1]['bjet2_eta'].to_list())
        rec_bjet_phi = event[row:row+1]['bjet1_phi'].to_list()
        rec_bjet_phi.extend(event[row:row+1]['bjet2_phi'].to_list())
        gen_bjet_eta = event[row:row+1]['gen_bjet1_eta'].to_list()
        gen_bjet_eta.extend(event[row:row+1]['gen_bjet2_eta'].to_list())
        gen_bjet_phi = event[row:row+1]['gen_bjet1_phi'].to_list()
        gen_bjet_phi.extend(event[row:row+1]['gen_bjet2_phi'].to_list())
        isgenmatched_n = isgenmatched(gen_bjet_eta, gen_bjet_phi,\
                                      rec_bjet_eta, rec_bjet_phi
        )
        if isgenmatched_n ==2:
            event_dict['genmatched_2'] = event[row:row+1]
        elif isgenmatched_n ==1:
            event_dict['genmatched_1'] = event[row:row+1]
        else:
             event_dict['genmatched_0'] = event[row:row+1]
    return event_dict

def evaluate_mem(model, event, trainvars):
    pred = model.predict(event[trainvars])
    return np.exp(np.square(pred)*-1)

def evaluate_npl(mem_s, mem_b):
    mem_sb = mem_s + mem_b
    return mem_s/(mem_sb) if mem_sb !=0 else 0

def model_performance(model, data, data_type):
    train = 'even_data' if data_type == 'odd_data' else 'odd_data'
    for sig in [2]:
        if sig==2:
            test_sample = data[data_type]
            train_sample = data[train]
        else:
            test_sample = data[data_type].loc[data[data_type]['isSignal']==sig]
        print('len of validation sample, sig: ', str(sig), '\t', len(test_sample))
        true = test_sample['logtarget'].values.astype(np.float)
        pred = model.predict(test_sample[data['trainvars']])
        hhvtre.correlation_true_vs_pred(true, pred.flatten(), output_dir, addition='%s_test_%s_sig_%s' %(target, data_type, str(sig)))
        hhvtre.hist_true_vs_pred(true, pred.flatten(), data[data_type]['genWeight'],\
            output_dir, target, addition='log_test_%s_sig_%s' %(data_type, str(sig)))
        true = train_sample['logtarget'].values.astype(np.float)
        pred = model.predict(train_sample[data['trainvars']])
        hhvtre.correlation_true_vs_pred(true, pred.flatten(), output_dir, addition='%s_train_%s_sig_%s' %(target, train, str(sig)))
        #hhvtre.hist_true_vs_pred(true, pred.flatten(), data[train]['genWeight'],\
            #  output_dir, target, addition='log_train_%s_sig_%s' %(train, str(sig)))

def main():
    dataloader = ra.DataLoader(channel, target, nbjets, nwjets, nbjets_medium)
    data, trainvars = dataloader.data, dataloader.trainvars
    data_dict = create_data_dict(data, trainvars)
    data_type = ['even_data', 'odd_data']
    if not evaluate:
        hhvt.plot_single_mode_correlation(data, trainvars+[target], output_dir,'mem')
        hhvtre.plot_trainvar_multi_distributions(data, trainvars+[target, 'logtarget'], output_dir)
        for choose_data in data_type:
            model = create_model(data_dict, choose_data)
            test_data = [d for d in data_type if d != choose_data][0]
            model_performance(model, data_dict, test_data)
    else:
        for choose_data in data_type:
            max_reg_prob = {}
            max_true_prob = {}
            sum_reg_prob = {}
            sum_true_prob = {}
            issignal = {}
            inputdir = os.path.join("/home", getpass.getuser(), input_dir, method, channel, 'memProbS', 'nbjet_'+ str(nbjets),\
                              'nbjet_medium_'+str(nbjets_medium))#"/home/" + getpass.getuser() + '/' + input_dir + '/' + method + '/'\
                           #+ channel + '/memProbS' + '/nbjets_'+str(nbjets)
            model_s = pickle.load(open('%s/%s_train_on_memProbS.pkl' %(inputdir,choose_data),"rb"))
            inputdir = inputdir.replace('memProbS', 'memProbB')
            model_b =  pickle.load(open('%s/%s_train_on_memProbB.pkl' %(inputdir,choose_data), 'rb'))
            data = data_dict['even_data'] if choose_data == 'odd_data'\
                   else data_dict['odd_data']
            print 'len========== ', len(data)
            data = data.loc[data['gen_nbjets']==2]
            data_bkg = data.loc[data['isSignal']==False]
            data_sig = data.loc[data['isSignal']==True]
            for data_type in [data_sig, data_bkg]:
                prev_run = data_type[0:1]['run'].values
                prev_lumi = data_type[0:1]['ls'].values
                prev_event = data_type[0:1]['event'].values
                prev_index = 0
                count = 0
                for row_index in range(1, len(data_type)):
                    info = data_type[row_index:row_index+1][['run', 'ls', 'event']].values
                    r = info[0][0]
                    l = info[0][1]
                    e = info[0][2]
                    if prev_run == r and prev_lumi == l and prev_event == e:
                        continue
                    else:
                        single_event = data_type[prev_index:row_index]
                        prev_run = r
                        prev_lumi = l
                        prev_event = e
                        prev_index = row_index
                    if len(list(set(single_event['isSignal']))) !=1: continue
                    #print 'run: ', r, '\t', 'lumi: ', l, '\t', 'evt: ', e, '\tlen: ', len(single_event)
                    event_dict = do_genmatch(single_event)
                    if method == 'nn':
                        model = cmsml.tensorflow.load_graph('%s/%s_data_train.pb' %(inputdir,choose_data))
                        pred = evaluate_model(model, single_event)
                        mem_b = max(np.exp(np.square(pred)*-1))
                        inputdir = inputdir.replace('memProbB', 'memProbS')
                        model =  cmsml.tensorflow.load_graph('%s/%s_data_train.pb' %(inputdir,choose_data))
                        pred = evaluate_model(model, single_event)
                        max_mem_s = max(np.exp(np.square(pred)*-1))
                        mem_sb = mem_s+ mem_b
                        prob = mem_s/(mem_sb) if mem_sb !=0 else 0
                        reg_prob.append(prob)
                    else:
                        for key in event_dict.keys():
                            event = event_dict[key]
                            mem_s = evaluate_mem(model_s, event, trainvars)
                            max_mem_s, sum_mem_s = max(mem_s), sum(mem_s)
                            mem_b = evaluate_mem(model_b, event, trainvars)
                            max_mem_b, sum_mem_b = max(mem_b), sum(mem_b)
                            true_s = event['memProbS']
                            true_b = event['memProbB']
                            max_true_s, sum_true_s = max(true_s), sum(true_s)
                            max_true_b, sum_true_b = max(true_b), sum(true_b)
                            if key not in max_reg_prob.keys():
                                max_reg_prob[key] = []
                                sum_reg_prob[key] = []
                                issignal[key] = []
                                max_true_prob[key] = []
                                sum_true_prob[key] = []
                            max_reg_prob[key].append(evaluate_npl(max_mem_s, max_mem_b))
                            sum_reg_prob[key].append(evaluate_npl(sum_mem_s, sum_mem_b))
                            issignal[key].append(list(set(event['isSignal'].to_list())))
                            max_true_prob[key].append(evaluate_npl(max_true_s, max_true_b))
                            sum_true_prob[key].append(evaluate_npl(sum_true_s, sum_true_b))
                        count +=1
                        #if count ==5000:
                         #   break
            for key in  ['genmatched_1', 'genmatched_1', 'genmatched_2']:
                plot_ROC(max_true_prob[key], max_reg_prob[key], issignal[key], '%s_max_%s' %(choose_data,key))
                plot_ROC(sum_true_prob[key], sum_reg_prob[key], issignal[key], '%s_sum_%s' %(choose_data, key))

if __name__ == '__main__':
    main()
