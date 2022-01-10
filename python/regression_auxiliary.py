import ROOT 
from root_numpy import tree2array
import os
import sys
import pandas
import numpy as np
import collections

files = collections.OrderedDict()
files = {'bb2l' : '/hdfs/local/veelken/hhAnalysis/2016/2022Jan04/histograms/hh_bbwwMEM_dilepton/histograms_harvested_stage2_hh_bbwwMEM_dilepton.root',
    'bb1l' : '/hdfs/local/veelken/hhAnalysis/2016/2021Oct10_test/histograms/hh_bbwwMEM_singlelepton/histograms_harvested_stage2_hh_bbwwMEM_singlelepton.root'
}

class DataLoader(object):
    def __init__(
            self,
            channel,
            target,
            nbjets,
            nwjets,
            nbjets_medium,
            normalize=True,
            remove_negative_weights=True,
            nr_events_per_file=-1,
            weight='genWeight'
    ):
        
        print('In DataLoader')
        self.channel = channel
        self.target = target
        self.nbjets = nbjets
        self.nwjets = nwjets
        self.nbjets_medium = nbjets_medium
        self.nr_events_per_file = nr_events_per_file
        self.weight = weight
        self.normalize = normalize
        self.remove_negative_weights = remove_negative_weights
        self.data = self.load_data()

    def read_trainvar(self,txtfile):
        trainvar = []
        with open(txtfile, 'r') as file:
            for line in file.readlines():
                trainvar.append(line.strip())
        return trainvar

    def check_trainvars(self):
        for trainvar in self.trainvars:
            if 'gen' in trainvar:
                print('gen level variable %s is present in trainvar' %trainvar)
                sys.exit()
            if  trainvar in ['run', 'lumi', 'event', 'isSignal']:
                print('variable %s is present in trainvar' %trainvar) 
                sys.exit()
            if 'mem' in trainvar:
                print('target variable %s is present in trainvar' %trainvar) 
                sys.exit()
            
    def apply_cut(self, data):
        data = data.loc[(data['nbjets'] == self.nbjets) & (data['nbjets_medium'] == self.nbjets_medium)\
                        & (data['isBoosted_Hbb'] == 0)]
        if 'wjets' in self.allvars:
            data = data.loc[data['nwjets'] == self.nwjets]
        if self.remove_negative_weights:
            data = data.loc[data[self.weight] >=0]
        return data

    def apply_weight(self, data):
        condition_sig = data['isSignal']==1
        factor = 100000/data.loc[condition_sig, [self.weight]].sum()
        data.loc[condition_sig, [self.weight]] *= factor
        condition_bkg = data['isSignal']==0
        factor = 100000/data.loc[condition_bkg, [self.weight]].sum()
        data.loc[condition_bkg, [self.weight]] *= factor
        print('tot sig weight: ' + str(data.loc[condition_sig, [self.weight]].sum()) + ' tot bkg weight: ' + \
              str(data.loc[condition_bkg, [self.weight]].sum()))

    def load_from_file(self, path):
        tfile = ROOT.TFile(path)
        datas = pandas.DataFrame({})
        input_trees = ["hh_bbwwMEM_dilepton/ntuples/background_lo/mem", "hh_bbwwMEM_dilepton/ntuples/signal_lo/mem"]
        for input_tree in input_trees:
            tree = tfile.Get(input_tree)
            chunk_arr = tree2array(
                tree, branches=self.allvars,
                stop=self.nr_events_per_file
            )
            data = pandas.DataFrame(chunk_arr)
            datas = datas.append(data, ignore_index=True, sort=False)
        tfile.Close()
        return datas
                
    def load_data(self):
        path = files[self.channel]
        all_trainvar_file = os.path.join(os.path.expandvars('$CMSSW_BASE'),\
           'src/machineLearning/machineLearning/info/HH/%s/all_trainvars_regression.txt' %self.channel)
        self.allvars = self.read_trainvar(all_trainvar_file)
        self.allvars.extend(['memProbB', 'memProbS'])
        trainvar_file = os.path.join(os.path.expandvars('$CMSSW_BASE'),\
              'src/machineLearning/machineLearning/info/HH/%s/trainvar_regression.txt' %self.channel)
        self.trainvars = self.read_trainvar(trainvar_file)
        self.check_trainvars()
        data = self.load_from_file(path)#pandas.DataFrame(self.load_from_file(path))
        data['logtarget'] = np.sqrt(np.log(data[self.target])*-1)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = data.dropna()
        data = self.apply_cut(data)
        self.apply_weight(data)
        return data
