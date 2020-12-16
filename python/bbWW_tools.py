from machineLearning.machineLearning.hh_tools import HHDataLoader
import os
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning.data_loader import DataLoader as dlt

class bbWWLoader(HHDataLoader):
    def __init__(
            self, data_normalizer, preferences, global_settings,
            nr_events_per_file=-1, weight='totalWeight',
            cancelled_trainvars=['gen_mHH'], normalize=True,
            reweigh=True, remove_negative_weights=True
    ):
        print('Using bbWW flavor of the HHDataLoader')
        super(bbWWLoader, self).__init__(
            data_normalizer, preferences, global_settings, nr_events_per_file,
            weight, cancelled_trainvars, normalize, reweigh,
            remove_negative_weights
        )

    def update_to_be_dropped_list(self, process):
        if 'nonres' in self.global_settings['scenario']:
            self.nonres_weights = [
                str('Weight_') + scenario
                for scenario in self.preferences['nonResScenarios']
            ]
            if 'signal' in process:
                if 'Base' in self.preferences['nonResScenarios']:
                    self.to_be_dropped.append('Weight_Base')
                    self.to_be_loaded.append('Weight_SM')
                else:
                    self.to_be_loaded.extend(self.nonres_weights)
            self.to_be_dropped.extend(
                list(self.preferences['nonResScenarios']))
            self.to_be_dropped.extend(['nodeX'])
        else:
            self.to_be_dropped.extend(['gen_mHH'])
        self.to_be_loaded.append('isHbb_boosted')

    def process_data_imputer(
            self, chunk_df, folder_name, target, data
    ):
        chunk_df.loc[chunk_df["process"].isin(
            ["TTW", "TTWW", "WW", "WZ", "ZZ", "TTH", "TH", "VH", "Other"]
        ), "process"] = "Other"
        chunk_df.loc[
            chunk_df["process"].str.contains('signal'),
            "process"
        ] = "signal_HH"
        if 'nonres' not in self.global_settings['scenario']:
            data = self.resonant_data_imputer(
                chunk_df, folder_name, target, data)
        else:
            data = self.nonresonant_data_imputer(
                chunk_df, target, data)
        return data

    def get_ntuple_paths(self, input_path, folder_name, file_type='hadd*Tight.root'):
       paths = []
       background_catfile = os.path.join(
            os.path.expandvars('$CMSSW_BASE'),
            'src/machineLearning/machineLearning/info',
            'HH',
            'background_categories.json'
        )
       background_categories = ut.read_json_cfg(background_catfile)
       if 'signal' not in folder_name and folder_name in background_categories.keys():
           bkg_elements = background_categories[folder_name]
           for bkg_element in bkg_elements:
               bkg_element_paths = self.find_paths_both_conventions(
                   input_path, bkg_element, file_type=file_type)
               print('--------------')
               print(bkg_element)
               print(bkg_element_paths)
               paths.extend(bkg_element_paths)
       else:
           paths = self.find_paths_both_conventions(
               input_path, folder_name + '*', file_type=file_type)
       return paths
    
    def load_data_from_tfile(
        self,
        process,
        folder_name,
        target,
        path,
        input_tree
    ):
        if 'TT' in folder_name:
          self.nr_events_per_file = 3000000 
        return dlt.load_data_from_tfile(
            self,
            process,
            folder_name,
            target,
            path,
            input_tree
          )
