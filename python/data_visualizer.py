""" Tools for visualizing the data """
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class DataVisualizer(object):
    """ Class for visualizing the data"""
    def __init__(
            self, data, output_dir, target='target', weight='totalWeight'
    ):
        """ Initializes the data visalization class
        Args:
            data: pandas.DataFrame
                The data to be visualized
            output_dir: str
                Path to the directory where the plot are outputted
            target: str
                Column name which separates different classes.
                (default: 'target')
            weight: str
                Column name with the event weights
                (default: 'totalWeight')
        """
        self.data = data
        self.target = target
        self.excluded_features = [
            'event', 'era', 'target', 'process', 'key', '*Weight']
        self.features = list(data.columns)
        self.classes = set(self.data[target])
        self.choose_features_for_plotting()
        self.output_dir = output_dir
        self.weight = weight
        if len(self.classes) == 2:
            self.mapping = {
                1: 'signal',
                0: 'background'
            }
        else:
            self.mapping = {class_: class_ for class_ in self.classes}

    def choose_features_for_plotting(self):
        """ Chooses features for plotting. The reason for iteration the
        feature list in opposite order is because when removing an element
        then the indices of the elements change, thus one might skip some
        elements accidentally
        """
        for exclusion in self.excluded_features:
            if '*' not in exclusion:
                if exclusion in self.features:
                    self.features.remove(exclusion)
            else:
                exclusion = exclusion.replace('*', '')
                last_index = len(self.features) - 1
                for idx in range(last_index, 0, -1):
                    feature = self.features[idx]
                    if exclusion in feature:
                        self.features.remove(feature)

    def distribution_plots(self):
        """ Creates the distribution plots for all the features separated
        into the classes given in the target column """
        distributions = os.path.join(self.output_dir, 'distributions')
        if not os.path.exists(distributions):
            os.makedirs(distributions)
        for feature in self.features:
            fig = plt.figure()
            ax = plt.subplot(111)
            bin_edges = np.histogram(self.data[feature], bins=25)[1]
            for class_ in self.classes:
                class_data = self.data.loc[
                    data[self.target] == class_, feature]
                weights = self.data.loc[
                    data[self.target] == class_, self.weight]
                ax.hist(
                    class_data,
                    bins=bin_edges,
                    histtype='bar',
                    alpha=0.4,
                    label=self.mapping[class_],
                    weights=weights
                )
            ax.legend()
            plot_out = os.path.join(distributions, feature + '.png')
            plt.title(feature)
            plt.savefig(plot_out, bbox_inches='tight')
            plt.close('all')
