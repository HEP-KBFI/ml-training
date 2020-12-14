""" Tools for visualizing the data """
import os
import numpy as np
import matplotlib
import ROOT
ROOT.gROOT.SetBatch(True)
matplotlib.use('agg')
import matplotlib.pyplot as plt
from machineLearning.machineLearning import CMS_lumi
from machineLearning.machineLearning import tdrstyle


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
            'event', 'era', 'target', 'process', 'key', '*Weight', 'nodeX*']
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
        self.distributions_dir = os.path.join(self.output_dir, 'distributions')
        self.correlations_dir = os.path.join(self.output_dir, 'correlations')


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

    def plot_distribution(self):
        """ Distribution plotting stub"""
        raise NotImplementedError('Please define distribution plotting')

    def plot_correlations(self):
        """ Creates correlation matrices for all different targets and
        a total data correlation matrix for all the features"""
        for class_ in self.classes:
            mode_data = self.data.loc[self.data['target'] == class_]
            self.plot_single_mode_correlation(
                mode_data, self.correlations_dir, self.mapping[class_])
        self.plot_single_mode_correlation(
            self.data, self.correlations_dir, 'total'
        )

    def plot_single_mode_correlation(self, data, output_dir, addition):
        """ Single mode correlation plotting stub"""
        raise NotImplementedError(
            'Please define single mode distribution plotting')

    def visualize_data(self):
        """ Collects all the visualizers """
        if not os.path.exists(self.distributions_dir):
            os.makedirs(self.distributions_dir)
        self.plot_distributions()
        if not os.path.exists(self.correlations_dir):
            os.makedirs(self.correlations_dir)
        self.plot_correlations()


class MPLDataVisualizer(object):
    """ Class for visualizing the data using matplotlib"""

    def __init__(
            self, data, output_dir, target='target', weight='totalWeight'
    ):
        super(MPLDataVisualizer, self).__init__(
            data, output_dir, target=target, weight=weight
        )

    def plot_distributions(self):
        """ Creates the distribution plots for all the features separated
        into the classes given in the target column """
        for feature in self.features:
            fig = plt.figure()
            ax = plt.subplot(111)
            bin_edges = np.histogram(self.data[feature], bins=25)[1]
            for class_ in self.classes:
                class_data = self.data.loc[
                    self.data[self.target] == class_, feature]
                weights = self.data.loc[
                    self.data[self.target] == class_, self.weight]
                ax.hist(
                    class_data,
                    bins=bin_edges,
                    histtype='bar',
                    alpha=0.4,
                    label=self.mapping[class_],
                    weights=weights
                )
            ax.legend()
            plot_out = os.path.join(self.distributions_dir, feature + '.png')
            plt.title(feature)
            plt.savefig(plot_out, bbox_inches='tight')
            plt.close('all')

    def plot_single_mode_correlation(self, data, output_dir, addition):
        """ Creates the correlation matrix for one specific target or for
        the sum of it

        Args:
            data: pandas.DataFrame
                dataframe containing all the data
            output_dir: str
                Path of the output directory for the correlations
            addition: str
                String that specifies the data class and is added to the
                end of the file name
        """
        correlations = data[self.features].corr()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap='viridis')
        ticks = np.arange(0, len(self.features), 1)
        plt.rc('axes', labelsize=8)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.features, rotation=-90)
        ax.set_yticklabels(self.features)
        fig.colorbar(cax)
        fig.tight_layout()
        plot_out = os.path.join(
            output_dir, str(addition) + '_correlations.png')
        plt.savefig(plot_out, bbox_inches='tight')
        plt.close('all')


class ROOTDataVisualizer(DataVisualizer):
    """ Class for visualizing the data using ROOT """
    def __init__(
            self, data, output_dir, target='target', weight='totalWeight',
            suffixes=['.pdf', '.root', '.png'], logy=False,
            analysis='hh-multilepton', channel='channelName'
    ):
        super(ROOTDataVisualizer, self).__init__(
            data, output_dir, target=target, weight=weight
        )
        self.W = 800
        self.H = 600
        self.T = 0.08 * self.H
        self.B = 0.12 * self.H
        self.L = 0.12 * self.W
        self.R = 0.04 * self.W
        self.suffixes = suffixes
        self.logy = logy
        self.histstyles = {}
        self.set_histogram_style()
        self.analysis = analysis
        self.channel = channel

    def set_histogram_style(self):
        """ Creates the histogram styles for different classes """
        pairs = [
            {
                'FillColor': ROOT.kRed,
                'FillStyle': 3006,
                'LineWidth': 1,
                'LineColor': 0
            },
            {
                'FillColor': ROOT.kBlue,
                'FillStyle': 3008,
                'LineWidth': 1,
                'LineColor': 0
            },
            {
                'FillColor': ROOT.kGreen,
                'FillStyle': 3004,
                'LineWidth': 1,
                'LineColor': 0
            },
            {
                'FillColor': ROOT.kOrange,
                'FillStyle': 3005,
                'LineWidth': 1,
                'LineColor': 0
            }
        ]
        if len(self.classes) == 2:
            self.histstyles['signal'] = {
                'FillColor': ROOT.kWhite, 'FillStyle': 3001,
                'LineWidth': 3, 'LineColor': ROOT.kRed
            }
            self.histstyles['background'] = pairs[1]
        else:
            for i, class_ in enumerate(self.classes):
                self.histstyles[class_] = pairs[i]


    def plot_distributions(self):
        """ Creates the distribution plots for all the features separated
        into the classes given in the target column """
        for feature in self.features:
            canvas = ROOT.TCanvas(feature, feature, 100, 100, self.W, self.H)
            self.modify_canvas(canvas)
            feature_min = min(self.data[feature])
            feature_max = max(self.data[feature])
            data_dict = {}
            for class_ in self.classes:
                histogram = ROOT.TH1F(
                    self.mapping[class_], self.mapping[class_],
                    25, feature_min, feature_max
                )
                class_data = self.data.loc[
                    self.data[self.target] == class_, feature]
                weights = self.data.loc[
                    self.data[self.target] == class_, self.weight]
                for event, weight in zip(class_data, weights):
                    histogram.Fill(event, weight)
                histogram.SetFillColor(
                    self.histstyles[self.mapping[class_]]['FillColor'])
                histogram.SetFillStyle(
                    self.histstyles[self.mapping[class_]]['FillStyle'])
                histogram.SetLineColor(
                    self.histstyles[self.mapping[class_]]['LineColor'])
                histogram.SetLineWidth(
                    self.histstyles[self.mapping[class_]]['LineWidth'])
                histogram.SetTitle("; %s; normalized Entries" % (feature))
                histogram.GetXaxis().SetLabelSize(0.35)
                histogram.GetYaxis().SetLabelSize(0.35)
                data_dict[class_] = histogram
                histogram.DrawNormalized('histsame')
            box = ROOT.TBox(0.5, 0., len(self.data), canvas.GetUymax())
            if self.logy:
                box = ROOT.TBox(
                    0.5, 0., len(self.data),
                    ROOT.TMath.Power(10, canvas.GetUymax())
                )
            box.SetFillColorAlpha(ROOT.kGray, 0.5)
            box.Draw("same")
            legend = ROOT.TLegend(0.8, 0.75, 0.9, 0.9)
            legend.SetNColumns(1)
            legend.SetFillStyle(0)
            for class_ in self.classes:
                hist = data_dict[class_]
                if hist.Integral() > 0:
                    legend.AddEntry(hist, hist.GetName(), "F")
            legend.Draw()
            CMS_lumi.lumi_sqrtS = '137.2 fb^-1 @ 13 TeV | %s:%s' %(
                self.analysis, self.channel
            )
            CMS_lumi.CMS_lumi(canvas, 0, 0)
            ROOT.gPad.SetTicks(1, 1)
            for suffix in self.suffixes:
                output_path = os.path.join(
                    self.distributions_dir, feature + suffix)
                canvas.SaveAs(output_path)
            canvas.Close()

    def modify_canvas(self, canvas):
        canvas.SetFillColor(0)
        canvas.SetBorderMode(0)
        canvas.SetFrameFillStyle(0)
        canvas.SetFrameBorderMode(0)
        canvas.SetLeftMargin(self.L / self.W)
        canvas.SetRightMargin(self.R / self.W)
        canvas.SetTopMargin(self.T / self.H)
        canvas.SetBottomMargin(self.B / self.H)
        canvas.SetTickx(0)
        canvas.SetTicky(0)
        canvas.SetGrid()
        canvas.cd()
        if self.logy:
            canvas.SetLogy()
        CMS_lumi.cmsText = "CMS"
        CMS_lumi.extraText = "       Preliminary"
        CMS_lumi.cmsTextSize = 0.65
        CMS_lumi.outOfFrame = True
        tdrstyle.setTDRStyle()

    def plot_single_mode_correlation(self, data, output_dir, addition):
        """ Creates the correlation matrix for one specific target or for
        the sum of it

        Args:
            data: pandas.DataFrame
                dataframe containing all the data
            output_dir: str
                Path of the output directory for the correlations
            addition: str
                String that specifies the data class and is added to the
                end of the file name
        """
        print('Plotting correlations with ROOT currently not available')

