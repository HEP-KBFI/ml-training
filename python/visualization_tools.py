''' Some helpful tools for plotting and data visualization
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_roc_curve(
        output_dir,
        auc_info
):
    '''Creates the ROC plot.

    Parameters:
    ----------
    output_dir : str
        Path to the directory where figures will be saved
    auc_info : dict
        Dictionary containing x and y values for test and train sample from
        ROC calculation

    Returns:
    -------
    Nothing
    '''
    plot_out = os.path.join(output_dir, 'roc.png')
    plt.xlabel('Proportion of false values')
    plt.ylabel('Proportion of true values')
    axis = plt.gca()
    axis.set_aspect('equal')
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    plt.plot(
        auc_info['x_train'], auc_info['y_train'], color='k', linestyle='--',
        label='optimized values, training data', zorder=100
    )
    plt.plot(
        auc_info['x_test'], auc_info['y_test'], color='r', linestyle='-',
        label='optimized values, testing data'
    )
    plt.tick_params(top=True, right=True, direction='in')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(plot_out)
    plt.close('all')


def plot_costfunction(avg_scores, output_dir, y_label='Fitness score'):
    '''Creates a plot of the cost function

    Parameters:
    ----------
    avg_scores : list
        List of average scores of all itereations of the evolutionary algorithm
    output_dir : str
        Path to the directory where the plot is saved

    Returns:
    -------
    Nothing
    '''
    plot_out = os.path.join(output_dir, 'costFunction.png')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        n_gens = len(avg_scores)
        gen_numbers = np.arange(n_gens)
        plt.plot(gen_numbers, avg_scores, color='k')
        plt.xlim(0, n_gens - 1)
        plt.xticks(np.arange(n_gens - 1))
    except:  # in case of a genetic algorithm with multiple subpopulations
        x_max = 0
        for i in avg_scores:
            n_gens = len(avg_scores[i])
            if n_gens > x_max:
                x_max = n_gens
            if i != 'final':
                gen_numbers = np.arange(n_gens)
                plt.plot(gen_numbers, avg_scores[i], color='b')
        for i in avg_scores:
            if len(avg_scores[i]) < x_max and i != 'final':
                line_length = x_max - len(avg_scores[i]) + 1
                y_values = [avg_scores[i][-1] for n in range(line_length)]
                x_values = np.arange(len(avg_scores[i]) - 1, x_max)
                plt.plot(x_values, y_values, color='b', linestyle='--', alpha=0.2)
        n_gens_final = x_max + len(avg_scores['final']) - 1
        gen_numbers = np.arange(x_max - 1, n_gens_final)
        plt.plot(gen_numbers, avg_scores['final'], color='k')
        plt.xlim(0, n_gens_final - 1)
        plt.xticks(np.arange(n_gens_final - 1))
    finally:
        plt.xlabel('Generation')
        plt.ylabel(y_label)
        axis = plt.gca()
        axis.set_aspect('auto', adjustable='box')
        axis.xaxis.set_major_locator(ticker.AutoLocator())
        plt.grid(True)
        plt.tick_params(top=True, right=True, direction='in')
        plt.savefig(plot_out)
        plt.close('all')