import numpy as np
# import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde


def print_statistics(results):
    """Show some stats.

    :param np.array results: Results to process
    """
    print('STATISTICS:\n')
    print('-- Mean: ' + str(np.mean(results)) + '\n')
    print('-- Median: ' + str(np.median(results)) + '\n')
    print('-- Standard deviation :' + str(np.std(results)) + '\n')
    print('-- 0.05-quantile: ' + str(np.percentile(results, 0.05)) + '\n')
    print('-- 0.95-quantile: ' + str(np.percentile(results, 0.95)) + '\n')
    # skweness + kurtosis


def histogram(results, xlab='Quantity of interest', ylab='Density', title='Histogram with a kernel density estimator'):
    """Plot an Histogram.an

    :param np.array results: Results to process
    :param str xlab: x label
    :param str ylab: y label
    :param str title: Title
    """
    title_font = {'fontname': 'sans-serif', 'size': '16', 'color': 'black',
                  'weight': 'normal', 'verticalalignment': 'bottom'}
    axis_font = {'fontname': 'sans-serif', 'size': '16'}
    # legend_font = fm.FontProperties(family='sans-serif', size=16)
    my_pdf = gaussian_kde(results)
    x = np.linspace(min(results), max(results), 100)
    fig = plt.figure()
    plt.plot(x, my_pdf(x), 'r')
    plt.hist(results, normed=1, alpha=.3)
    plt.title(title, title_font)
    plt.xlabel(xlab, axis_font)
    plt.ylabel(ylab, axis_font)
    fig.tight_layout()
    fig.savefig('./histo_waterlevel.pdf',
                transparent=True, bbox_inches='tight')
    plt.close('all')

# def plotInOut(X,Y):
#   df = DataFrame(np.concatenate(X,Y))
#   scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
