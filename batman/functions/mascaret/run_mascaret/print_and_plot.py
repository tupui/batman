import numpy as np
# import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde


def print_statistics(results):
    """Show some stats.

    :param np.array results: Results to process
    """
    print("STATISTICS:\n"
          "-- Mean: {}\n"
          "-- Median: {}\n"
          "-- Standard deviation {}\n"
          "-- 0.05-quantile: {}\n"
          "-- 0.95-quantile: {}\n")
          .format(np.mean(results), np.median(results), np.std(results),
                  np.percentile(results, 0.05), np.percentile(results, 0.95))


def histogram(results, xlab='Quantity of interest', ylab='Density', title='Histogram with a kernel density estimator'):
    """Plot an Histogram.

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
    fig.savefig('./histogram.pdf',
                transparent=True, bbox_inches='tight')
    plt.close('all')
