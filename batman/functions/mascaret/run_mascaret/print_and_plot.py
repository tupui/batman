from io import BytesIO
import numpy as np
# import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.patches import Polygon
from scipy.stats.kde import gaussian_kde


def print_statistics(results):
    """Show some stats.

    :param np.array results: Results to process
    """
    print(("STATISTICS:\n"
           "-- Mean: {}\n"
           "-- Median: {}\n"
           "-- Standard deviation {}\n"
           "-- 0.05-quantile: {}\n"
           "-- 0.95-quantile: {}\n")
           .format(np.mean(results), np.median(results), np.std(results),
                   np.percentile(results, 0.05), np.percentile(results, 0.95)))


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
    my_pdf = gaussian_kde(results.flatten())
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


def read_opt(filename='ResultatsOpthyca.opt'):
    """Read the results :file:`ResultatsOpthyca.opt`.

    :param str filename: path of the results file
    :return: Opt data
    :rtype: np.array
    """
    with open(filename, 'rb') as myfile:
        opt_data = myfile.read().decode('utf8').replace('"', '')

    opt_data = np.genfromtxt(BytesIO(opt_data.encode('utf8')),
                             delimiter=';', skip_header=14)

    return opt_data


def plot_opt(filename='ResultatsOpthyca.opt', xlab='Curvilinear abscissa (m)', ylab1='Water level (m)',
                 ylab2='Flow rate (m3/s)', title='Water level along the open-channel at final time'):
    """Plots results contained in the results file :file:`ResultatsOpthyca.opt`.

    :param str xlab: label x
    :param str ylab1: label y1
    :param str ylab2: label y2
    :param str title: Title
    """
    opt_data = read_opt(filename)

    nb = int(max(opt_data[:, 2]))
    x = opt_data[-nb:-1, 3]
    level = opt_data[-nb:-1, 5]
    bathy = opt_data[-nb:-1, 4]
    flowrate = opt_data[-nb:-1, -1]

    fig, ax1 = plt.subplots()
    ax1.plot(x, bathy, color='black')
    ax1.plot(x, level, color='blue')
    ax1.fill_between(x, bathy, level, facecolor='blue', alpha=0.5)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab1, color='blue')
    ax1.tick_params('y', colors='blue')
    y_formatter = tick.ScalarFormatter(useOffset=False)
    ax1.yaxis.set_major_formatter(y_formatter)
    ax2 = ax1.twinx()
    ax2.plot(x, flowrate, color='red')
    ax2.set_ylabel(ylab2, color='red')
    ax2.tick_params('y', colors='red')
    ax2.yaxis.set_major_formatter(y_formatter)
    plt.title(title)
    fig.tight_layout()
    fig.savefig('./waterlevel.pdf', transparent=True, bbox_inches='tight')
    plt.close('all')
