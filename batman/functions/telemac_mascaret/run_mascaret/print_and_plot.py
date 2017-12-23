"""Some statistics and plots."""
from io import BytesIO
import numpy as np
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


def histogram(results, xlab='Quantity of interest', ylab='Density',
              title='Histogram with a kernel density estimator'):
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


def plot_opt_time(filename='ResultatsOpthyca.opt', xlab='Time (s)',
                  ylab1='Water level (m)', ylab2='Flow rate (m3/s)',
                  numabscurv=200):
    """Plot results contained in the results file :file:`ResultatsOpthyca.opt`.
       with 463 output points (Garonne specific) """

    opt_data = read_opt(filename)
    t = opt_data[0:-1:463, 0]
    z = opt_data[:, 5]
    q = opt_data[:, -1]
    z = np.reshape(z, (-1, 463))
    q = np.reshape(q, (-1, 463))

    fig, ax1 = plt.subplots()
    ax1.plot(t, z[:, numabscurv], color='blue')
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab1, color='blue')
    ax1.tick_params('y', colors='blue')
    y_formatter = tick.ScalarFormatter(useOffset=False)
    ax1.yaxis.set_major_formatter(y_formatter)
    ax2 = ax1.twinx()
    ax2.plot(t, q[:, numabscurv], color='red')
    ax2.set_ylabel(ylab2, color='red')
    ax2.tick_params('y', colors='red')
    ax2.yaxis.set_major_formatter(y_formatter)
    title = 'Water level over time at location number '+str(numabscurv)
    plt.title(title)
    fig.tight_layout()
    filefig = 'waterlevel_discharge_time_x='+str(numabscurv)+'.pdf'
    fig.savefig(filefig, transparent=True, bbox_inches='tight')
    plt.close('all')


def plot_opt(filename='ResultatsOpthyca.opt', xlab='Curvilinear abscissa (m)',
             ylab1='Water level (m)', ylab2='Flow rate (m3/s)',
             title='Water level along the open-channel at final time'):
    """Plot results contained in the results file :file:`ResultatsOpthyca.opt`.

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


def read_storage(filename='Resultats.res_casier'):
    """Read the storage results :file:`Resultats.res_casier`.

    :param str filename: path of the storage area results file
    :return: res casier file data
    :rtype: np.array
    """
    with open(filename, 'rb') as myfile:
        storage_data = myfile.read().decode('utf8').replace('"', '')

    storage_data = np.genfromtxt(BytesIO(storage_data.encode('utf8')),
                                 delimiter=';', skip_header=6)

    print(storage_data)
    return storage_data


def plot_storage(filename='Resultats.res_casier', tlab='time (s)', xlab='Storage area',
                 ylab1='Water level (m)', ylab2='Volume (m3)',
                 titlez='Water level in storage area',
                 titlev='Volume in storage area'):
    """Plot results contained in the results file for storage area:file:`Resultats.res_casier` that contains 15 storage areas (Garonne specific).

    :param str xlab: label x
    :param str ylab1: label y1
    :param str ylab2: label y2
    :param str title: Title
    """
    storage_data = read_storage(filename)

    # num = np.array(range(15)) + 1
    t = storage_data[0:-1:15, 0]
    z = storage_data[:, 2]
    zsto1 = z[0:-1:15]
    zsto2 = z[1:-1:15]
    zsto3 = z[2:-1:15]
    zsto4 = z[3:-1:15]
    zsto5 = z[4:-1:15]
    zsto6 = z[5:-1:15]
    zsto7 = z[6:-1:15]
    zsto8 = z[7:-1:15]
    zsto9 = z[8:-1:15]
    zsto10 = z[9:-1:15]
    zsto11 = z[10:-1:15]
    zsto12 = z[11:-1:15]
    zsto13 = z[12:-1:15]
    zsto14 = z[13:-1:15]
    zsto15 = z[14:-1:15]
    zsto15 = np.append(zsto15, z[-1])
    v = storage_data[:, 4]
    vsto1 = v[0:-1:15]
    vsto2 = v[1:-1:15]
    vsto3 = v[2:-1:15]
    vsto4 = v[3:-1:15]
    vsto5 = v[4:-1:15]
    vsto6 = v[5:-1:15]
    vsto7 = v[6:-1:15]
    vsto8 = v[7:-1:15]
    vsto9 = v[8:-1:15]
    vsto10 = v[9:-1:15]
    vsto11 = v[10:-1:15]
    vsto12 = v[11:-1:15]
    vsto13 = v[12:-1:15]
    vsto14 = v[13:-1:15]
    vsto15 = v[14:-1:15]
    vsto15 = np.append(vsto15, v[-1])

    fig, ax1 = plt.subplots()
    l1, = ax1.plot(t, zsto1, color='black')
    l2, = ax1.plot(t, zsto2, color='black')
    l3, = ax1.plot(t, zsto3, color='black')
    l4, = ax1.plot(t, zsto4, color='black')
    l5, = ax1.plot(t, zsto5, color='black')
    l6, = ax1.plot(t, zsto6, color='black')
    l7, = ax1.plot(t, zsto7, color='yellow')
    l8, = ax1.plot(t, zsto8, color='orange')
    l9, = ax1.plot(t, zsto9, color='red')
    l10, = ax1.plot(t, zsto10, color='magenta')
    l11, = ax1.plot(t, zsto11, color='purple')
    l12, = ax1.plot(t, zsto12, color='blue')
    l13, = ax1.plot(t, zsto13, color='green')
    l14, = ax1.plot(t, zsto14, color='brown')
    l15, = ax1.plot(t, zsto15, color='brown')
    ax1.plot(t, zsto15, color='grey')
    ax1.set_xlabel(tlab)
    ax1.set_ylabel(ylab1, color='black')
    ax1.tick_params('y', colors='black')
    y_formatter = tick.ScalarFormatter(useOffset=False)
    ax1.yaxis.set_major_formatter(y_formatter)
    plt.legend([l1, l2, l3, l4, l5, l6, l7, l8, l9,
                l10, l11, l12, l13, l14, l15],
               ["1", "2", "3", "4", "5", "6", "7", "8", "9",
                "10", "11", "12", "13", "14", "15"])
    plt.title(titlez)
    fig.tight_layout()
    fig.savefig('./test_sto-z.pdf', transparent=True, bbox_inches='tight')
    plt.close('all')

    fig, ax2 = plt.subplots()
    # ax2 = ax1.twinx()
    l1, = ax2.plot(t, vsto1, color='black')
    l2, = ax2.plot(t, vsto2, color='black')
    l3, = ax2.plot(t, vsto3, color='black')
    l4, = ax2.plot(t, vsto4, color='black')
    l5, = ax2.plot(t, vsto5, color='black')
    l6, = ax2.plot(t, vsto6, color='black')
    l7, = ax2.plot(t, vsto7, color='yellow')
    l8, = ax2.plot(t, vsto8, color='orange')
    l9, = ax2.plot(t, vsto9, color='red')
    l10, = ax2.plot(t, vsto10, color='magenta')
    l11, = ax2.plot(t, vsto11, color='purple')
    l12, = ax2.plot(t, vsto12, color='blue')
    l13, =  ax2.plot(t, vsto13, color='green')
    l14, = ax2.plot(t, vsto14, color='brown')
    l15, =  ax2.plot(t, vsto15, color='grey')
    ax2.set_ylabel(ylab2, color='black')
    ax2.tick_params('y', colors='black')
    ax2.yaxis.set_major_formatter(y_formatter)
    plt.legend([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10,
                l11, l12, l13, l14, l15],
               ["1", "2", "3", "4", "5", "6", "7", "8", "9",
                "10", "11", "12", "13", "14", "15"])
    plt.title(titlev)
    fig.tight_layout()
    fig.savefig('./test_sto-v.pdf', transparent=True, bbox_inches='tight')
    plt.close('all')


def tecplot_reader(file, nb_var):
    """Tecplot reader.

    :param str file: file path
    :param int nb_var: number of variables to extract
    :return: Extracted variables
    :rtype: array_like shape (n_features, data)
    """
    arrays = []
    append = arrays.append
    with open(file, 'r') as a:
        for idx, line in enumerate(a.readlines()):
            if idx < 3:
                continue
            else:
                append([float(s) for s in line.split()])

    arrays = np.concatenate(arrays)
    output = np.split(arrays, nb_var)

    return output


def plot_pdf(filename='pdf.dat', xlab='Curvilinear abscissa (m)',
             ylab1='Water level (m)', title='pdf'):
    """Plot results contained in the results file :file:`pdf.dat`."""
    z = {'name': "Z", 'label': r"$Z$ (m)", 'data': None, 'shape': 463}
    x_pdf, z['data'], pdf = tecplot_reader(filename, 3)
