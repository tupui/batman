"""
Response surface 4D
---------------------

Define function related to design of experiments.

* :func:`response surface 4d`.
"""
from itertools import combinations_with_replacement
import numpy as np
from scipy.interpolate import griddata
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import batman as bat

def response_surface_4d(bounds, sample=None, data=None, fun=None, doe=None,
                        resampling=0, xdata=None, axis_disc=None, flabel='F', 
                        plabels=None, featorder=None, ticks_nbr=None, fname=None):
    """Response surface visualization in 2d (image), 3d (movie) or 4d (movies)

    You have to set either (i) :attr:`sample` with :attr:`data` or  (ii)
    :attr:`fun` depending on your data. If (i), the data are interpolated
    on a mesh in order to be plotted as a surface. Otherwize, :attr:`fun` is
    directly used to generate correct data.

    The DoE can also be plotted by setting :attr:`doe` along with
    :attr:`resampling`.

    :param array_like bounds: sample boundaries
    ([min, n_features], [max, n_features]).
    :param array_like sample: sample (n_samples, n_features).
    :param array_like data: function evaluations(n_samples, [n_features]).
    :param callable fun: function to plot the response from.
    :param array_like doe: design of experiment (n_samples, n_features).
    :param int resampling: number of resampling points.
    :param array_like xdata: 1D discretization of the function (n_features,).
    :param array_like axis_disc: discretisation of the sample on each axis
    (n_features).
    :param str flabel: name of the quantity of interest.
    :param list(str) plabels: parameters' labels.
    :param array_like featorder: order of features for multi-dimensional plot
    (n_features).
    :param int ticks_nbr: number of color isolines for response surfaces. 
    :param str fname: wether to export to filename or display the figures.
    :returns: figure.
    :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
    """

    dim = len(bounds[0])
    if dim == 1:
        n_samples = 50
    elif dim == 2:
        n_samples = 625
    elif dim == 3:
        n_samples = 8000
    elif dim ==4:
        n_samples = 50625
    n_samples = int(np.floor(np.power(n_samples, 1.0 / dim)))

    grids = [np.linspace(bounds[0][i], bounds[1][i], n_samples) for i in range(dim)]

    if dim == 2:
        grids = np.meshgrid(*grids)
        xsample, ysample = grids
        xsample = xsample.flatten()
        ysample = ysample.flatten()
    if dim == 3:
        grids = np.meshgrid(*grids)
        xsample, ysample, zsample = grids
        xsample = xsample.flatten()
        ysample = ysample.flatten()
        zsample = zsample.flatten()
    if dim == 4:
        grids = np.meshgrid(*grids)
        xsample, ysample, zsample, zzsample = grids
        xsample = xsample.flatten()
        ysample = ysample.flatten()
        zsample = zsample.flatten()
        zzsample = zzsample.flatten()

    if axis_disc is None:
        axis_disc = [n_samples for i in range(dim)]

    if fun is not None:
        data = fun(np.stack([grid.flatten() for grid in grids]).T)

    if xdata is not None:
        data = np.trapz(data[:], xdata) / (np.max(xdata) - np.min(xdata))

    if fun is None:
        data = griddata(sample, data, tuple(grids), method='nearest')

    data = data.flatten()

    if plabels is None:
        plabels = ["x" + str(i) for i in range(dim)]

    if featorder is not None:
        isample = []
        old_labels = []
        for i in range(dim):
            isample.append(0)
            old_labels.append(0)
            old_labels[i] = plabels[i]
        isample[0] = xsample
        isample[1] = ysample
        if dim > 2:
            isample[2] = zsample
        if dim > 3:
            isample[3] = zzsample

        for i in range(dim):
            if (featorder[0]==(i+1)):
                xsample = isample[i]
                plabels[0] = old_labels[i]
            elif (featorder[1]==(i+1)):
                ysample = isample[i]
                plabels[1] = old_labels[i]
            elif (featorder[2]==(i+1)):
                zsample = isample[i]
                plabels[2] = old_labels[i]
            elif (featorder[3]==(i+1)):
                zzsample = isample[i]
                plabels[3] = old_labels[i]

    if ticks_nbr is None:
        ticks_nbr = 10

    fig = plt.figure('Response Surface')
    n_movies = 1
    n_plot = 1

    if (dim == 1):
        plt.plot(grids[0], data)
        plt.ylabel(flabel, fontsize=28)
        plt.xlabel(plabels[0], fontsize=28)
        plt.tick_params(axis='x', labelsize=28)
        plt.tick_params(axis='y', labelsize=28)
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close('all')

    else:

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support')
        writer = FFMpegWriter(fps=5, metadata=metadata)
        if fname is not None:
            movie_name = fname
        else:
            movie_name = 'Response_surface'

        if dim > 2:
            min_z = np.min(zsample)
            max_z = np.max(zsample)
            z_step = (max_z - min_z) / (axis_disc[2] - 1)
            n_plot = axis_disc[2]
        if dim == 4:
            min_zz = np.min(zzsample)
            max_zz = np.max(zzsample)
            zz_step = (max_zz - min_zz) / (axis_disc[3] - 1)
            n_movies = axis_disc[3]

        for movie in range(n_movies):

            with writer.saving(fig, movie_name + '_' + str(movie) + '.mp4', 100):
                for plot in range(n_plot):

                    plt.clf()
                    vticks = np.linspace(np.min(data), np.max(data), num=ticks_nbr)
                    if dim == 4:
                        msk1 = [(zzsample[i] == (min_zz + movie * zz_step)) for i, _ in enumerate(zzsample)]
                        msk2 = [(zsample[i] == (min_z + plot * z_step)) for i, _ in enumerate(zsample)]
                        msk_total = [(msk1[i] and msk2[i]) for i, _ in enumerate(zsample)]
                    elif dim == 3:
                        msk_total = [(zsample[i] == (min_z + plot * z_step)) for i, _ in enumerate(zsample)]
                    else:
                        msk_total = [True] * len(ysample)

                    xsample_plot = [(xsample[i]) for i, _ in enumerate(xsample) if msk_total[i]]
                    ysample_plot = [ysample[i] for i, _ in enumerate(ysample) if msk_total[i]]
                    data_plot = [data[i] for i, _ in enumerate(data) if msk_total[i]]

                    if dim == 3:
                        plt.title(plabels[2] + ' = ' + str(min_z + plot * z_step), loc='left')
                    elif dim == 4:
                        plt.title(plabels[3] + ' = ' + str(min_zz + movie * zz_step) + '\n' +
                                  plabels[2] + ' = ' + str(min_z + plot * z_step), loc='left')

                    Ccont = plt.tricontour(xsample_plot, ysample_plot, data_plot,
                            levels=[1.2], colors=('w',), linestyles=('-',), linewidths=(1,))
                    C = plt.tricontourf(xsample_plot, ysample_plot, data_plot,
                            vticks, antialiased=True, cmap=cm.viridis)

                    #if doe is not None:
                    #    doe = np.asarray(doe)
                    #    len_sampling = len(doe) - resampling
                    #    plt.plot(doe[:, 0][0:len_sampling], doe[:, 1][0:len_sampling], 'ko')
                    #    plt.plot(doe[:, 0][len_sampling:], doe[:, 1][len_sampling:], 'r^')

                    plt.xlabel(plabels[0])
                    plt.ylabel(plabels[1])
                    plt.clabel(Ccont, fmt='%2.1d', colors='w')
                    plt.tick_params(axis='x')
                    plt.tick_params(axis='y')
                    cbar = plt.colorbar(ticks=vticks)
                    cbar.set_label(flabel)
                    cbar.ax.tick_params()

                    if dim == 2:
                        if fname is not None:
                            plt.savefig(fname + str(plot) + '.pdf', transparent=True, bbox_inches='tight')
                        else:
                            plt.show()
                        plt.close('all')
                    
                    writer.grab_frame()

        return fig
