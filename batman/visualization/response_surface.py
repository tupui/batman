"""
Response Surface
---------------------

Define function related to creation of the response surface.

* :func:`response_surface`.
"""
import numpy as np
from scipy.interpolate import griddata
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


def response_surface(bounds, sample=None, data=None, fun=None, doe=None,
                     resampling=0, xdata=None, axis_disc=None, flabel='F',
                     plabels=None, feat_order=None, ticks_nbr=10,
                     range_cbar=None, contours=None, fname=None):
    """Response surface visualization in 2d (image), 3d (movie) or 4d (movies).

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
    :param array_like feat_order: order of features for multi-dimensional plot
        (n_features).
    :param int ticks_nbr: number of color isolines for response surfaces.
    :param array_like range_cbar: min and max values for colorbar range (2).
    :param array_like contours: isocontour values to plot on response surface.
    :param str fname: whether to export to filename or display the figures.
    :returns: figure.
    :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
    """
    # Get the input parameters number (dimension)
    dim = len(bounds[0])

    # Default sample size as a function of dimension
    if dim == 1:
        n_samples = 50
    elif dim == 2:
        n_samples = 625
    elif dim == 3:
        n_samples = 8000
    elif dim == 4:
        n_samples = 50625
    n_samples = int(np.floor(np.power(n_samples, 1.0 / dim)))

    if doe is not None:
        n_samples = n_samples if len(doe) < n_samples else len(doe)

    # If axis discretisation is not given through option axis_disc,
    # apply default discretisation (same discretisation for every dimension).
    if axis_disc is None:
        axis_disc = [n_samples for i in range(dim)]

    grids = [np.linspace(bounds[0][i], bounds[1][i], axis_disc[i]) for i in range(dim)]
    grids = np.meshgrid(*grids)
    list_grid = [grid.flatten() for grid in grids]

    if dim == 2:
        xsample, ysample = list_grid
    if dim == 3:
        xsample, ysample, zsample = list_grid
    if dim == 4:
        xsample, ysample, zsample, zzsample = list_grid

    # Get the data
    if fun is not None:
        data = fun(np.stack([grid.flatten() for grid in grids]).T)

    if xdata is not None:
        data = np.trapz(data[:], xdata) / (np.max(xdata) - np.min(xdata))

    if fun is None:
        data = griddata(sample, data, tuple(grids), method='nearest')

    data = data.flatten()

    # Give a default name for the input parameters
    if plabels is None:
        plabels = ["x" + str(i) for i in range(dim)]

    # Sort the input parameters and sample points
    # get a copy of the unsorted variables to save the data before sorting
    if feat_order is not None:
        isample = [None] * dim
        old_labels = np.copy(plabels)
        old_axis_disc = np.copy(axis_disc)
        if doe is not None:
            doe = np.array(doe)
            old_doe = np.copy(doe)
        isample[0] = xsample
        isample[1] = ysample
        if dim > 2:
            isample[2] = zsample
        if dim > 3:
            isample[3] = zzsample

        # Perform the actual sorting of the variables
        for i in range(dim):
            if feat_order[0] == i + 1:
                xsample = isample[i]
                plabels[0] = old_labels[i]
                axis_disc[0] = old_axis_disc[i]
                if doe is not None:
                    doe[:, 0] = old_doe[:, i]
            elif feat_order[1] == i + 1:
                ysample = isample[i]
                plabels[1] = old_labels[i]
                axis_disc[1] = old_axis_disc[i]
                if doe is not None:
                    doe[:, 1] = old_doe[:, i]
            elif feat_order[2] == i + 1:
                zsample = isample[i]
                plabels[2] = old_labels[i]
                axis_disc[2] = old_axis_disc[i]
                if doe is not None:
                    doe[:, 2] = old_doe[:, i]
            elif feat_order[3] == i + 1:
                zzsample = isample[i]
                plabels[3] = old_labels[i]
                axis_disc[3] = old_axis_disc[i]
                if doe is not None:
                    doe[:, 3] = old_doe[:, i]

    # Default values
    fig = plt.figure('Response Surface')
    n_movies = 1
    n_plot = 1

    if dim == 1:
        # Create the 1D response surface
        plt.plot(grids[0], data)
        plt.ylabel(flabel, fontsize=28)
        plt.xlabel(plabels[0], fontsize=28)
        plt.tick_params(axis='x', labelsize=28)
        plt.tick_params(axis='y', labelsize=28)
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname + '.pdf',
                        transparent=True, bbox_inches='tight')
        else:
            plt.savefig('Response_Function.pdf',
                        transparent=True, bbox_inches='tight')
        plt.close('all')

    else:
        # Create the response surfaces and movies in 2D, 3D and 4D.
        if dim > 2:
            # Animation options
            FFMpegWriter = manimation.writers['ffmpeg']
            metadata = dict(title='Response Surface', artist='Batman',
                            comment=("In 2D, show the response surface. "
                                     "In 3D, display the 3rd variable as a function of time. "
                                     "In 4D, generate a movie for each 4th variable "
                                     "discretisation."))
            writer = FFMpegWriter(fps=5, metadata=metadata)
            if fname is not None:
                movie_name = fname
            else:
                movie_name = 'Response_Surface'

            # Define discretisation parameters for 3rd and 4th dimensions
            min_z = np.min(zsample)
            max_z = np.max(zsample)
            z_step = (max_z - min_z) / (axis_disc[2] - 1)
            n_plot = axis_disc[2]
        if dim == 4:
            min_zz = np.min(zzsample)
            max_zz = np.max(zzsample)
            zz_step = (max_zz - min_zz) / (axis_disc[3] - 1)
            n_movies = axis_disc[3]

        # Loop on the movies
        for movie in range(n_movies):
            if dim > 2:
                writer.setup(fig, movie_name + '_' + str(movie) + '.mp4', 100)

            # Loop on the response surface to create
            for plot in range(n_plot):
                plt.clf()
                if range_cbar is None:
                    vticks = np.linspace(np.min(data), np.max(data),
                                         num=ticks_nbr)
                    range_cbar = [np.min(data), np.max(data)]
                else:
                    vticks = np.linspace(range_cbar[0], range_cbar[1],
                                         num=ticks_nbr)

                # Create masks to apply on the data and sample points, in order
                # to keep only the data to be plotted on the x and y axis
                if dim == 4:
                    msk1 = [(zzsample[i] == (min_zz + movie * zz_step))
                            for i, _ in enumerate(zzsample)]
                    msk2 = [(zsample[i] == (min_z + plot * z_step))
                            for i, _ in enumerate(zsample)]
                    msk_total = [(msk1[i] and msk2[i]) for i, _ in enumerate(zsample)]
                elif dim == 3:
                    msk_total = [(zsample[i] == (min_z + plot * z_step))
                                 for i, _ in enumerate(zsample)]
                else:
                    # Mask is always true in 2D (keep all data)
                    msk_total = [True] * len(ysample)

                # Apply mask on the data to be plotted
                xsample_plot = [xsample[i] for i, _ in enumerate(xsample) if msk_total[i]]
                ysample_plot = [ysample[i] for i, _ in enumerate(ysample) if msk_total[i]]
                data_plot = [data[i] for i, _ in enumerate(data) if msk_total[i]]

                # Title display the 3rd and 4th parameter and their values at screen
                if dim == 3:
                    plt.title(plabels[2] + ' = ' + str(min_z + plot * z_step), loc='left')
                elif dim == 4:
                    plt.title(plabels[3] + ' = ' + str(min_zz + movie * zz_step) + '\n' +
                              plabels[2] + ' = ' + str(min_z + plot * z_step), loc='left')

                # If coutours option activated, generate contours
                if contours is not None:
                    surface_contour = plt.tricontour(xsample_plot,
                                                     ysample_plot, data_plot,
                                                     levels=contours, colors=('w',),
                                                     linestyles=('-',), linewidths=(1,))
                # Generate the response surface
                plt.tricontourf(xsample_plot, ysample_plot, data_plot,
                                vticks, antialiased=True, vmin=range_cbar[0],
                                vmax=range_cbar[1], cmap=cm.viridis)

                # If doe option activated, generate the points corresponding to
                # the doe and display them on the graph.
                # Requires mask generation like for the response surface.
                # Datas are already correctly sorted if option feat_order is activated.
                if doe is not None:
                    doe = np.asarray(doe)
                    len_sampling = len(doe) - resampling
                    if dim == 4:
                        msk_doe = [((min_zz + (movie - 0.5) * zz_step) < doe[i, 3]
                                    < (min_zz + (movie + 0.5) * zz_step)) and
                                   ((min_z + (plot - 0.5) * z_step) < doe[i, 2]
                                    < (min_z + (plot + 0.5) * z_step))
                                   for i, _ in enumerate(doe)]
                    elif dim == 3:
                        msk_doe = [((min_z + (plot - 0.5) * z_step) < doe[i, 2]
                                    < (min_z + (plot + 0.5) * z_step))
                                   for i, _ in enumerate(doe)]
                    else:
                        msk_doe = msk_total
                    doe_0 = [doe[i, 0] for i in range(len_sampling) if msk_doe[i]]
                    doe_1 = [doe[i, 1] for i in range(len_sampling) if msk_doe[i]]
                    plt.plot(doe_0, doe_1, 'ko')
                    doe_0 = [doe[i, 0] for i in range(len_sampling, len(doe))
                             if msk_doe[i]]
                    doe_1 = [doe[i, 1] for i in range(len_sampling, len(doe))
                             if msk_doe[i]]
                    plt.plot(doe_0, doe_1, 'r^')

                # Colorbar and axis display options
                plt.xlabel(plabels[0])
                plt.ylabel(plabels[1])
                plt.tick_params(axis='x')
                plt.tick_params(axis='y')
                cbar = plt.colorbar(ticks=vticks)
                cbar.set_label(flabel)
                cbar.ax.tick_params()
                if contours is not None:
                    plt.clabel(surface_contour, colors='w')

                # Save the response surface in a pdf file in 2D
                if dim == 2:
                    if fname is not None:
                        plt.savefig(fname + str(plot) + '.pdf',
                                    transparent=True, bbox_inches='tight')
                    else:
                        plt.savefig('Response_Surface.pdf',
                                    transparent=True, bbox_inches='tight')
                    plt.close('all')

                else:
                    # Grab the current response surface for movie generation
                    writer.grab_frame()

            if dim > 2:
                writer.finish()

        # Return last response surface created
        return fig
