"""
2D Mesh
----------------------

Define fiunctions related to the creation of the 2D graph for statistic representations
over a provided 2D mesh.

* :func:`mesh_2D`.
"""
import numpy as np
from matplotlib import cm
from matplotlib.tri import Triangulation, TriAnalyzer
import matplotlib.pyplot as plt
import batman as bat
from ..input_output import formater


def mesh_2D(fname, var=None, flabels=None, fformat='csv', xlabel='X axis',
            ylabel='Y axis', vmins=None, output_path=None):
    """Visualization of specific variable on a user provided 2D mesh.

    The provided mesh should contain two columns (x,y coordinates for each
    mesh point) and be one of :func:`batman.input_output.available_formats`.
    (x, y) must be respectively the first and second column. Any other column
    is treated as an extra variable and will be used to plot a figure.
    If :attr:`var` is not `None`, its content will be used as plotting
    variables.

    :param str fname: name of mesh file.
    :param array_like var: data to be plotted shape (n_coords, n_vars).
    :param list(str) flabels: names of the variables.
    :param str fformat: format of the mesh file.
    :param str xlabel: name of the x-axis.
    :param str ylabel: name of the y-axis.
    :param lst(double) vmins: value of the minimal output for data filtering.
    :param str output_path: name of the output path.
    :returns: figure.
    :rtype: Matplotlib figure instances.
    """
    # Read the mesh file
    io = formater(fformat)
    mesh = io.read(fname)

    if var is not None:
        var = np.asarray(var)
    else:
        var = mesh[:, 2:]

    if flabels is None:
        flabels = ['y' + str(i) for i in range(var.shape[1])]

    # Input variables
    var_len = var.shape[0]

    if var_len != len(mesh):
        raise ValueError('Variable size not equal: Variable {} - Mesh {}'
                         .format(var_len, len(mesh)))

    if vmins is None:
        vmins = [None] * var_len

    # Meshing with Delaunay triangulation
    tri = Triangulation(mesh[:, 0], mesh[:, 1])

    # Masking badly shaped triangles at the border of the triangular mesh
    mask = TriAnalyzer(tri).get_flat_tri_mask(0.01)
    tri.set_mask(mask)

    # Loop over input parameters
    figs, axs = [], []
    for i, _ in enumerate(var[0]):
        fig, ax = plt.subplots()
        figs.append(fig)
        axs.append(ax)

        cmap = cm.viridis
        cmap.set_bad(alpha=0.0)
        cmap.set_under('w', alpha=0.0)
        plt.tricontourf(tri, var[:, i],
                        antialiased=True, cmap=cmap, vmin=vmins[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tick_params(axis='x')
        plt.tick_params(axis='y')
        cbar = plt.colorbar()
        cbar.set_label(flabels[i])
        cbar.ax.tick_params()

    bat.visualization.save_show(output_path, figs, extend='neither')

    return figs, axs
