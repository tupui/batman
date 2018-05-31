"""
2D Mesh
----------------------

Define fiunctions related to the creation of the 2D graph for statistic representations
over a provided 2D mesh.

* :func:`mesh_2D`.
"""
import numpy as np
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.tri import Triangulation, TriAnalyzer
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


def read_file(fname=None):
    """Reader of .txt file format.

    :param str fname: name of the file to read
    :returns: data in the input file, corresponding line
    """

    file_mesh = open(fname, 'r')
    i = 0
    for line in file_mesh:
        i = i + 1
        coord_line = line.split()
    coords = np.zeros((i, len(coord_line)))
    file_mesh.close()

    file_mesh = open(fname, 'r')
    i = 0
    for line in file_mesh:
        coord_line = line.split()
        for n in range(len(coord_line)):
            coords[i, n] = float(coord_line[n])
        i = i + 1

    file_mesh.close()

    return coords, i


def mesh_2D(fname=None, fformat='txt', xlabel='X axis', ylabel='Y axis', 
            title2D=None, outlabel=None, vmin=0.0, output_path=None):
    """Visualization of some statistics on a user provided 2D mesh.

    The provided mesh should contain more than two columns (x,y coordinates for each mesh point,
    and additional columns for the output parameters) and be in one of the following format:
    - .txt

    :param str fname: name of mesh file
    :param str fformat: format of the mesh file
    :param str xlabel: name of the x-axis
    :param str ylabel: name of the y-axis
    :param str title2D: title of the graph
    :param str outlabel: name of the output variable
    :param real vmin: Value of the minimal output for data filtering
    :param str output_path: name of the output path
    :returns: figure.
    :rtype: Matplotlib figure instances.
    """

    # Read the mesh file
    coords, i = read_file(fname=fname)

    # Minimum circle ratio for triangle quality
    min_circle_ratio = 0.01

    # If more than 2 coordinates are provided, additional columns are considered
    # to be values associated to coordinates (x,y) and to be plotted
    if len(coords[1]) > 2:
        for col in range(len(coords[1]) - 2):
            col = col + 2

            # Masking contours with no datas
            mask = [(coords[j, col] != 0.0) for j in range(i)]
            x_coords = [coords[j, 0] for j, _ in enumerate(coords[:, 0]) if mask[j]]
            y_coords = [coords[j, 1] for j, _ in enumerate(coords[:, 1]) if mask[j]]
            z_coords = [coords[j, col] for j, _ in enumerate(coords[:, col]) if mask[j]]

            # Meshing with Delaunay triangulation
            tri = Triangulation(x_coords, y_coords)

            # Masking badly shaped triangles at the border of the triangular mesh
            mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
            tri.set_mask(mask)

            # Plot figure
            plt.figure()
            if title2D is not None:
                plt.title(title2D)
            else:
                plt.title("Additional variable number : " + str(col-1))
            cmap = cm.viridis
            cmap.set_bad(alpha=0.0)
            cmap.set_under('w', alpha=0.0)
            plt.tricontourf(tri, z_coords, antialiased=True, cmap=cmap, vmin=vmin)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tick_params(axis='x')
            plt.tick_params(axis='y')
            cbar = plt.colorbar()
            if outlabel is not None:
                cbar.set_label(outlabel)
            else:
                cbar.set_label('Column number : ' + str(col-1))
            cbar.ax.tick_params()

            # Plot the graph
            if output_path is not None:
                plt.savefig(output_path + '_' + str(col-1) + '.pdf', transparent=True,
                            bbox_inches='tight', extend='neither')


def mesh_2D_add_var(fname=None, fformat='txt', xlabel='X axis', ylabel='Y axis',
                    vmin=0.0, var=None, var_name="Sobol", plabels=None,
                    output_path=None):
    """Visualization of some statistics (Sobol' indices) on a user provided 2D mesh.

    The provided mesh should contain two columns (x,y coordinates for each mesh point)
    and be in one of the following format:
    - .txt

    :param str fname: name of mesh file
    :param str fformat: format of the mesh file
    :param str xlabel: name of the x-axis
    :param str ylabel: name of the y-axis
    :param real vmin: value of the minimal output for data filtering
    :param real list var: datas to be plotted
    :param str var_name: global plotted variable (Sobol' indices by default)
    :param str list plabels: names of the input parameters for the Sobol' indices
    :param str output_path: name of the output path
    :returns: figure.
    :rtype: Matplotlib figure instances.
    """

    # Read the mesh file
    coords, i = read_file(fname=fname)

    # Minimum circle ratio for triangle quality
    min_circle_ratio = 0.01

    if len(var) == len(coords):
        # Loop over input parameters
        for i in range(len(var[0])):

            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            z_coords = [0.0]*len(var)
            for j in range(len(var)):
                z_coords_tuple = var[j]
                z_coords[j] = z_coords_tuple[i]

            # Meshing with Delaunay triangulation
            tri = Triangulation(x_coords, y_coords)

            # Masking badly shaped triangles at the border of the triangular mesh
            mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
            tri.set_mask(mask)

            # Plot figure
            plt.figure()
            plt.title(var_name)
            cmap = cm.viridis
            cmap.set_bad(alpha=0.0)
            cmap.set_under('w', alpha=0.0)
            plt.tricontourf(tri, z_coords, antialiased=True, cmap=cmap, vmin=vmin)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tick_params(axis='x')
            plt.tick_params(axis='y')
            cbar = plt.colorbar()
            cbar.set_label(var_name)
            cbar.ax.tick_params()

            # Plot the graph
            if output_path is not None:
                plt.savefig(output_path + '_' + var_name + '_' + plabels[i] + '.pdf',
                            transparent=True, bbox_inches='tight', extend='neither')
