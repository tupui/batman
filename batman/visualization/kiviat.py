"""
Kiviat in 3D
------------
"""
import copy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as manimation
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import batman as bat
from .hdr import HdrBoxplot


class Arrow3D(FancyArrowPatch):
    """Render 3D arrows."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Create a FancyArrow from two points' coordinates."""
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Overright drawing methods."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Kiviat3D(object):
    """3D version of the Kiviat plot.

    Each realization is stacked on top of each other. The axis represent the
    parameters used to perform the realization.
    """

    def __init__(self, sample, data, idx=None, bounds=None, plabels=None,
                 range_cbar=None, stack_order='qoi', cbar_order='qoi'):
        """Prepare params for Kiviat plot.

        :param array_like sample: Sample of parameters of Shape
          (n_samples, n_params).
        :param array_like data: Sample of realization which corresponds to the
          sample of parameters :attr:`sample` (n_samples, n_features).
        :param int idx: Index on the functional data to consider.
        :param array_like bounds: Boundaries to scale the colors
           shape ([min, n_features], [max, n_features]).
        :param list(str) plabels: Names of each parameters (n_features).
        :param array_like range_cbar: Minimum and maximum values for output
          function (2 values).
        :param str/int stack_order: Set stacking order ['qoi', 'hdr']. If an
          integer, it represents the input variable to take into account.
        :param str cbar_order: Set color mapping order ['qoi', 'hdr'].
        """
        self.sample = np.asarray(sample)
        self.data = np.asarray(data)

        # Adapt how data are reduced to scalar values
        if idx is None:
            self.data = np.median(self.data, axis=1).reshape(-1, 1)
        else:
            self.data = self.data[:, idx].reshape(-1, 1)

        # Stacking and coloring orders
        if 'hdr' in [stack_order, cbar_order]:
            hdr = HdrBoxplot(data)
            pdf = np.exp(hdr.ks_gaussian.score_samples(hdr.data_r)).flatten()

        if stack_order == 'qoi':
            order = np.argsort(self.data, axis=0).flatten()
        elif stack_order == 'hdr':
            order = np.argsort(pdf, axis=0).flatten()
        elif isinstance(stack_order, int):
            order = np.argsort(self.sample[:, stack_order], axis=0).flatten()
        else:
            raise ValueError('Stacking order {} is not valid. Options are qoi or hdr'
                             .format(stack_order))

        if cbar_order == 'qoi':
            pass
        elif cbar_order == 'hdr':
            self.data = pdf[:, None]
        else:
            raise ValueError('Colorbar order {} is not valid. Options are qoi or hdr'
                             .format(cbar_order))

        self.sample = self.sample[order]
        self.data = self.data[order]

        # Color scaling with function evaluations min and max
        self.cmap = cm.get_cmap('viridis')
        scaler = MinMaxScaler()

        # Sample scaling and take into account n_features < 3
        left_for_triangle = 3 - self.sample.shape[1]
        if left_for_triangle > 0:
            self.sample = np.hstack((self.sample, np.ones((self.sample.shape[0],
                                                           left_for_triangle))))
            if bounds is not None:
                bounds = copy.deepcopy(bounds)
                bounds[0].extend([0] * left_for_triangle)
                bounds[1].extend([2] * left_for_triangle)
        if bounds is None:
            bounds = copy.deepcopy(self.sample)

            if left_for_triangle > 0:
                bounds[0, -left_for_triangle:] = 0
                bounds[1, -left_for_triangle:] = 2

        self.scale = scaler.fit(bounds)

        # Colorbar
        self.range_cbar = range_cbar
        if self.range_cbar is None:
            self.scale_f = Normalize(vmin=np.percentile(self.data, 3),
                                     vmax=np.percentile(self.data, 97), clip=True)
        else:
            self.scale_f = Normalize(vmin=self.range_cbar[0],
                                     vmax=self.range_cbar[1], clip=True)

        self.n_params = self.sample.shape[1]
        alpha = 2 * np.pi / self.n_params
        self.alphas = [alpha * (i + 1) for i in range(self.n_params)]
        self.plabels = ['x' + str(i) for i in range(self.n_params)]\
            if plabels is None else plabels
        self.z_offset = - 1
        self.ticks = [0.2, 0.5, 0.8]
        self.ticks = np.tile(self.ticks, self.n_params).reshape(-1, len(self.ticks)).T
        self.ticks_values = self.scale.inverse_transform(self.ticks).T
        self.x_ticks = np.cos(self.alphas)
        self.y_ticks = np.sin(self.alphas)

        # Mesh containing: (X, Y, Z, feval, parameter_index)
        self.mesh_ratio = 3

    def _plane(self, ax, params, feval, idx, fill=True):
        """Create a Kiviat in 2D.

        From a set of parameters and the corresponding function evaluation,
        a 2D Kiviat plane is created. Create the mesh in polar coordinates and
        compute corresponding Z.

        :param ax: Matplotlib AxesSubplot instances to draw to.
        :param array_like params: Parameters of the plane (n_params).
        :param feval: Function evaluation corresponding to :attr:`params`.
        :param idx: *Z* coordinate of the plane.
        :param bool fill: Whether to fill the surface.
        :return: List of artists added.
        :rtype: list.
        """
        params = self.scale.transform(np.asarray(params).reshape(1, -1))[0]

        X = params * np.cos(self.alphas)
        Y = params * np.sin(self.alphas)
        Z = [idx] * (self.n_params + 1)  # +1 to close the geometry

        # Random numbers to prevent null surfaces area
        X[np.where(X == 0.0)] = np.random.rand(1) * 0.001
        Y[np.where(Y == 0.0)] = np.random.rand(1) * 0.001

        # Construct mesh
        self.mesh.append(np.stack((X, Y, Z[:-1], [feval] * self.n_params,
                                   [i for i in range(self.n_params)]),
                                  axis=1))

        # Add first point to close the geometry
        X = np.concatenate((X, [X[0]]))
        Y = np.concatenate((Y, [Y[0]]))

        # Plot the surface
        color = self.cmap(self.scale_f(feval))
        if fill:
            polygon = Poly3DCollection([np.stack([X, Y, Z], axis=1)],
                                       alpha=0.3)
            polygon.set_color(color)
            polygon.set_edgecolor(color)
            out = ax.add_collection3d(polygon)
        else:
            out = ax.plot(X, Y, Z, alpha=0.5, lw=3, c=color)

        return out

    def _axis(self, ax):
        """n-dimentions axis definition.

        Create axis arrows along with annotations with parameters name and
        ticks.

        :param ax: Matplotlib AxesSubplot instances to draw to.
        :return: List of artists added
          [[axis, plabel, [tick, tick_label] * n_ticks] * n_features].
        :rtype: list.
        """
        out = []
        for i in range(self.n_params):
            # Create axis
            out.append(ax.add_artist(
                Arrow3D([0, self.x_ticks[i]], [0, self.y_ticks[i]],
                        [self.z_offset, self.z_offset],
                        mutation_scale=20, lw=1, arrowstyle="-|>", color="k")))
            # Annotate with plabels
            out.append(ax.text(1.1 * self.x_ticks[i], 1.1 * self.y_ticks[i],
                               self.z_offset, self.plabels[i], fontsize=14,
                               ha='center', va='center', color='k'))

            # Add ticks with values
            for j, tick in enumerate(self.ticks[:, 0]):
                x = tick * self.x_ticks[i]
                y = tick * self.y_ticks[i]
                out.append(ax.scatter(x, y, self.z_offset, c='k', marker='|'))
                out.append(ax.text(x, y, self.z_offset,
                                   '{:0.2f}'.format(self.ticks_values[i][j]),
                                   fontsize=8, ha='right', va='center', color='k'))

        return out

    def plot(self, fname=None, flabel='F', ticks_nbr=10, fill=True):
        """Plot 3D kiviat.

        Along with the matplotlib visualization, a VTK mesh is created.

        :param str fname: Whether to export to filename or display the figures.
        :param str flabel: Name of the output function to be plotted next to
          the colorbar.
        :param int ticks_nbr: Number of ticks in the colorbar.
        :param bool fill: Whether to fill the surface.
        :returns: figure.
        :rtype: Matplotlib figure instance, Matplotlib AxesSubplot instances.
        """
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_axis_off()

        m = cm.ScalarMappable(cmap=self.cmap, norm=self.scale_f)
        m.set_array(self.data)
        vticks = np.linspace(self.range_cbar[0], self.range_cbar[1], num=ticks_nbr)\
            if self.range_cbar is not None else None
        cbar = plt.colorbar(m, shrink=0.5, extend='both', ticks=vticks)
        cbar.set_label(flabel)

        self.mesh = []
        for i, (point, f_eval) in enumerate(zip(self.sample, self.data)):
            self._plane(ax, point, f_eval[0], i, fill)

        self._axis(ax)

        ax.set_zlim(self.z_offset, len(self.data))

        bat.visualization.save_show(fname, [fig])

        # Rescale Z axis and export the mesh
        self.mesh = np.array(self.mesh).reshape(-1, 5)
        self.mesh[:, 2] = self.mesh[:, 2] / self.data.shape[0] * self.mesh_ratio

        fname = fname.rsplit('.', 1)[0] + '.vtk'\
            if fname is not None else 'mesh_kiviat.vtk'
        connectivity = self.mesh_connectivity(self.mesh.shape[0], self.n_params)
        self.mesh_vtk_ascii(self.mesh[:, :3],
                            self.mesh[:, 3:], connectivity,
                            fname=fname)

        return fig, ax

    def f_hops(self, frame_rate=400, fname='kiviat-HOPs.mp4', flabel='F',
               ticks_nbr=10, fill=True):
        """Plot HOPs 3D kiviat.

        Each frame consists in a 3D Kiviat with an additional outcome
        highlighted.

        :param int frame_rate: Time between two outcomes (in milliseconds).
        :param str fname: Export movie to filename.
        :param str flabel: Name of the output function to be plotted next to
          the colorbar.
        :param bool fill: Whether to fill the surface.
        :param int ticks_nbr: Number of ticks in the colorbar.
        """
        # Base plot
        self.cmap = cm.get_cmap('gray')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_axis_off()

        self.mesh = []
        for i, (point, f_eval) in enumerate(zip(self.sample, self.data)):
            self._plane(ax, point, f_eval[0], i, fill)

        self._axis(ax)

        ax.set_zlim(self.z_offset, len(self.data))

        # Movie part
        self.cmap = cm.get_cmap('viridis')
        m = cm.ScalarMappable(cmap=self.cmap, norm=self.scale_f)
        m.set_array(self.data)
        vticks = np.linspace(self.range_cbar[0], self.range_cbar[1], num=ticks_nbr)\
            if self.range_cbar is not None else None
        cbar = plt.colorbar(m, shrink=0.5, extend='both', ticks=vticks)
        cbar.set_label(flabel)

        movie_writer = manimation.writers['ffmpeg']
        metadata = {'title': 'kiviat-HOPs',
                    'artist': 'batman',
                    'comment': "Kiviat Hypothetical Outcome Plots at {} ms"
                               .format(frame_rate)}

        writer = movie_writer(fps=1000 / frame_rate, metadata=metadata)

        azim_step = 360 / self.data.shape[0]
        elev_step = 40 / self.data.shape[0]

        with writer.saving(fig, fname, dpi=500):
            for i, (point, f_eval) in enumerate(zip(self.sample, self.data)):
                self._plane(ax, point, f_eval[0], i, fill)
                # Rotate the view
                ax.view_init(elev=-20 + elev_step * i, azim=i * azim_step)

                label = "Parameters: {}\nValue: {}".format(point, f_eval[0])
                scatter_proxy = Line2D([0], [0], linestyle="none")
                ax.legend([scatter_proxy], [label], markerscale=0,
                          loc='upper left', handlelength=0, handletextpad=0)

                writer.grab_frame()

    @staticmethod
    def mesh_connectivity(n_points, n_params):
        """Compute connectivity for Kiviat.

        Using the :attr:`n_points` and :attr:`n_params`, it creates the
        connectivity required by VTK's pixel elements::

                   4
            3 *----*----* 5
              |    |    |
            0 *----*----* 2
                   1

        This will output::

            4 0 1 3 4
            4 1 2 4 5

        :param int n_points: Number of points.
        :param int n_params: Number of features.
        :return: Connectivity.
        :rtype: array_like of shape (n_cells, 5)
        """
        if n_points % n_params != 0:
            raise ValueError("Incorrect points number to create cells, "
                             "n_points % n_params must be 0")
        n_cell = n_points // n_params // 2

        # Building list of node for a given parameter: X1 -> 0 3 6 9
        base = np.array([i * n_params for i in range(n_cell)])
        base = np.array([base + i for i in range(n_params * 2)]).T

        # Building cells for the first stack
        first_cycle = [i for i in range(n_params)]
        second_cycle = [i for i in range(n_params, n_params * 2)]

        # 01 12 23 30
        first_stack = [[first_cycle[i], first_cycle[i] + 1] for i in range(n_params)]
        second_stack = [[second_cycle[i], second_cycle[i] + 1] for i in range(n_params)]
        first_stack[-1][1] = 0
        second_stack[-1][1] = n_params

        first_cells = np.array(list(zip(first_stack, second_stack))).reshape(-1, 4)

        # Combining base using first_cells indices
        out = [base[:, first_cells[i]] for i in range(n_params)]
        connectivity = np.array(list(zip(*out))).reshape(-1, 4)

        connectivity = np.concatenate((4 * np.ones((connectivity.shape[0], 1)),
                                       connectivity), axis=1)
        connectivity = connectivity.astype(int, copy=False)

        return connectivity

    @staticmethod
    def mesh_vtk_ascii(coords, data, connectivity, fname='mesh_kiviat.vtk'):
        """Write mesh file in VTK ascii format.

        Format is as following (example with 3 cells)::

            # vtk DataFile Version 2.0
            Kiviat 3D
            ASCII

            DATASET UNSTRUCTURED_GRID

            POINTS 6 float
            -0.40  0.73 0.00
            -0.00 -0.03 0.00
             0.50  0.00 0.00
            -0.40  0.85 0.04
            -0.00 -0.12 0.04
             0.50  0.00 0.04


            CELLS 3 15
            4 0 1 3 4
            4 1 2 4 5
            4 2 0 5 3

            CELL_TYPES 3
            8
            8
            8

            POINT_DATA 6
            SCALARS value double
            LOOKUP_TABLE default
            17.770e+0
            17.770e+0
            17.770e+0
            17.774e+0
            17.774e+0
            17.774e+0

        :param array_like coordinates: Sample coordinates of shape
          (n_samples, n_features).
        :param array_like data: function evaluations of shape
          (n_samples, n_features).
        """
        n_points = coords.shape[0]

        with open(fname, 'wb') as f:
            header = ("# vtk DataFile Version 2.0\n"
                      "Kiviat 3D\n"
                      "ASCII\n\n"
                      "DATASET UNSTRUCTURED_GRID\n\n")
            f.write(header.encode('utf8'))

            # Sample coordinates
            np.savetxt(f, coords, header='POINTS {} float'.format(n_points),
                       delimiter=' ', footer=' ', comments='')
            # Connectivity
            np.savetxt(f, connectivity, fmt='%d',
                       header='CELLS {} {}'.format(connectivity.shape[0],
                                                   np.prod(connectivity.shape)),
                       footer=' ', comments='')

            # Cell types
            cell_types = 'CELL_TYPES {}\n'.format(connectivity.shape[0])
            f.write(cell_types.encode('utf8'))
            f.writelines(['8\n'.encode('utf8')
                          for _ in range(connectivity.shape[0])])

            # Point data
            data_header = ('\nPOINT_DATA {}\n'.format(n_points))
            f.write(data_header.encode('utf8'))
            for i, datum in enumerate(np.split(data, data.shape[1], axis=1)):
                np.savetxt(f, datum,
                           header=("SCALARS value{} double\n"
                                   "LOOKUP_TABLE default".format(i)),
                           footer=' ', comments='')
