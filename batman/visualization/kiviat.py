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


class Kiviat3D:
    """3D version of the Kiviat plot.

    Each realization is stacked on top of each other. The axis represent the
    parameters used to perform the realization.
    """

    def __init__(self, sample, data, bounds=None, plabels=None,
                 range_cbar=None):
        """Prepare params for Kiviat plot.

        :param array_like sample: Sample of parameters of Shape
          (n_samples, n_params).
        :param array_like data: Sample of realization which corresponds to the
          sample of parameters :attr:`sample` (n_samples, n_features).
        :param array_like bounds: Boundaries to scale the colors
           shape ([min, n_features], [max, n_features]).
        :param list(str) plabels: Names of each parameters (n_features).
        :param array_like range_cbar: Minimum and maximum values for output
          function (2 values).
        """
        self.sample = np.asarray(sample)
        self.data = np.asarray(data)
        # Adapt how data are reduced to 1D
        self.data = np.median(self.data, axis=1).reshape(-1, 1)

        order = np.argsort(self.data, axis=0).flatten()
        self.sample = self.sample[order]
        self.data = self.data[order]

        # Color scaling with function evaluations min and max
        self.cmap = cm.get_cmap('viridis')
        self.ticks = [0.2, 0.5, 0.8]
        scaler = MinMaxScaler()

        # Sample scaling and take into account n_features < 3
        left_for_triangle = 3 - self.sample.shape[1]
        if left_for_triangle > 0:
            self.sample = np.hstack((self.sample, np.ones((self.sample.shape[0],
                                                           left_for_triangle))))
            if bounds is not None:
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
        self.ticks = np.tile(self.ticks, self.n_params).reshape(-1, len(self.ticks)).T
        self.ticks_values = self.scale.inverse_transform(self.ticks).T
        self.x_ticks = np.cos(self.alphas)
        self.y_ticks = np.sin(self.alphas)

    def plane(self, ax, params, feval, idx, fill=True):
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
        :return: List of artists added.
        :rtype: list.
        """
        out = []
        for i in range(self.n_params):
            # Create axis
            a = Arrow3D([0, self.x_ticks[i]], [0, self.y_ticks[i]],
                        [self.z_offset, self.z_offset],
                        mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
            out.append(ax.add_artist(a))
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

        return ax

    def plot(self, fname=None, flabel='F', ticks_nbr=10, fill=True):
        """Plot 3D kiviat.

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

        for i, (point, f_eval) in enumerate(zip(self.sample, self.data)):
            self.plane(ax, point, f_eval[0], i, fill)

        self._axis(ax)

        ax.set_zlim(self.z_offset, len(self.data))

        bat.visualization.save_show(fname, [fig])

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

        for i, (point, f_eval) in enumerate(zip(self.sample, self.data)):
            self.plane(ax, point, f_eval[0], i, fill)

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
                self.plane(ax, point, f_eval[0], i, fill)
                # Rotate the view
                ax.view_init(elev=-20 + elev_step * i, azim=i * azim_step)

                label = "Parameters: {}\nValue: {}".format(point, f_eval[0])
                scatter_proxy = Line2D([0], [0], linestyle="none")
                ax.legend([scatter_proxy], [label], markerscale=0,
                          loc='upper left', handlelength=0, handletextpad=0)

                writer.grab_frame()
