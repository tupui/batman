"""Tree."""
import copy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as manimation
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Wedge, Circle
from matplotlib.lines import Line2D
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from batman.visualization.hdr import HdrBoxplot

np.set_printoptions(precision=3)


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


class Tree:
    """Tree."""

    def __init__(self, sample, data, bounds=None, plabels=None, range_cbar=None):
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

        if bounds is None:
            bounds = copy.deepcopy(self.sample)

        self.scale = scaler.fit(bounds)

        # Colorbar
        self.range_cbar = range_cbar
        if self.range_cbar is None:
            self.scale_f = Normalize(vmin=np.percentile(self.data, 3),
                                     vmax=np.percentile(self.data, 97), clip=True)
        else:
            self.scale_f = Normalize(vmin=self.range_cbar[0],
                                     vmax=self.range_cbar[1], clip=True)

        self.n_params = 2
        self.alphas = [0, np.pi]
        self.plabels = ['x' + str(i) for i in range(self.n_params)]\
            if plabels is None else plabels
        self.z_offset = - 1
        self.ticks = np.tile(self.ticks, self.n_params).reshape(-1, len(self.ticks)).T
        self.ticks_values = self.scale.inverse_transform(self.ticks).T
        self.x_ticks = [-1, 1]

        self.y_ticks = [0] * 2

        # Distance based on HDR
        hdr = HdrBoxplot(data)
        centroide = hdr.pca.transform(hdr.median.reshape(1, -1))
        dists = cdist(centroide, hdr.pca.transform(data))[0]
        scaler = MinMaxScaler(feature_range=[0, np.pi / 4])
        self.dists = scaler.fit_transform(dists.reshape(-1, 1))

        hdr_90 = hdr.pca.transform(np.array(hdr.hdr_90))
        dists = cdist(centroide, hdr_90)[0]
        self.dist_max = max(scaler.transform(dists.reshape(-1, 1)))

    def leaf(self, ax, params, feval, idx):
        """Create a Kiviat in 2D.

        From a set of parameters and the corresponding function evaluation,
        a 2D Kiviat plane is created. Create the mesh in polar coordinates and
        compute corresponding Z.

        :param ax: Matplotlib AxesSubplot instances to draw to.
        :param array_like params: Parameters of the plane (n_params).
        :param feval: Function evaluation corresponding to :attr:`params`.
        :param idx: *Z* coordinate of the plane.
        :return: List of artists added.
        :rtype: list.
        """
        params = self.scale.transform(np.asarray(params).reshape(1, -1))[0]

        X = params * np.cos(self.dists[idx]) * [-1, 1]
        Y = params * np.sin(self.dists[idx]) * [-1, 1]
        Z = [idx] * (self.n_params)

        # Plot the surface
        color = self.cmap(self.scale_f(feval))
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

        # Vertical line on z to separate coordinates
        a = Arrow3D([0, 0], [0, 0], [self.z_offset, len(self.data)],
                    mutation_scale=20, lw=2, arrowstyle="-", color="k")
        out.append(ax.add_artist(a))

        # Wedges for HDR
        x_ = np.cos(self.dist_max)[0]
        y_ = np.sin(self.dist_max)[0]
        # xx = np.array([-x_, -x_, x_, x_])
        # yy = np.array([-y_, -y_, y_, y_])
        # z = np.array([self.z_offset, len(self.data),
        #               self.z_offset, len(self.data)]).reshape(-1, 1)
        # ax.plot_surface(xx, yy, z, alpha=0.2)

        a = Arrow3D([x_*0.85, x_], [y_*0.85, y_],
                    [self.z_offset, self.z_offset],
                    mutation_scale=20, lw=2, arrowstyle="-", color="r")
        out.append(ax.add_artist(a))
        out.append(ax.text(1.2 * x_, 1.2 * y_,
                           self.z_offset, '90% HDR', fontsize=10,
                           ha='center', va='center', color='k'))
        a = Arrow3D([-x_*0.85, -x_], [-y_*0.85, -y_],
                    [self.z_offset, self.z_offset],
                    mutation_scale=20, lw=2, arrowstyle="-", color="r")
        out.append(ax.add_artist(a))
        out.append(ax.text(-1.2 * x_, -1.2 * y_,
                           self.z_offset, '90% HDR', fontsize=10,
                           ha='center', va='center', color='k'))

        def wedge(alpha_i, alpha_e):
            """Wedge in 3-dimensions."""
            # First create the x and y coordinates of the points.
            n_angles = 50
            n_radii = 2
            min_radius = 0.9
            radii = np.linspace(min_radius, 0.95, n_radii)

            angles = np.linspace(alpha_i, alpha_e, n_angles, endpoint=False)
            angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
            angles[:, 1::2] += np.pi/n_angles

            x = (radii * np.cos(angles)).flatten()
            y = (radii * np.sin(angles)).flatten()

            # z set to small scale to appear as flat
            scaler = MinMaxScaler(feature_range=[0, 1e-5])
            z = scaler.fit_transform(angles.reshape(-1, 1)).flatten()

            # Create the Triangulation; no triangles so Delaunay triangulation
            triang = tri.Triangulation(x, y)

            # Mask off unwanted triangles.
            xmid = x[triang.triangles].mean(axis=1)
            ymid = y[triang.triangles].mean(axis=1)
            mask = np.where(xmid**2 + ymid**2 < min_radius**2, 1, 0)
            triang.set_mask(mask)

            return triang, z

        triang, z = wedge(0, np.pi / 4)
        ax.plot_trisurf(triang, z, cmap=cm.get_cmap('inferno'), edgecolor='none')
        triang, z = wedge(np.pi, np.pi + np.pi / 4)
        ax.plot_trisurf(triang, z, cmap=cm.get_cmap('inferno'), edgecolor='none')

        return ax

    def plot(self, fname=None, flabel='F', ticks_nbr=10):
        """Plot 3D kiviat.

        :param str fname: Whether to export to filename or display the figures.
        :param str flabel: Name of the output function to be plotted next to
          the colorbar.
        :param int ticks_nbr: Number of ticks in the colorbar.
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
            self.leaf(ax, point, f_eval[0], i)

        self._axis(ax)

        ax.set_zlim(self.z_offset, len(self.data))

        bat.visualization.save_show(fname, [fig])

        return fig, ax

    def f_hops(self, frame_rate=400, fname='kiviat-HOPs.mp4', flabel='F', ticks_nbr=10):
        """Plot HOPs 3D kiviat.

        Each frame consists in a 3D Kiviat with an additional outcome
        highlighted.

        :param int frame_rate: Time between two outcomes (in milliseconds).
        :param str fname: Export movie to filename.
        :param str flabel: Name of the output function to be plotted next to
          the colorbar.
        :param int ticks_nbr: Number of ticks in the colorbar.
        """
        # Base plot
        self.cmap = cm.get_cmap('gray')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_axis_off()

        for i, (point, f_eval) in enumerate(zip(self.sample, self.data)):
            self.leaf(ax, point, f_eval[0], i)

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
                self.leaf(ax, point, f_eval[0], i)
                # Rotate the view
                ax.view_init(elev=-20 + elev_step * i, azim=i * azim_step)

                label = "Parameters: {}\nValue: {}".format(point, f_eval[0])
                scatter_proxy = Line2D([0], [0], linestyle="none")
                ax.legend([scatter_proxy], [label], markerscale=0,
                          loc='upper left', handlelength=0, handletextpad=0)

                writer.grab_frame()



from batman.space import Space
from batman.functions import Mascaret, el_nino
import batman as bat

data_ = el_nino()
data_.toarray()
feval = data_.data


settings = {
        "corners": [[15., 3000.], [60., 6000.]],
        "init_size": 100
        }
f = Mascaret()
space = Space(settings["corners"], settings["init_size"])
space.sampling(kind="halton", dists=["Uniform(15, 60)", "Normal(4035, 400)"])
feval = f(space)#[:, 0].reshape(-1, 1)
space = np.array(space)
print('Space shape: {}\nFunction shape: {}'.format(space.shape, feval.shape))

tree = Tree(space, feval, plabels=['Ks', 'Q'])
tree.plot(flabel='Water level (m)')
# tree.f_hops(frame_rate=400)


