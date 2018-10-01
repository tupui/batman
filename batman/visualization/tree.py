"""
Tree for 2D
-----------
"""
import copy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import cm
import matplotlib.tri as tri
from .hdr import HdrBoxplot
from .kiviat import Arrow3D, Kiviat3D

np.set_printoptions(precision=3)


class Tree(Kiviat3D):
    """Tree.

    Extend principle of :class:`batman.visualization.Kiviat3D` but for 2D
    parameter space. Sample are represented by segments and an azimutal
    component encode the value from :class:`batman.visualization.HdrBoxplot`.

    Subclass :class:`batman.visualization.Kiviat3D` by overwriting
    :func:`batman.visualization.Kiviat3D._axis` and
    :func:`batman.visualization.Kiviat3D.plane`.
    """

    def __init__(self, sample, data, bounds=None, plabels=None, range_cbar=None):
        """Prepare params for Tree plot.

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
        super(Tree, self).__init__(sample, data, bounds=bounds,
                                   plabels=plabels, range_cbar=range_cbar)

        scaler = MinMaxScaler()
        self.sample = self.sample[:, :-1]

        if bounds is None:
            bounds = copy.deepcopy(self.sample)
        else:
            bounds = np.asarray(bounds)

        self.scale = scaler.fit(bounds)

        self.n_params = 2
        self.alphas = [0, np.pi]
        self.plabels = ['x' + str(i) for i in range(self.n_params)]\
            if plabels is None else plabels
        self.ticks = [0.2, 0.5, 0.8]
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

    def _plane(self, ax, params, feval, idx, *args):
        """Create a Leaf in 2D.

        From a set of parameters and the corresponding function evaluation,
        a 2D segment is created: a leaf. This leaf has an azimutal component
        conresponding to the distance of the given sample from the mediane
        given by :class:`batman.visualization.HdrBoxplot`.

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
        :return: List of artists added starting with
          [[axis, plabel, [tick, tick_label] * n_ticks] * n_features] and
          complemented with [vert_axis, [lim_HDR, label] * 2, [wedge] * 2].
        :rtype: list.
        """
        out = super(Tree, self)._axis(ax)

        # Vertical line on z to separate coordinates
        out.append(ax.add_artist(
            Arrow3D([0, 0], [0, 0], [self.z_offset, len(self.data)],
                    mutation_scale=20, lw=2, arrowstyle="-", color="k")))

        # Wedges for HDR
        x_ = np.cos(self.dist_max)[0]
        y_ = np.sin(self.dist_max)[0]

        # Separating plane
        # xx = np.array([-x_, -x_, x_, x_])
        # yy = np.array([-y_, -y_, y_, y_])
        # z = np.array([self.z_offset, len(self.data),
        #               self.z_offset, len(self.data)]).reshape(-1, 1)
        # ax.plot_surface(xx, yy, z, alpha=0.2)

        # Limit HDR
        out.append(ax.add_artist(
            Arrow3D([x_ * 0.85, x_], [y_ * 0.85, y_],
                    [self.z_offset, self.z_offset],
                    mutation_scale=20, lw=2, arrowstyle="-", color="r")))
        out.append(ax.text(1.2 * x_, 1.2 * y_, self.z_offset, '90% HDR',
                           fontsize=10, ha='center', va='center', color='k'))

        out.append(ax.add_artist(
            Arrow3D([-x_ * 0.85, -x_], [-y_ * 0.85, -y_],
                    [self.z_offset, self.z_offset],
                    mutation_scale=20, lw=2, arrowstyle="-", color="r")))
        out.append(ax.text(-1.2 * x_, -1.2 * y_, self.z_offset, '90% HDR',
                           fontsize=10, ha='center', va='center', color='k'))

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
            scaler = MinMaxScaler(feature_range=[self.z_offset,
                                                 self.z_offset + 1e-5])
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
        out.append(ax.plot_trisurf(triang, z, cmap=cm.get_cmap('inferno'),
                                   edgecolor='none'))
        triang, z = wedge(np.pi, np.pi + np.pi / 4)
        out.append(ax.plot_trisurf(triang, z, cmap=cm.get_cmap('inferno'),
                                   edgecolor='none'))

        return out
