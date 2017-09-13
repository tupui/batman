"""
Kiviat in 3D
------------
"""
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.io import wavfile
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import pyplot as plt
plt.switch_backend('Agg')
plt.rc('text', usetex=True)


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

    """3D version of the Kiviat plot."""

    def __init__(self, params, bounds, feval, labels=None):
        """Prepare params for Kiviat plot."""
        self.params = np.asarray(params)
        self.bounds = bounds

        if self.params.shape[1] < 3:
            self.params = np.hstack((params, np.ones((self.params.shape[0], 1))))
            self.bounds[0].extend([0])
            self.bounds[1].extend([2])

        self.data = np.asarray(feval)
        # Adapt how data are reduced to 1D
        self.feval = np.median(self.data, axis=1).reshape(-1, 1)

        order = np.argsort(self.feval, axis=0).flatten()
        self.params = self.params[order]
        self.feval = self.feval[order]

        self.cmap = cm.get_cmap('viridis')
        self.ticks = [0.2, 0.5, 0.8]
        scaler = MinMaxScaler()
        self.scale = scaler.fit(self.bounds)
        self.scale_f = Normalize(vmin=np.percentile(self.feval, 3),
                                 vmax=np.percentile(self.feval, 97), clip=True)

        self.n_params = self.params.shape[1]
        alpha = 2 * np.pi / self.n_params
        self.alphas = [alpha * (i + 1) for i in range(self.n_params)]
        if labels is None:
            self.labels = ['x' + str(i) for i in range(self.n_params)]
        else:
            self.labels = labels
        self.z_offset = - 10
        self.ticks = np.tile(self.ticks, self.n_params).reshape(-1, len(self.ticks)).T
        self.ticks_values = self.scale.inverse_transform(self.ticks).T
        self.x_ticks = np.cos(self.alphas)
        self.y_ticks = np.sin(self.alphas)

    def plane(self, params, feval, idx, ax):
        """Create a Kiviat in 2D.

        Create the mesh in polar coordinates and compute corresponding Z.

        :param params:
        :param feval:
        :param idx: *Z* coordinate of the plane
        """
        params = self.scale.transform(np.asarray(params).reshape(1, -1))[0]

        X = params * np.cos(self.alphas)
        Y = params * np.sin(self.alphas)
        Z = [idx] * (self.n_params + 1)  # +1 to close the geometry

        # Add first point to close the geometry
        X = np.concatenate((X, [X[0]]))
        Y = np.concatenate((Y, [Y[0]]))

        # Plot the surface
        color = self.cmap(self.scale_f(feval))
        ax.plot_trisurf(X, Y, Z, alpha=0.1, antialiased=True, color=color)
        ax.plot(X, Y, Z, alpha=0.5, lw=3, c=color)

    def axis(self, ax):
        """n-dimentions axis definition."""
        for i in range(self.n_params):
            # Create axis
            a = Arrow3D([0, self.x_ticks[i]], [0, self.y_ticks[i]],
                        [self.z_offset, self.z_offset],
                        mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
            ax.add_artist(a)
            # Annotate with labels
            ax.text(1.1 * self.x_ticks[i], 1.1 * self.y_ticks[i],
                    self.z_offset, self.labels[i],
                    fontsize=14, ha='center', va='center', color='k')

            # Add ticks with values
            for j, tick in enumerate(self.ticks[:, 0]):
                x = tick * self.x_ticks[i]
                y = tick * self.y_ticks[i]
                ax.scatter(x, y, self.z_offset, c='k', marker='|')
                ax.text(x, y, self.z_offset * 1.5, self.ticks_values[i][j],
                        fontsize=8, ha='right', va='center', color='k')

    def plot(self, fname=None):
        """Plot 3D kiviat.

        :param str fname: wether to export to filename or display the figures
        :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        m = cm.ScalarMappable(cmap=self.cmap, norm=self.scale_f)
        m.set_array(self.feval)
        cbar = plt.colorbar(m, shrink=0.5, extend='both')
        cbar.set_label('F')

        for i, (point, f_eval) in enumerate(zip(self.params, self.feval)):
            self.plane(point, f_eval[0], i, ax)

        self.axis(ax)

        plt.tight_layout()

        if fname is not None:
            plt.savefig(fname, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close('all')

        return fig, ax

    def f_hops(self, frame_rate=400, fname='kiviat-HOPs.mp4', labels=None):
        """Plot HOPs 3D kiviat."""
        # Base plot
        self.cmap = cm.get_cmap('gray')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()

        for i, (point, f_eval) in enumerate(zip(self.params, self.feval)):
            self.plane(point, f_eval[0], i, ax)

        self.axis(ax)

        # Movie part
        self.cmap = cm.get_cmap('viridis')
        m = cm.ScalarMappable(cmap=self.cmap, norm=self.scale_f)
        m.set_array(self.feval)
        cbar = plt.colorbar(m, shrink=0.5, extend='both')
        cbar.set_label('F')

        movie_writer = manimation.writers['ffmpeg']
        metadata = {'title': 'kiviat-HOPs',
                    'artist': 'batman',
                    'comment': "Kiviat Hypothetical Outcome Plots at {} ms"
                              .format(frame_rate)}

        writer = movie_writer(fps=1000 / frame_rate, metadata=metadata)

        azim_step = 360 / self.feval.shape[0]
        elev_step = 40 / self.feval.shape[0]

        plt.tight_layout()

        with writer.saving(fig, fname, dpi=500):
            for i, (point, f_eval) in enumerate(zip(self.params, self.feval)):
                self.plane(point, f_eval[0], i, ax)
                # Rotate the view
                ax.view_init(elev=-20 + elev_step * i, azim=i * azim_step)
                label = 'HOP: ' + str(labels[i]) \
                    if labels is not None else 'HOP'
                plt.legend([label], loc='best')

                writer.grab_frame()
