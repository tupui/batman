# coding: utf8
import os
import copy
import pytest
import numpy as np
import numpy.testing as npt
from scipy.io import wavfile
import openturns as ot
from mock import patch
from batman.visualization import (HdrBoxplot, Kiviat3D, Tree, pdf,
                                  sensitivity_indices, corr_cov, reshow,
                                  response_surface, doe, doe_ascii, pairplot,
                                  mesh_2D, cusunoro, moment_independent)
from batman.visualization.density import ecdf
from batman.surrogate import SurrogateModel
from batman.space import Space
from batman.functions import (Ishigami, db_Mascaret, el_nino)
import matplotlib.pyplot as plt

try:
    import matplotlib.animation as manimation
    manimation.writers['ffmpeg']
    have_ffmpeg = True
except (RuntimeError, KeyError):
    have_ffmpeg = False

dataset = el_nino()
labels, data = dataset.space, dataset.data

# dataset_tahiti = tahiti()
# labels_tahiti, data_tahiti = dataset_tahiti.space, dataset_tahiti.data


class TestHdr:

    @pytest.fixture(scope="session")
    def hdr(self):
        np.random.seed(123456)
        ot.RandomGenerator.SetSeed(123456)
        return HdrBoxplot(data)

    def test_hdr_basic(self, hdr, tmp, seed):
        print('Data shape: ', data.shape)

        assert len(hdr.extra_quantiles) == 0

        median_t = [24.27, 25.67, 25.98, 25.05, 23.76, 22.40,
                    21.31, 20.43, 20.20, 20.47, 21.17, 22.37]

        npt.assert_almost_equal(hdr.median, median_t, decimal=2)

        quant = np.vstack([hdr.outliers, hdr.hdr_90, hdr.hdr_50])
        quant_t = np.vstack([[27.20, 28.16, 29.00, 28.94, 28.27, 27.24,
                              25.84, 24.01, 22.37, 22.24, 22.38, 23.26],
                             [23.94, 26.16, 27.07, 26.50, 26.40, 25.92,
                              25.36, 24.70, 24.52, 24.67, 25.76, 27.02],
                             [28.01, 28.83, 29.12, 28.23, 27.18, 25.33,
                              23.41, 22.11, 21.25, 21.56, 21.64, 23.01],
                             [25.63, 26.99, 27.63, 27.11, 26.10, 24.65,
                              23.55, 22.50, 22.13, 22.51, 23.37, 24.54],
                             [23.04, 24.58, 24.71, 23.41, 21.98, 20.74,
                              19.85, 19.09, 18.85, 19.04, 19.58, 20.80],
                             [24.85, 26.15, 26.56, 25.78, 24.58, 23.20,
                              22.11, 21.17, 20.93, 21.25, 22.00, 23.23],
                             [23.67, 25.14, 25.46, 24.28, 22.94, 21.62,
                              20.59, 19.75, 19.51, 19.73, 20.37, 21.54]])

        npt.assert_almost_equal(quant, quant_t, decimal=0)

        figs, axs = hdr.plot(fname=os.path.join(tmp, 'hdr_boxplot.pdf'),
                             labels=labels,
                             x_common=np.linspace(1, 12, 12),
                             xlabel='Month of the year (-)',
                             flabel='Water surface temperature (C)')

        assert len(figs) == 3
        assert len(axs) == 3

        fig = reshow(figs[2])
        plt.plot([0, 10], [25, 25])
        axs[2].plot([0, 6], [4, -3])
        fig.savefig(os.path.join(tmp, 'hdr_boxplot_change_sample.pdf'))

        fig = reshow(figs[1])
        axs[1][0].plot([0, 6], [4, -3])
        fig.savefig(os.path.join(tmp, 'hdr_boxplot_change_scatter.pdf'))

    @pytest.mark.xfail(raises=AssertionError, reason='Global optimization')
    @patch("matplotlib.pyplot.show")
    def test_hdr_alpha(self, mock_show, seed):
        hdr = HdrBoxplot(data, alpha=[0.7])
        extra_quant_t = np.vstack([[25.1, 26.4, 26.9, 26.3, 25.2, 23.9,
                                    22.7, 21.8, 21.5, 21.8, 22.5, 23.7],
                                   [23.4, 25.0, 25.1, 24.0, 22.6, 21.3,
                                    20.3, 19.5, 19.2, 19.5, 20.0, 21.2]])
        npt.assert_almost_equal(hdr.extra_quantiles, extra_quant_t, decimal=1)
        hdr.plot()
        hdr.plot(samples=10)

    @pytest.mark.xfail(raises=AssertionError, reason='Global optimization')
    @patch("matplotlib.pyplot.show")
    def test_hdr_multiple_alpha(self, mock_show, seed):
        hdr = HdrBoxplot(data, alpha=[0.4, 0.92])
        extra_quant_t = [[26., 27., 28., 27., 26., 25., 24., 23., 22., 23., 23., 25.],
                         [23., 25., 25., 23., 22., 21., 20., 19., 19., 19., 20., 21.],
                         [25., 26., 26., 26., 24., 23., 22., 21., 21., 21., 22., 23.],
                         [24., 25., 26., 25., 23., 22., 21., 20., 20., 23., 21., 25.]]
        npt.assert_almost_equal(hdr.extra_quantiles, np.vstack(extra_quant_t), decimal=0)
        hdr.plot()

    def test_hdr_threshold(self, seed):
        hdr = HdrBoxplot(data, alpha=[0.8], threshold=0.93)
        labels_pos = np.all(np.isin(data, hdr.outliers), axis=1)
        outliers = labels[labels_pos]
        npt.assert_equal([[1982], [1983], [1997], [1998]], outliers)

    def test_hdr_outliers_method(self, seed):
        hdr = HdrBoxplot(data, threshold=0.93, outliers_method='forest')
        labels_pos = np.all(np.isin(data, hdr.outliers), axis=1)
        outliers = labels[labels_pos]
        npt.assert_equal([[1982], [1983], [1997], [1998]], outliers)

    def test_hdr_optimize_bw(self, seed):
        hdr = HdrBoxplot(data, optimize=True)
        median_t = [24.27, 25.67, 25.98, 25.05, 23.76, 22.40,
                    21.31, 20.43, 20.20, 20.47, 21.17, 22.37]
        npt.assert_almost_equal(hdr.median, median_t, decimal=2)

    @patch("matplotlib.pyplot.show")
    def test_hdr_variance(self, mock_show):
        hdr = HdrBoxplot(data, variance=0.9)
        median_t = [24.37, 25.74, 26.02, 25.07, 23.76, 22.40,
                    21.31, 20.44, 20.23, 20.52, 21.24, 22.44]

        npt.assert_almost_equal(hdr.median, median_t, decimal=2)
        hdr.plot()

    @patch("matplotlib.pyplot.show")
    def test_hdr_plot_data(self, mock_show, hdr):
        hdr.plot(samples=data, labels=labels.tolist())

    @pytest.mark.skipif(not have_ffmpeg, reason='ffmpeg not available')
    def test_hdr_fhops(self, hdr, tmp):
        hdr.f_hops(x_common=np.linspace(1, 12, 12),
                   labels=labels,
                   xlabel='Month of the year (-)',
                   flabel='Water surface temperature (C)',
                   fname=os.path.join(tmp, 'f-HOPs.mp4'))
        hdr.f_hops(samples=10, fname=os.path.join(tmp, 'f-HOPs.mp4'))
        hdr.f_hops(samples=data, fname=os.path.join(tmp, 'f-HOPs.mp4'))

        hdr = HdrBoxplot(data, outliers_method='forest')
        hdr.f_hops(fname=os.path.join(tmp, 'f-HOPs.mp4'))

    def test_hdr_sound(self, hdr, tmp):
        hdr.sound(fname=os.path.join(tmp, 'song-fHOPs-samples.wav'),
                  samples=5, distance=False)
        _, song = wavfile.read(os.path.join(tmp, 'song-fHOPs-samples.wav'))
        assert song.shape[0] == 5 * 44100 * 400 / 1000.0

        hdr.sound(fname=os.path.join(tmp, 'song-fHOPs-data.wav'),
                  samples=data)

        frame_rate = 1000
        hdr.sound(frame_rate=frame_rate, fname=os.path.join(tmp, 'song-fHOPs.wav'))
        _, song = wavfile.read(os.path.join(tmp, 'song-fHOPs.wav'))
        assert song.shape[0] == data.shape[0] * 44100 * frame_rate / 1000.0

    def test_hdr_sample(self, hdr):
        samples = hdr.sample(10)
        assert samples.shape[0] == 10
        assert samples.shape[1] == 12

        samples = hdr.sample([[0, 0], [-1, 3]])
        samples_t = [[24.39, 25.85, 26.23, 25.38, 24.18, 22.86,
                      21.77, 20.85, 20.57, 20.85, 21.55, 22.73],
                     [25.41, 26.54, 26.94, 26.18, 24.65, 22.79,
                      21.35, 20.09, 19.54, 19.74, 20.15, 21.27]]

        npt.assert_almost_equal(samples, samples_t, decimal=2)

    # @pytest.mark.skipif(not have_ffmpeg, reason='ffmpeg not available')
    # @patch("matplotlib.pyplot.show")
    # def test_hdr_tahiti(self, mock_show, tmp):
    #     hdr = HdrBoxplot(data_tahiti)
    #     print('Data tahiti shape: ', data_tahiti.shape)

    #     labels_pos = np.all(np.isin(data_tahiti, hdr.outliers), axis=1)
    #     outliers = labels_tahiti[labels_pos]
    #     npt.assert_equal([1975, 1983, 1998, 2010], outliers)

    #     hdr.plot(fname=os.path.join(tmp, 'hdr_boxplot.pdf'))
    #     hdr.f_hops(samples=10, fname=os.path.join(tmp, 'f-HOPs.mp4'))
    #     hdr.sound(fname=os.path.join(tmp, 'song-fHOPs.wav'))


class TestKiviat:

    @pytest.fixture(scope="session")
    def kiviat_data(self):
        sample = [[30, 4000], [15, 5000]]
        data = [[12], [15]]
        plabels = ['Ks', 'Q', '-']
        bounds = [[15.0, 2500.0], [60.0, 6000.0]]
        kiviat = Kiviat3D(sample, data, bounds=bounds, plabels=plabels)

        return kiviat

    @patch("matplotlib.pyplot.show")
    def test_kiviat_plot(self, mock_show, tmp):
        sample = [[30, 4000], [15, 5000]]
        data = [[12], [15]]
        functional_data = [[12, 300], [15, 347]]

        kiviat = Kiviat3D([[30], [15]], data)
        kiviat.plot()

        kiviat = Kiviat3D(sample, data)
        kiviat.plot(fill=False, ticks_nbr=12)

        kiviat = Kiviat3D(sample, functional_data)
        kiviat = Kiviat3D(sample, functional_data, stack_order='qoi', cbar_order='hdr')
        kiviat = Kiviat3D(sample, functional_data, stack_order='hdr', cbar_order='qoi')
        kiviat = Kiviat3D(sample, functional_data, stack_order=1, cbar_order='hdr')
        kiviat = Kiviat3D(sample, functional_data, idx=1, cbar_order='hdr',
                          range_cbar=[0, 1])
        kiviat.plot(fname=os.path.join(tmp, 'kiviat.pdf'))

    @pytest.mark.skipif(not have_ffmpeg, reason='ffmpeg not available')
    def test_kiviat_fhops(self, kiviat_data, tmp):
        kiviat_data.f_hops(frame_rate=40, ticks_nbr=30,
                           fname=os.path.join(tmp, 'kiviat_fill.mp4'))
        kiviat_data.f_hops(fname=os.path.join(tmp, 'kiviat.mp4'), fill=False)

    @patch("matplotlib.pyplot.show")
    @pytest.mark.skipif(not have_ffmpeg, reason='ffmpeg not available')
    def test_tree(self, mock_show, tmp):
        sample = [[30, 4000], [15, 5000], [20, 4500]]
        functional_data = [[12, 300], [15, 347], [14, 320]]
        tree = Tree(sample, functional_data,
                    bounds=[[10.0, 2500.0], [60.0, 6000.0]])
        tree.plot(fname=os.path.join(tmp, 'tree.pdf'),
                  flabel='Water level (m)')
        tree.f_hops(fname=os.path.join(tmp, 'tree.mp4'))

    def test_connectivity(self):
        connectivity = Kiviat3D.mesh_connectivity(6, 3)
        connectivity_t = np.array([[4, 0, 1, 3, 4],
                                   [4, 1, 2, 4, 5],
                                   [4, 2, 0, 5, 3]], dtype=int)
        npt.assert_equal(connectivity, connectivity_t)

        with pytest.raises(ValueError):
            Kiviat3D.mesh_connectivity(6, 4)

        connectivity = Kiviat3D.mesh_connectivity(8, 4)
        connectivity_t = np.array([[4, 0, 1, 4, 5],
                                   [4, 1, 2, 5, 6],
                                   [4, 2, 3, 6, 7],
                                   [4, 3, 0, 7, 4]], dtype=int)
        npt.assert_equal(connectivity, connectivity_t)


class TestPdf:

    def test_pdf_1D(self, tmp):
        pdf(data[:10, 5].reshape(-1, 1), fname=os.path.join(tmp, 'pdf.pdf'))

    @patch("matplotlib.pyplot.show")
    def test_pdf_surrogate(self, mock_show, ishigami_data):
        dist = ot.ComposedDistribution(ishigami_data.dists)
        surrogate = SurrogateModel('rbf', ishigami_data.space.corners,
                                   ishigami_data.space.plabels)
        surrogate.fit(ishigami_data.space, ishigami_data.target_space)
        settings = {
            "dist": dist,
            "model": surrogate,
            "method": 'kriging',
            "bounds": ishigami_data.space.corners
        }
        pdf(settings)

    @patch("matplotlib.pyplot.show")
    def test_pdf_nD(self, mock_show, tmp):
        fig_pdf = pdf(data, xdata=np.linspace(1, 12, 12),
                      range_cbar=[0, 0.5], ticks_nbr=6,
                      fname=os.path.join(tmp, 'pdf_nd.pdf'))
        reshow(fig_pdf)
        plt.plot([0, 10], [25, 25])
        plt.show()
        plt.close()

    def test_pdf_nD_moments(self, tmp):
        pdf(data, xlabel='s', flabel='Y', moments=True,
            fname=os.path.join(tmp, 'pdf_nd_moments.pdf'))

    def test_pdf_dotplot(self, tmp):
        pdf(data[:10, 5].reshape(-1, 1), dotplot=True,
            fname=os.path.join(tmp, 'pdf_dotplot.pdf'))


class TestSensitivity:

    @patch("matplotlib.pyplot.show")
    def test_sobols_aggregated(self, mock_show, tmp):
        fun = Ishigami()
        indices = [fun.s_first, fun.s_total]
        fig = sensitivity_indices(indices, conf=0.05)
        fig = reshow(fig[0])
        plt.plot([0, 10], [0.5, 0.5])
        fig.show()
        sensitivity_indices(indices, plabels=['x1', 't', 'y'],
                            fname=os.path.join(tmp, 'sobol.pdf'))
        sensitivity_indices(indices, polar=True, conf=[[0.2, 0.1, 0.1], [0.1, 0.1, 0.1]])
        sensitivity_indices([indices[0]])

    @patch("matplotlib.pyplot.show")
    def test_sobols_map(self, mock_show, tmp):
        fun = db_Mascaret()
        indices = [fun.s_first, fun.s_total, fun.s_first_full, fun.s_total_full]
        sensitivity_indices(indices)
        sensitivity_indices(indices, plabels=['Ks', 'Q'], xdata=fun.x,
                            fname=os.path.join(tmp, 'sobol_map.pdf'))


class TestResponseSurface:

    @patch("matplotlib.pyplot.show")
    def test_response_surface_1D(self, mock_show, tmp):

        def fun(x):
            return x ** 2
        bounds = [[-7], [10]]
        path = os.path.join(tmp, 'rs_1D.pdf')
        response_surface(bounds=bounds, fun=fun, fname=path)

        xdata = np.linspace(0, 1, 10)

        def fun(x):
            return (xdata * x) ** 2
        sample = np.array(range(5)).reshape(-1, 1)
        data = fun(sample)
        response_surface(bounds=bounds, sample=sample, data=data, xdata=xdata)

    @pytest.mark.xfail(raises=ValueError)
    @patch("matplotlib.pyplot.show")
    def test_response_surface_2D_scalar(self, mock_show, tmp, branin_data, seed):
        space = branin_data.space
        bounds = [[-7, 0], [10, 15]]
        path = os.path.join(tmp, 'rs_2D_vector.pdf')
        response_surface(bounds=bounds, sample=space, data=branin_data.target_space)
        response_surface(bounds=bounds, fun=branin_data.func, doe=space, resampling=4,
                         fname=path, feat_order=[2, 1])

    @patch("matplotlib.pyplot.show")
    def test_response_surface_2D_vector(self, mock_show, tmp, mascaret_data):
        space = mascaret_data.space
        data = mascaret_data.target_space
        xdata = mascaret_data.func.x
        bounds = [[15.0, 2500.0], [60, 6000.0]]
        order = [1, 2]
        path = os.path.join(tmp, 'rs_2D_vector.pdf')
        response_surface(bounds=bounds, sample=space, data=data, xdata=xdata, fname=path)
        response_surface(bounds=bounds, fun=mascaret_data.func, xdata=xdata,
                         plabels=['Ks', 'Q'], feat_order=order, flabel='Z')

    @pytest.mark.skipif(not have_ffmpeg, reason='ffmpeg not available')
    def test_response_surface_3D(self, ishigami_data, tmp):
        space = ishigami_data.space
        fun = ishigami_data.func
        bounds = [[-4, -4, -4], [4, 4, 4]]
        order = [1, 2, 3]
        path = os.path.join(tmp, 'rs_3D_vector')
        response_surface(bounds=bounds, fun=fun, doe=space, resampling=30,
                         contours=[-20, 0, 20], fname=path, feat_order=order)

    @pytest.mark.skipif(not have_ffmpeg, reason='ffmpeg not available')
    def test_response_surface_4D(self, g_function_data, tmp):
        space = g_function_data.space
        fun = g_function_data.func
        bounds = g_function_data.space.corners
        order = [2, 3, 4, 1]
        path = os.path.join(tmp, 'rs_4D_vector')
        response_surface(bounds=bounds, fun=fun, doe=space, resampling=10,
                         axis_disc=[2, 15, 15, 15], fname=path, feat_order=order)


class Test2Dmesh:

    def test_2D_mesh(self, tmp):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        fname = os.path.join(datadir, 'data_Garonne.csv')
        path = os.path.join(tmp, 'garonne_2D.pdf')
        vmin = 2.5
        mesh_2D(fname=fname, fformat='csv', xlabel='x label',
                flabels=['Variable'], vmins=[vmin], output_path=path)

    def test_2D_mesh_add_var(self, tmp, mascaret_data):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        fname = os.path.join(datadir, 'data_2D_mesh.csv')
        var_sobol = [[0.1, 0.2], [0.3, 0.2], [0.88, 0.2], [0.9, 1.0], [0.1, 0.12]]
        path = os.path.join(tmp, 'data_2D.pdf')
        flabels = ['Ks', 'Q']
        mesh_2D(fname=fname, fformat='csv', xlabel='x label', var=var_sobol,
                flabels=flabels, output_path=path)

        var_sobol = [[0.1, 0.2], [0.3, 0.2], [0.88, 0.2], [0.9, 1.0]]
        with pytest.raises(ValueError):
            mesh_2D(fname=fname, var=var_sobol)


class TestDoe:

    @patch("matplotlib.pyplot.show")
    def test_doe(self, mock_show, mascaret_data):
        doe(mascaret_data.space)

    def test_doe_3D(self, ishigami_data, tmp):
        fig, ax = doe(ishigami_data.space, fname=os.path.join(tmp, 'DOE.pdf'))

        fig = reshow(fig)
        ax[0].plot([0, 6], [4, -3])
        fig.savefig(os.path.join(tmp, 'DOE_change.pdf'))

    def test_doe_mufi(self, ishigami_data, tmp):
        doe(ishigami_data.space, multifidelity=True,
            fname=os.path.join(tmp, 'DOE_mufi.pdf'))

    def test_doe_ascii(self, ishigami_data, tmp):
        doe_ascii(ishigami_data.space, plabels=['a', 'b', 'c'],
                  bounds=[[-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]])

        doe_ascii([[0.2, 0.8], [0.3, 0.7]], fname=os.path.join(tmp, 'DOE_ascii.pdf'))

    def test_pairplot(self, ishigami_data, tmp):
        pairplot(ishigami_data.space, ishigami_data.target_space,
                 plabels=['x1', 'x2', 'x3'],
                 fname=os.path.join(tmp, 'pairplot.pdf'))


@patch("matplotlib.pyplot.show")
def test_corr_cov(mock_show, mascaret_data, tmp):
    func = mascaret_data.func
    dist = ot.ComposedDistribution(mascaret_data.dists)
    sample = np.array(ot.LHSExperiment(dist, 500).generate())
    data = func(sample)
    corr_cov(data, sample, func.x, interpolation='lanczos', plabels=['Ks', 'Q'])
    corr_cov(data, sample, func.x, fname=os.path.join(tmp, 'corr_cov.pdf'))


class TestDensity:

    def test_cusunoro(self, ishigami_data, tmp):
        Y = ishigami_data.target_space.flatten()
        X = ishigami_data.space
        cuso = cusunoro(X, Y, plabels=['x1', 'x2', 'x3'],
                        fname=os.path.join(tmp, 'cusunoro.pdf'))

        npt.assert_almost_equal(cuso[2], [0.328, 0.353, 0.018], decimal=3)

    def test_ecdf(self):
        data = np.array([1, 3, 6, 10, 2])
        xs, ys = ecdf(data)
        npt.assert_equal(xs, [1, 2, 3, 6, 10])
        npt.assert_equal(ys, [0, 0.25, 0.5, 0.75, 1.])

    def test_moment_independant(self, ishigami_data, tmp):
        ishigami_data_ = copy.deepcopy(ishigami_data)
        ishigami_data_.space.max_points_nb = 5000
        X = ishigami_data_.space.sampling(5000, 'olhs')
        Y = ishigami_data_.func(X).flatten()

        momi = moment_independent(X, Y, plabels=['x1', 'x2', 'x3'],
                                  fname=os.path.join(tmp, 'moment_independent.pdf'))

        npt.assert_almost_equal(momi[2]['Kolmogorov'], [0.236, 0.377, 0.107], decimal=2)
        npt.assert_almost_equal(momi[2]['Kuiper'], [0.257, 0.407, 0.199], decimal=2)
        npt.assert_almost_equal(momi[2]['Delta'], [0.211, 0.347, 0.162], decimal=2)
        npt.assert_almost_equal(momi[2]['Sobol'], [0.31, 0.421, 0.002], decimal=2)

        # Cramer
        space = Space(corners=[[-5, -5], [5, 5]], sample=5000)
        space.sampling(dists=['Normal(0, 1)', 'Normal(0, 1)'])
        Y = [np.exp(x_i[0] + 2 * x_i[1]) for x_i in space]
        X = np.array(space)

        momi = moment_independent(X, Y,
                                  fname=os.path.join(tmp, 'moment_independent.pdf'))

        npt.assert_almost_equal(momi[2]['Cramer'], [0.113, 0.572], decimal=2)
