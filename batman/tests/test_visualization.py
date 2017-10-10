# coding: utf8
import os
import pytest
import numpy as np
import numpy.testing as npt
from scipy.io import wavfile
import openturns as ot
from mock import patch
from batman.visualization import (HdrBoxplot, Kiviat3D, pdf, sobol, reshow,
                                  response_surface, doe, corr_cov)
from batman.surrogate import SurrogateModel
from batman.functions import Ishigami, Mascaret
import matplotlib.pyplot as plt

try:
    import matplotlib.animation as manimation
    manimation.writers['ffmpeg']
    have_ffmpeg = True
except (RuntimeError, KeyError):
    have_ffmpeg = False

# Water surface temperature data from:
# http://www.cpc.ncep.noaa.gov/data/indices/
path = os.path.dirname(os.path.realpath(__file__))
labels, data = np.loadtxt(os.path.join(path, 'functional_dataset/elnino.dat'),
                          skiprows=1, usecols=(0, 2), unpack=True)
labels = labels.reshape(-1, 12)[:, 0]
data = data.reshape(-1, 12)

# labels_tahiti, *data_tahiti = np.loadtxt(os.path.join(path, 'functional_dataset/tahiti.dat'),
#                                          skiprows=4, usecols=range(0, 13),
#                                          unpack=True)
# data_tahiti = np.array(data_tahiti).T


@pytest.fixture(scope="session")
def hdr():
    return HdrBoxplot(data)


@pytest.mark.xfail(raises=AssertionError, reason='Global optimization')
def test_hdr_basic(hdr, tmp):
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
                         ylabel='Water surface temperature (C)')

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
def test_hdr_alpha(mock_show):
    hdr = HdrBoxplot(data, alpha=[0.7])
    extra_quant_t = np.vstack([[25.1, 26.4, 26.9, 26.3, 25.2, 23.9,
                                22.7, 21.8, 21.5, 21.8, 22.5, 23.7],
                               [23.4, 25.0, 25.1, 24.0, 22.6, 21.3,
                                20.3, 19.5, 19.2, 19.5, 20.0, 21.2]])
    npt.assert_almost_equal(hdr.extra_quantiles, extra_quant_t, decimal=1)
    hdr.plot()


@pytest.mark.xfail(raises=AssertionError, reason='Global optimization')
@patch("matplotlib.pyplot.show")
def test_hdr_multiple_alpha(mock_show):
    hdr = HdrBoxplot(data, alpha=[0.4, 0.92])
    extra_quant_t = [[25.712, 27.052, 27.711, 27.200,
                      26.162, 24.833, 23.639, 22.378,
                      22.250, 22.640, 23.472, 24.649],
                     [22.973, 24.526, 24.608, 23.343,
                      21.908, 20.655, 19.750, 19.046,
                      18.812, 18.989, 19.520, 20.685],
                     [24.667, 26.033, 26.416, 25.584,
                      24.308, 22.849, 21.684, 20.948,
                      20.483, 21.019, 21.751, 22.890],
                     [23.873, 25.371, 25.667, 24.644,
                      23.177, 21.923, 20.791, 20.015,
                      19.697, 19.951, 20.622, 21.858]]
    npt.assert_almost_equal(hdr.extra_quantiles, np.vstack(extra_quant_t), decimal=0)
    hdr.plot()


def test_hdr_threshold():
    hdr = HdrBoxplot(data, alpha=[0.8], threshold=0.93)
    labels_pos = np.all(np.isin(data, hdr.outliers), axis=1)
    outliers = labels[labels_pos]
    npt.assert_equal([1982, 1983, 1997, 1998], outliers)


def test_hdr_outliers_method():
    hdr = HdrBoxplot(data, threshold=0.93, outliers_method='forest')
    labels_pos = np.all(np.isin(data, hdr.outliers), axis=1)
    outliers = labels[labels_pos]
    npt.assert_equal([1982, 1983, 1997, 1998], outliers)


def test_hdr_optimize_bw():
    hdr = HdrBoxplot(data, optimize=True)
    median_t = [24.27, 25.67, 25.98, 25.05, 23.76, 22.40,
                21.31, 20.43, 20.20, 20.47, 21.17, 22.37]
    npt.assert_almost_equal(hdr.median, median_t, decimal=2)


@patch("matplotlib.pyplot.show")
def test_hdr_variance(mock_show):
    hdr = HdrBoxplot(data, variance=0.9)
    median_t = [24.37, 25.74, 26.02, 25.07, 23.76, 22.40,
                21.31, 20.44, 20.23, 20.52, 21.24, 22.44]

    npt.assert_almost_equal(hdr.median, median_t, decimal=2)
    hdr.plot()


@patch("matplotlib.pyplot.show")
def test_hdr_plot_data(mock_show, hdr):
    hdr.plot(samples=data, labels=labels.tolist())


@pytest.mark.skipif(not have_ffmpeg, reason='ffmpeg not available')
def test_hdr_fhops(hdr, tmp):
    hdr.f_hops(x_common=np.linspace(1, 12, 12),
               labels=labels,
               xlabel='Month of the year (-)',
               ylabel='Water surface temperature (C)',
               fname=os.path.join(tmp, 'f-HOPs.mp4'))
    hdr.f_hops(samples=10, fname=os.path.join(tmp, 'f-HOPs.mp4'))
    hdr.f_hops(samples=data, fname=os.path.join(tmp, 'f-HOPs.mp4'))

    hdr = HdrBoxplot(data, outliers_method='forest')
    hdr.f_hops(fname=os.path.join(tmp, 'f-HOPs.mp4'))


def test_hdr_sound(hdr, tmp):
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


def test_hdr_sample(hdr):
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
# def test_hdr_tahiti(mock_show, tmp):
#     hdr = HdrBoxplot(data_tahiti)
#     print('Data tahiti shape: ', data_tahiti.shape)

#     labels_pos = np.all(np.isin(data_tahiti, hdr.outliers), axis=1)
#     outliers = labels_tahiti[labels_pos]
#     npt.assert_equal([1975, 1983, 1998, 2010], outliers)

#     hdr.plot(fname=os.path.join(tmp, 'hdr_boxplot.pdf'))
#     hdr.f_hops(samples=10, fname=os.path.join(tmp, 'f-HOPs.mp4'))
#     hdr.sound(fname=os.path.join(tmp, 'song-fHOPs.wav'))


@pytest.fixture(scope="session")
def kiviat_data():
    space = [[30, 4000], [15, 5000]]
    feval = [[12], [15]]
    param_names = ['Ks', 'Q', '-']

    corners = [[15.0, 2500.0], [60.0, 6000.0]]
    kiviat = Kiviat3D(space, corners, feval)

    corners = [[15.0, 2500.0], [60.0, 6000.0]]
    kiviat = Kiviat3D(space, corners, feval, param_names=param_names)

    labels = ["Ks={}, Q={}".format(ks, q) for (ks, q) in space]

    return kiviat, labels


@pytest.mark.skipif(not have_ffmpeg, reason='ffmpeg not available')
def test_kiviat_fhops(kiviat_data, tmp):
    kiviat, labels = kiviat_data
    kiviat.f_hops(frame_rate=400, labels=labels,
                  fname=os.path.join(tmp, 'kiviat.mp4'))
    kiviat.f_hops(fname=os.path.join(tmp, 'kiviat.mp4'))


@patch("matplotlib.pyplot.show")
def test_kiviat_plot(mock_show, kiviat_data, tmp):
    kiviat, labels = kiviat_data
    kiviat.plot(fname=os.path.join(tmp, 'kiviat.pdf'))
    kiviat.plot()


def test_pdf_1D(tmp):
    pdf(data[:, 5].reshape(-1, 1), fname=os.path.join(tmp, 'pdf.pdf'))


@patch("matplotlib.pyplot.show")
def test_pdf_surrogate(mock_show, ishigami_data):
    dist = ot.ComposedDistribution(ishigami_data[1], ot.IndependentCopula(3))
    f_3d = ishigami_data[0]
    space = ishigami_data[5]
    surrogate = SurrogateModel('kriging', space.corners)
    surrogate.fit(space, f_3d(space))
    settings = {
        "dist": dist,
        "model": surrogate,
        "method": 'kriging',
        "bounds": space.corners
    }
    pdf(settings)


@patch("matplotlib.pyplot.show")
def test_pdf_nD(mock_show, tmp):
    fig_pdf = pdf(data, xdata=np.linspace(1, 12, 12),
                  fname=os.path.join(tmp, 'pdf_nd.pdf'))
    reshow(fig_pdf)
    plt.plot([0, 10], [25, 25])
    plt.show()


def test_pdf_nD_moments(tmp):
    pdf(data, moments=True,
        fname=os.path.join(tmp, 'pdf_nd_moments.pdf'))


@patch("matplotlib.pyplot.show")
def test_sobols_aggregated(mock_show, tmp):
    fun = Ishigami()
    indices = [fun.s_first, fun.s_total]
    fig = sobol(indices, conf=0.05)
    fig = reshow(fig[0])
    plt.plot([0, 10], [0.5, 0.5])
    fig.show()
    sobol(indices, p_lst=['x1', 't', 'y'], fname=os.path.join(tmp, 'sobol.pdf'))


@patch("matplotlib.pyplot.show")
def test_sobols_map(mock_show, tmp):
    fun = Mascaret()
    indices = [fun.s_first, fun.s_total, fun.s_first_full, fun.s_total_full]
    sobol(indices)
    sobol(indices, p_lst=['Ks', 'Q'],
          xdata=fun.x, fname=os.path.join(tmp, 'sobol_map.pdf'))


@patch("matplotlib.pyplot.show")
def test_response_surface_1D(mock_show, tmp):
    def fun(x):
        return x ** 2
    bounds = [[-7], [10]]
    path = os.path.join(tmp, 'rs_1D_vector.pdf')
    response_surface(bounds=bounds, fun=fun, fname=path)

    xdata = np.linspace(0, 1, 10)

    def fun(x):
        return (xdata * x) ** 2
    sample = np.array(range(5)).reshape(-1, 1)
    data = fun(sample)
    response_surface(bounds=bounds, sample=sample, data=data, xdata=xdata)


@pytest.mark.xfail(raises=ValueError)
@patch("matplotlib.pyplot.show")
def test_response_surface_2D_scalar(mock_show, branin_data):
    space = branin_data[5]
    data = branin_data[6]
    fun = branin_data[0]
    bounds = [[-7, 0], [10, 15]]
    response_surface(bounds=bounds, sample=space, data=data)
    response_surface(bounds=bounds, fun=fun, doe=space, resampling=4)


@patch("matplotlib.pyplot.show")
def test_response_surface_2D_vector(mock_show, mascaret_data, tmp):
    space = mascaret_data[5]
    data = mascaret_data[6]
    xdata = mascaret_data[0].x
    bounds = [[15.0, 2500.0], [60, 6000.0]]
    response_surface(bounds=bounds, sample=space, data=data, xdata=xdata)
    path = os.path.join(tmp, 'rs_2D_vector.pdf')
    response_surface(bounds=bounds, fun=mascaret_data[0], xdata=xdata,
                     plabels=['Ks', 'Q'], flabel='Z', fname=path)


@patch("matplotlib.pyplot.show")
def test_doe(mock_show, mascaret_data):
    space = mascaret_data[5]
    doe(space)


def test_doe_3D(ishigami_data, tmp):
    space = ishigami_data[5]
    fig, ax = doe(space, fname=os.path.join(tmp, 'DOE.pdf'))

    fig = reshow(fig)
    ax[0].plot([0, 6], [4, -3])
    fig.savefig(os.path.join(tmp, 'DOE_change.pdf'))


def test_doe_mufi(ishigami_data, tmp):
    space = ishigami_data[5]
    doe(space, multifidelity=True, fname=os.path.join(tmp, 'DOE_mufi.pdf'))


@patch("matplotlib.pyplot.show")
def test_corr_cov(mock_show, mascaret_data, tmp):
    fun = mascaret_data[0]
    dist = ot.ComposedDistribution(mascaret_data[1], ot.IndependentCopula(2))
    sample = np.array(ot.LHSExperiment(dist, 500).generate())
    data = fun(sample)
    corr_cov(data, sample, fun.x, interpolation='lanczos', plabels=['Ks', 'Q'])
    corr_cov(data, sample, fun.x, fname=os.path.join(tmp, 'corr_cov.pdf'))
