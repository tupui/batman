import numpy as np
import numpy.testing as npt
import pytest
from batman.pod import Pod


@pytest.fixture(scope="session")
def pod(tolerance=1, dim_max=3):
    snapshots = np.array([[37., 40., 41., 49., 42., 46., 45., 48.],
                          [40., 43., 47., 46., 41., 46., 45., 48.],
                          [40., 41., 42., 45., 44., 46., 45., 47.]])
    pod = Pod([[30, 30, 30], [40, 40, 40]], 8, tolerance, dim_max)
    pod._decompose(snapshots)

    return pod


def test_pod(pod):
    npt.assert_almost_equal(pod.mean_snapshot, [43.5, 44.5, 43.75], decimal=1)
    npt.assert_almost_equal(pod.S, [14.003, 4.747, 2.208], decimal=3)


def test_filtering(pod):
    U, S, V = pod.filtering(pod.U, pod.S, pod.V, 1, 3)
    npt.assert_almost_equal(U, pod.U)
    npt.assert_almost_equal(S, pod.S)
    npt.assert_almost_equal(V, pod.V)

    U, S, V = pod.filtering(pod.U, pod.S, pod.V, 1, 2)

    U_out = [[-0.776, 0.336], [-0.453, -0.886], [-0.439, 0.321]]
    S_out = [14.003, 4.747]
    V_out = [[0.623, 0.329, 0.112, -0.392, 0.189, -0.258, -0.138, -0.464],
             [0.126, -0.154, -0.762, 0.194, 0.564, 0.049, 0.097, -0.115]]

    npt.assert_almost_equal(U_out, U, decimal=3)
    npt.assert_almost_equal(S_out, S, decimal=3)
    npt.assert_almost_equal(V_out, V.T, decimal=3)


def test_low_rank_init():
    pod_1 = pod(1, 2)
    pod_2 = pod(0.8, 3)

    npt.assert_allclose(pod_1.S, pod_2.S)
    npt.assert_allclose(pod_1.U, pod_2.U)
    npt.assert_allclose(pod_1.V, pod_2.V)


def test_update(pod):
    snapshots = np.array([[37., 40., 41., 49., 42., 46., 45., 48.],
                          [40., 43., 47., 46., 41., 46., 45., 48.],
                          [40., 41., 42., 45., 44., 46., 45., 47.]])

    pod_empty = Pod([[30, 30, 30], [40, 40, 40]], 8, 1, 3)
    [pod_empty._update(snapshots[:, i]) for i in range(8)]

    npt.assert_almost_equal(abs(pod.U), abs(pod_empty.U), decimal=3)
    npt.assert_almost_equal(pod.S, pod_empty.S, decimal=3)
    npt.assert_almost_equal(abs(pod.V), abs(pod_empty.V), decimal=3)
    npt.assert_almost_equal(pod.mean_snapshot, [43.5, 44.5, 43.75], decimal=1)


def test_downsample(pod):
    snapshots = np.array([[37., 40., 41., 49., 42., 46., 45.],
                          [40., 43., 47., 46., 41., 46., 45.],
                          [40., 41., 42., 45., 44., 46., 45.]])

    pod_downsampled = Pod([[30, 30, 30], [40, 40, 40]], 8, 1, 3)
    pod_downsampled._decompose(snapshots)

    V_1 = np.delete(pod.V, 7, 0)
    U, S, V = pod.downgrade(pod.S, V_1)

    U = np.dot(pod.U, U)

    npt.assert_almost_equal(S, pod_downsampled.S, decimal=3)
    npt.assert_almost_equal(abs(V), abs(pod_downsampled.V), decimal=3)
    npt.assert_almost_equal(pod.mean_snapshot, [43.5, 44.5, 43.75], decimal=1)
