# coding: utf8
import pytest
import os
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
import openturns as ot
from batman.surrogate import (PC, Kriging, RBFnet, Evofusion, SurrogateModel)
from batman.tasks import Snapshot
from batman.tests.conftest import sklearn_q2


def test_PC_1d(ishigami_data):
    f_3d, dists, model, point, target_point, space, target_space = ishigami_data

    surrogate = PC(function=f_3d, input_dists=dists,
                   out_dim=1, n_sample=1000, total_deg=10, strategy='LS')
    pred = np.array(surrogate.evaluate(point))
    assert pred == pytest.approx(target_point, 0.01)

    surrogate = PC(function=f_3d, input_dists=dists,
                   out_dim=1, total_deg=10, strategy='Quad')
    pred = np.array(surrogate.evaluate(point))
    assert pred == pytest.approx(target_point, 0.01)

    # Test space evaluation
    pred = np.array(surrogate.evaluate(space))
    npt.assert_almost_equal(target_space, pred, decimal=1)

    # Compute predictivity coefficient Q2
    surrogate = ot.PythonFunction(3, 1, surrogate.evaluate)
    q2 = sklearn_q2(dists, model, surrogate)
    assert q2 == pytest.approx(1, 0.1)


def test_GP_1d(ishigami_data):
    f_3d, dists, model, point, target_point, space, target_space = ishigami_data

    surrogate = Kriging(space, target_space)

    # Test one point evaluation
    pred, _ = np.array(surrogate.evaluate(point))
    assert pred == pytest.approx(target_point, 0.1)

    # Test space evaluation
    pred, _ = np.array(surrogate.evaluate(space))
    npt.assert_almost_equal(target_space, pred, decimal=1)

    # Compute predictivity coefficient Q2
    def wrap_surrogate(x):
        evaluation, _ = surrogate.evaluate(x)
        return [evaluation]
    surrogate_ot = ot.PythonFunction(3, 1, wrap_surrogate)
    q2 = sklearn_q2(dists, model, surrogate_ot)
    assert q2 == pytest.approx(1, 0.1)


def test_RBFnet_1d(ishigami_data):
    f_3d, dists, model, point, target_point, space, target_space = ishigami_data

    surrogate = RBFnet(space, target_space)

    # Test one point evaluation
    pred = np.array(surrogate.evaluate(point))
    assert pred == pytest.approx(target_point, 0.3)

    # Test space evaluation
    pred = np.array(surrogate.evaluate(space))
    npt.assert_almost_equal(target_space, pred, decimal=1)

    # Compute predictivity coefficient Q2
    def wrap_surrogate(x):
        evaluation = surrogate.evaluate(x)
        return [evaluation]
    surrogate_ot = ot.PythonFunction(3, 1, wrap_surrogate)
    q2 = sklearn_q2(dists, model, surrogate_ot)
    assert q2 == pytest.approx(0.86, 0.1)

    surrogate = RBFnet(space, target_space, regtree=1)


def test_PC_14d(mascaret_data):
    f, dists, model, point, target_point, space, target_space = mascaret_data

    surrogate = PC(function=f, input_dists=dists,
                   out_dim=14, n_sample=300, total_deg=10,  strategy='LS')
    pred = np.array(surrogate.evaluate(point)).reshape(14)
    npt.assert_almost_equal(target_point, pred, decimal=2)

    surrogate = PC(function=f, input_dists=dists,
                   out_dim=14, total_deg=11,  strategy='Quad')

    # Test point evaluation
    pred = np.array(surrogate.evaluate(point)).reshape(14)
    npt.assert_almost_equal(target_point, pred, decimal=2)

    # Test space evaluation
    pred = np.array(surrogate.evaluate(space))
    npt.assert_almost_equal(target_space, pred, decimal=0)

    # Compute predictivity coefficient Q2
    surrogate_ot = ot.PythonFunction(2, 14, surrogate.evaluate)
    q2 = sklearn_q2(dists, model, surrogate_ot)
    assert q2 == pytest.approx(1, 0.1)


def test_GP_14d(mascaret_data):
    f, dists, model, point, target_point, space, target_space = mascaret_data

    surrogate = Kriging(space, target_space)

    # Test point evaluation
    pred, _ = np.array(surrogate.evaluate(point))
    npt.assert_almost_equal(target_point, pred, decimal=1)

    # Test space evaluation
    pred, _ = np.array(surrogate.evaluate(space))
    npt.assert_almost_equal(target_space, pred, decimal=1)

    # Compute predictivity coefficient Q2
    model = ot.PythonFunction(2, 14, f)

    def wrap_surrogate(x):
        evaluation, _ = surrogate.evaluate(x)
        return evaluation
    surrogate_ot = ot.PythonFunction(2, 14, wrap_surrogate)
    q2 = sklearn_q2(dists, model, surrogate_ot)
    assert q2 == pytest.approx(1, 0.1)


def test_SurrogateModel_class(tmp, ishigami_data, settings_ishigami):
    f_3d, dists, model, point, target_point, space, target_space = ishigami_data

    Snapshot.initialize(settings_ishigami['snapshot']['io'])

    surrogate = SurrogateModel('pc', space.corners)
    surrogate.fit(space, target_space)
    pred, sigma = surrogate(point)
    assert sigma is None
    assert pred[0] == pytest.approx(target_point, 0.5)
    surrogate.write(tmp)
    if not os.path.isfile(os.path.join(tmp, 'surrogate.dat')):
        assert False

    surrogate = SurrogateModel('kriging', space.corners)
    surrogate.fit(space, target_space)
    surrogate.write(tmp)
    if not os.path.isfile(os.path.join(tmp, 'surrogate.dat')):
        assert False

    surrogate = SurrogateModel('kriging', space.corners)
    surrogate.read(tmp)
    assert surrogate.predictor is not None
    assert surrogate.space == space

    pred, _ = surrogate(point)
    assert pred[0] == pytest.approx(target_point, 0.1)

    pred, _ = surrogate(point)
    assert pred[0] == pytest.approx(target_point, 0.1)

    pred, _ = surrogate(point, path=tmp)
    assert pred[0].data == pytest.approx(target_point, 0.1)
    if not os.path.isdir(os.path.join(tmp, 'Newsnap0')):
        assert False

    # Compute predictivity coefficient Q2
    def wrap_surrogate(x):
        evaluation, _ = surrogate(x)
        return [evaluation]
    surrogate_ot = ot.PythonFunction(3, 1, wrap_surrogate)
    q2 = sklearn_q2(dists, model, surrogate_ot)
    assert q2 == pytest.approx(1, 0.1)


def test_quality(tmp, mufi_data):
    _, _, _, _, _, _, space, target_space = mufi_data

    surrogate = SurrogateModel('kriging', space.corners)

    # Split into cheap and expensive arrays
    space = np.array(space)
    target_space = np.array(target_space)
    space = [space[space[:, 0] == 0][:, 1],
             space[space[:, 0] == 1][:, 1]]
    n_e = space[0].shape[0]
    n_c = space[1].shape[0]
    space = [space[0].reshape((n_e, -1)),
             space[1].reshape((n_c, -1))]
    target_space = [target_space[:n_e].reshape((n_e, -1)),
                    target_space[n_e:].reshape((n_c, -1))]

    surrogate.fit(space[1], target_space[1])

    assert surrogate.estimate_quality()[0] == pytest.approx(1, 0.1)


def test_evofusion(mufi_data):
    f_e, f_c, dist, model, point, target_point, space, target_space = mufi_data

    surrogate = Evofusion(space, target_space)

    # Test one point evaluation
    pred, _ = np.array(surrogate.evaluate(point))
    assert pred == pytest.approx(target_point, 0.1)

    # Compute predictivity coefficient Q2
    def wrap_surrogate(x):
        evaluation, _ = surrogate.evaluate([x])
        return [evaluation]
    surrogate_ot = ot.PythonFunction(1, 1, wrap_surrogate)
    q2 = sklearn_q2(dist, model, surrogate_ot)
    assert q2 == pytest.approx(1, 0.1)

    # Plotting
    x = np.linspace(0, 1, 200).reshape(-1, 1)

    # Split into cheap and expensive arrays
    space = np.array(space)
    target_space = np.array(target_space)
    space = [space[space[:, 0] == 0][:, 1],
             space[space[:, 0] == 1][:, 1]]
    n_e = space[0].shape[0]
    n_c = space[1].shape[0]
    space = [space[0].reshape((n_e, -1)),
             space[1].reshape((n_c, -1))]
    target_space = [target_space[:n_e].reshape((n_e, -1)),
                    target_space[n_e:].reshape((n_c, -1))]

    surrogate_e = Kriging(space[0], target_space[0])
    surrogate_c = Kriging(space[1], target_space[1])
    pred_evo, _ = np.array(surrogate.evaluate(x))
    pred_e, _ = np.array(surrogate_e.evaluate(x))
    pred_c, _ = np.array(surrogate_c.evaluate(x))

    # Plotting
    fig = plt.figure("Evofusion on Forrester's functions")
    plt.plot(space[0], target_space[0], 'o', label=r'$y_e$')
    plt.plot(space[1], target_space[1], '^', label=r'$y_c$')
    plt.plot(x, f_e(x), ls='-', label=r'$f_e$')
    plt.plot(x, f_c(x), ls='--', label=r'$f_c$')
    plt.plot(x, pred_evo, ls='-.', label=r'$evofusion$')
    plt.plot(x, pred_e, '>', markevery=20, ls=':', label=r'kriging through $y_e$')
    plt.plot(x, pred_c, '<', markevery=20, ls=':', label=r'kriging through $y_c$')
    plt.xlabel('x', fontsize=28)
    plt.ylabel('y', fontsize=28)
    plt.tick_params(axis='x', labelsize=26)
    plt.tick_params(axis='y', labelsize=26)
    plt.legend(fontsize=26, loc='upper left')
    fig.tight_layout()
    path = 'evofusion_forrester.pdf'
    fig.savefig(path, transparent=True, bbox_inches='tight')
    # plt.show()
    plt.close('all')
