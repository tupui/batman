# coding: utf8
import pytest
import os
import numpy as np
import numpy.testing as npt
import openturns as ot
from batman.surrogate import (PC, Kriging, SurrogateModel)
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
    test_output = npt.assert_almost_equal(target_space, pred, decimal=1)
    assert True if test_output is None else False

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
    test_output = npt.assert_almost_equal(target_space, pred, decimal=1)
    assert True if test_output is None else False

    # Compute predictivity coefficient Q2
    def wrap_surrogate(x):
        evaluation, _ = surrogate.evaluate(x)
        return [evaluation]
    surrogate_ot = ot.PythonFunction(3, 1, wrap_surrogate)
    q2 = sklearn_q2(dists, model, surrogate_ot)
    assert q2 == pytest.approx(1, 0.1)


def test_PC_14d(mascaret_data):
    f, dists, model, point, target_point, space, target_space = mascaret_data

    surrogate = PC(function=f, input_dists=dists,
                   out_dim=14, n_sample=300, total_deg=10,  strategy='LS')
    pred = np.array(surrogate.evaluate(point)).reshape(14)
    test_output = npt.assert_almost_equal(target_point, pred, decimal=2)
    assert True if test_output is None else False

    surrogate = PC(function=f, input_dists=dists,
                   out_dim=14, total_deg=11,  strategy='Quad')

    # Test point evaluation
    pred = np.array(surrogate.evaluate(point)).reshape(14)
    test_output = npt.assert_almost_equal(target_point, pred, decimal=2)
    assert True if test_output is None else False

    # Test space evaluation
    pred = np.array(surrogate.evaluate(space))
    test_output = npt.assert_almost_equal(target_space, pred, decimal=0)
    assert True if test_output is None else False

    # Compute predictivity coefficient Q2
    surrogate_ot = ot.PythonFunction(2, 14, surrogate.evaluate)
    q2 = sklearn_q2(dists, model, surrogate_ot)
    assert q2 == pytest.approx(1, 0.1)


def test_GP_14d(mascaret_data):
    f, dists, model, point, target_point, space, target_space = mascaret_data

    surrogate = Kriging(space, target_space)

    # Test space evaluation
    pred, _ = np.array(surrogate.evaluate(point))
    test_output = npt.assert_almost_equal(target_point, pred, decimal=1)
    assert True if test_output is None else False

    # Test space evaluation
    pred, _ = np.array(surrogate.evaluate(space))
    test_output = npt.assert_almost_equal(target_space, pred, decimal=1)
    assert True if test_output is None else False

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
    assert pred[0].data == pytest.approx(target_point, 0.1)

    pred, _ = surrogate(point, snapshots=False)
    assert pred == pytest.approx(target_point, 0.1)

    pred, _ = surrogate(point, path=tmp)
    assert pred[0].data == pytest.approx(target_point, 0.1)
    if not os.path.isdir(os.path.join(tmp, 'Newsnap0000')):
        assert False

    # Compute predictivity coefficient Q2
    def wrap_surrogate(x):
        evaluation, _ = surrogate(x)
        return [evaluation[0].data]
    surrogate_ot = ot.PythonFunction(3, 1, wrap_surrogate)
    q2 = sklearn_q2(dists, model, surrogate_ot)
    assert q2 == pytest.approx(1, 0.1)
