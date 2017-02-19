# coding: utf8
import pytest
import numpy as np
import numpy.testing as npt
import openturns as ot
from sklearn.metrics import r2_score
from jpod.functions import (Ishigami, Mascaret)
from jpod.surrogate import (PC, Kriging)
from jpod.space import Space, Point
from jpod.functions import output_to_sequence

settings = {
    "space": {
        "corners": [[-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]],
        "sampling": {
            "init_size": 200,
            "method": "halton"
        }
    }
}

def ot_q2(dists, model, surrogate):
    dim = len(dists)
    dists = ot.ComposedDistribution(dists, ot.IndependentCopula(dim))
    experiment = ot.LHSExperiment(dists, 1000)
    sample = experiment.generate()
    ref = model(sample)
    pred = surrogate(sample)

    return r2_score(ref, pred, multioutput='uniform_average')


def test_PC_1d():
    f_3d = Ishigami()
    point = [2.20, 1.57, 3]
    target = f_3d(point)
    x1 = ot.Uniform(-3.1415, 3.1415)
    dists = [x1] * 3

    surrogate = PC(function=f_3d, input_dists=dists,
                   out_dim=1, n_sample=1000, total_deg=10, strategy='LS')
    pred = np.array(surrogate.evaluate(point))
    assert pred == pytest.approx(target, 0.01)

    surrogate = PC(function=f_3d, input_dists=dists,
                   out_dim=1, total_deg=10, strategy='Quad')
    pred = np.array(surrogate.evaluate(point))
    assert pred == pytest.approx(target, 0.01)

    # Compute predictivity coefficient Q2
    model = ot.PythonFunction(3, 1, output_to_sequence(f_3d))
    surrogate = ot.PythonFunction(3, 1, surrogate.evaluate)

    q2 = ot_q2(dists, model, surrogate)

    assert q2 == pytest.approx(1, 0.1)


def test_PC_14d():
    f = Mascaret()
    point = [31.54, 4237.025]
    target = f(point)
    x1 = ot.Uniform(15., 60.)
    x2 = ot.Normal(4035., 400.)
    dists = [x1, x2]

    surrogate = PC(function=f, input_dists=dists,
                   out_dim=14, n_sample=300, total_deg=10,  strategy='LS')
    pred = np.array(surrogate.evaluate(point))
    test_output = npt.assert_almost_equal(target, pred, decimal=2)
    assert True if test_output is None else False

    surrogate = PC(function=f, input_dists=dists,
                   out_dim=14, total_deg=11,  strategy='Quad')

    pred = np.array(surrogate.evaluate(point))
    test_output = npt.assert_almost_equal(target, pred, decimal=2)
    assert True if test_output is None else False

    # Compute predictivity coefficient Q2
    model = ot.PythonFunction(2, 14, f)
    surrogate = ot.PythonFunction(2, 14, surrogate.evaluate)
    q2 = ot_q2(dists, model, surrogate)

    assert q2 == pytest.approx(1, 0.1)


def test_GP_1d():
    f_3d = Ishigami()
    point = Point([0.20, 1.57, -1.4])
    target = f_3d(point)
    space = Space(settings)
    space.sampling(150)
    y = f_3d(space)

    x1 = ot.Uniform(-3.1415, 3.1415)
    dists = [x1] * 3

    surrogate = Kriging(space, y)
    pred, _ = np.array(surrogate.evaluate(point))

    assert pred == pytest.approx(target, 0.1)

    # Compute predictivity coefficient Q2
    model = ot.PythonFunction(3, 1, output_to_sequence(f_3d))

    def wrap_surrogate(x):
        evaluation, _ = surrogate.evaluate(x)
        return [evaluation]
    surrogate_ot = ot.PythonFunction(3, 1, wrap_surrogate)

    q2 = ot_q2(dists, model, surrogate_ot)

    assert q2 == pytest.approx(1, 0.1)


def test_GP_14d():
    f = Mascaret()
    point = [31.54, 4237.025]
    target = f(point)
    x1 = ot.Uniform(15., 60.)
    x2 = ot.Normal(4035., 400.)
    dists = [x1, x2]

    settings["space"]["corners"] = [[15.0, 2500.0], [60, 6000.0]]
    space = Space(settings)
    space.sampling(50)
    y = f(space)

    surrogate = Kriging(space, y)
    pred, _ = np.array(surrogate.evaluate(point))

    print(target.shape)
    print(pred.shape)

    test_output = npt.assert_almost_equal(target, pred, decimal=0)
    assert True if test_output is None else False

    # Compute predictivity coefficient Q2
    model = ot.PythonFunction(2, 14, f)
    def wrap_surrogate(x):
        evaluation, _ = surrogate.evaluate(x)
        return evaluation
    surrogate_ot = ot.PythonFunction(2, 14, wrap_surrogate)
    q2 = ot_q2(dists, model, surrogate_ot)

    assert q2 == pytest.approx(1, 0.1)
