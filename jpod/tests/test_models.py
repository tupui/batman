# coding: utf8
import pytest
import numpy as np
import numpy.testing as npt
import openturns as ot
from pyuq import (Ishigami, Mascaret, PC)


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
    model = ot.PythonFunction(3, 1, f_3d)
    surrogate = ot.PythonFunction(3, 1, surrogate.evaluate)

    dists = ot.ComposedDistribution(dists, ot.IndependentCopula(3))
    experiment = ot.LHSExperiment(dists, 1000)
    sample = experiment.generate()
    ref = model(sample)

    val = ot.MetaModelValidation(sample, ref, surrogate)
    q2 = val.computePredictivityFactor()

    assert q2 == pytest.approx(1, 0.01)


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
                   out_dim=14, total_deg=10,  strategy='Quad')

    pred = np.array(surrogate.evaluate(point))
    test_output = npt.assert_almost_equal(target, pred, decimal=2)
    assert True if test_output is None else False

    # Compute predictivity coefficient Q2
    model = ot.PythonFunction(2, 14, f)
    surrogate = ot.PythonFunction(2, 14, surrogate.evaluate)

    dists = ot.ComposedDistribution(dists, ot.IndependentCopula(2))
    experiment = ot.LHSExperiment(dists, 1000)
    sample = experiment.generate()

    ref = np.array(model(sample))
    pred = np.array(surrogate(sample))
    mean = ref.mean(axis=1).reshape(-1, 1)
    var = ((pred - mean) ** 2).sum(axis=0)
    mse = ((pred - ref) ** 2).sum(axis=0)

    q2 = (np.ones(14) - mse / var).mean()

    assert q2 == pytest.approx(1, 0.01)
