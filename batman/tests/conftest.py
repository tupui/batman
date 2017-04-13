# coding: utf8

import pytest
import numpy as np
import copy
from sklearn.metrics import r2_score
import openturns as ot
from batman.functions import (Ishigami, Mascaret)
from batman.functions import output_to_sequence
from batman.space import (Space, Point)
from batman import Driver


@pytest.fixture(scope="session")
def tmp(tmpdir_factory):
    """Create a common temp directory."""
    return str(tmpdir_factory.mktemp('tmp_test'))


@pytest.fixture(scope="session")
def settings_ishigami():
    f_ishigami = Ishigami()
    settings = {
        "space": {
            "corners": [[-np.pi, -np.pi, -np.pi],[np.pi, np.pi, np.pi]],
            "sampling": {"init_size": 150,"method": "halton"},
            "resampling": {"delta_space": 0.08, "resamp_size": 1,
                "method": "sigma", "q2_criteria": 0.9}},
        "pod": { "dim_max": 100, "tolerance": 0.99, "server": None, "type": "static"},
        "snapshot": {"max_workers": 10,
            "io": {"shapes": {"0": [[1]]}, "format": "fmt_tp_fortran",
                "variables": ["F"], "point_filename": "header.py",
                "filenames": {"0": ["function.dat"]}, "template_directory": None,
                "parameter_names": ["x1", "x2", "x3"]},
            "provider": f_ishigami},
        "surrogate": {"predictions": [[0, 2, 1]], "method": "kriging"},
        "uq": {
            "sample": 2000, "test": "Ishigami",
            "pdf": ["Uniform(-3.1415, 3.1415)", "Uniform(-3.1415, 3.1415)", "Uniform(-3.1415, 3.1415)"],
        "type": "aggregated","method": "sobol"}}
    return settings


@pytest.fixture(scope="session")
def driver_init(tmp, settings_ishigami):
    """Initialize driver with settings from Ishigami"""
    driver = Driver(settings_ishigami, tmp)
    driver.sampling()
    return driver


@pytest.fixture(scope="session")
def ishigami_data(settings_ishigami):
    f_3d = Ishigami()
    x1 = ot.Uniform(-3.1415, 3.1415)
    dists = [x1] * 3
    model = ot.PythonFunction(3, 1, output_to_sequence(f_3d))
    point = Point([2.20, 1.57, 3])
    target_point = f_3d(point)
    space = Space(settings_ishigami)
    space.sampling(150)
    target_space = f_3d(space)
    return (f_3d, dists, model, point, target_point, space, target_space)


@pytest.fixture(scope="session")
def mascaret_data(settings_ishigami):
    f = Mascaret()
    x1 = ot.Uniform(15., 60.)
    x2 = ot.Normal(4035., 400.)
    dists = [x1, x2]
    model = ot.PythonFunction(2, 14, f)
    point = [31.54, 4237.025]
    target_point = f(point)
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings["space"]["corners"] = [[15.0, 2500.0], [60, 6000.0]]
    test_settings["snapshot"]["io"]["parameter_names"] = ["Ks", "Q"]
    space = Space(test_settings)
    space.sampling(50)
    target_space = f(space)
    return (f, dists, model, point, target_point, space, target_space)


def sklearn_q2(dists, model, surrogate):
    dim = len(dists)
    dists = ot.ComposedDistribution(dists, ot.IndependentCopula(dim))
    experiment = ot.LHSExperiment(dists, 1000)
    sample = experiment.generate()
    ref = model(sample)
    pred = surrogate(sample)

    return r2_score(ref, pred, multioutput='uniform_average')