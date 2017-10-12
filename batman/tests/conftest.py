# coding: utf8

import pytest
import numpy as np
import copy
from sklearn.metrics import r2_score
import openturns as ot
from batman.functions import (Ishigami, Michalewicz, Branin, G_Function
                              Mascaret, Forrester)
from batman.functions import output_to_sequence
from batman.space import (Space, Point)
from batman.driver import Driver


class Datatest(object):

    """Wrap results."""

    def __init__(self, kwds):
        self.__dict__.update(kwds)


@pytest.fixture(scope="session")
def tmp(tmpdir_factory):
    """Create a common temp directory."""
    return str(tmpdir_factory.mktemp('tmp_test'))


@pytest.fixture(scope='session')
def settings_ishigami():
    f_ishigami = Ishigami()
    settings = {
        "space": {
            "corners": [[-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]],
            "sampling": {"init_size": 150, "method": "halton"},
            "resampling": {"delta_space": 0.08, "resamp_size": 1,
                           "method": "sigma", "q2_criteria": 0.9}},
        "pod": {"dim_max": 100, "tolerance": 0.99, "server": None, "type": "static"},
        "snapshot": {"max_workers": 10,
                     "io": {"shapes": {"0": [[1]]}, "format": "fmt_tp_fortran",
                            "variables": ["F"], "point_filename": "header.py",
                            "filenames": {"0": ["function.dat"]},
                            "template_directory": None,
                            "parameter_names": ["x1", "x2", "x3"]},
                     "provider": f_ishigami},
        "surrogate": {"predictions": [[0, 2, 1]], "method": "kriging"},
        "uq": {
            "sample": 2000, "test": "Ishigami",
            "pdf": ["Uniform(-3.1415, 3.1415)", "Uniform(-3.1415, 3.1415)",
                    "Uniform(-3.1415, 3.1415)"],
            "type": "aggregated", "method": "sobol"}}
    return settings


@pytest.fixture(scope='session')
def driver_init(tmp, settings_ishigami):
    """Initialize driver with settings from Ishigami."""
    driver = Driver(settings_ishigami, tmp)
    driver.sampling()
    return driver


@pytest.fixture(scope='session')
def ishigami_data(settings_ishigami):
    data = {}
    data['func'] = Ishigami()
    x1 = ot.Uniform(-3.1415, 3.1415)
    data['dists'] = [x1] * 3
    data['point'] = Point([2.20, 1.57, 3])
    data['target_point'] = data['func'](data['point'])
    data['space'] = Space(settings_ishigami)
    data['space'].sampling(150)
    data['target_space'] = data['func'](data['space'])
    return Datatest(data)


@pytest.fixture(scope="session")
def branin_data(settings_ishigami):
    data = {}
    data['func'] = Branin()
    data['dists'] = [ot.Uniform(-5, 10), ot.Uniform(0, 15)]
    data['point'] = Point([2., 2.])
    data['target_point'] = data['func'](data['point'])
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['space']['corners'] = [[-7, 0], [10, 15]]
    test_settings['space']['sampling']['method'] = 'discrete'
    test_settings['snapshot']['io']['parameter_names'] = ['x1', 'x2']
    data['space'] = Space(test_settings)
    data['space'].sampling(10)
    data['target_space'] = data['func'](data['space'])
    return Datatest(data)


@pytest.fixture(scope='session')
def g_function_data(settings_ishigami):
    data = {}
    data['func'] = G_Function()
    data['dists'] = [ot.Uniform(0, 1)] * 4
    data['point'] = Point([0.5, 0.2, 0.7, 0.1])
    data['target_point'] = data['func'](data['point'])
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['space']['corners'] = [[0, 1]] * 4
    test_settings['space']['sampling']['method'] = 'discrete'
    test_settings['snapshot']['io']['parameter_names'] = ['x1', 'x2', 'x3', 'x4']
    data['space'] = Space(test_settings)
    data['space'].sampling(10)
    data['target_space'] = data['func'](data['space'])
    return Datatest(data)


@pytest.fixture(scope='session')
def mascaret_data(settings_ishigami):
    data = {}
    data['func'] = Mascaret()
    x1 = ot.Uniform(15., 60.)
    x2 = ot.Normal(4035., 400.)
    data['dists'] = [x1, x2]
    data['point'] = [31.54, 4237.025]
    data['target_point'] = data['func'](data['point'])
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['space']['corners'] = [[15.0, 2500.0], [60, 6000.0]]
    test_settings['snapshot']['io']['parameter_names'] = ['Ks', 'Q']
    data['space'] = Space(test_settings)
    data['space'].sampling(50)
    data['target_space'] = data['func'](data['space'])
    return Datatest(data)


@pytest.fixture(scope='session')
def mufi_data(settings_ishigami):
    f_e = Forrester('e')
    f_c = Forrester('c')
    data['dists'] = [ot.Uniform(0.0, 1.0)]
    data['point'] = [0.65]
    data['target_point'] = f_e(data['point'])
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['space']['corners'] = [[0.0], [1.0]]
    test_settings['space']['sampling']['init_size'] = 10
    test_settings['snapshot']['io']['parameter_names'] = ['fidelity', 'x']
    test_settings['surrogate']['method'] = 'evofusion'
    test_settings['surrogate']['cost_ratio'] = 5.1
    test_settings['surrogate']['grand_cost'] = 13.0

    data['space'] = Space(test_settings)
    data['space'].sampling()

    working_space = np.array(space)

    data['target_space'] = np.vstack([f_e(working_space[working_space[:, 0] == 0][:, 1:]),
                                      f_c(working_space[working_space[:, 0] == 1][:, 1:])])
    data['func'] = [f_e, f_c]

    return Datatest(data)


def sklearn_q2(dists, model, surrogate):
    dists = ot.ComposedDistribution(dists)
    experiment = ot.LHSExperiment(dists, 1000)
    sample = np.array(experiment.generate())
    ref = model(sample)
    pred = surrogate(sample)

    return r2_score(ref, pred, multioutput='uniform_average')
