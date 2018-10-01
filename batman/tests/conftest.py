# coding: utf8

import pytest
import numpy as np
from sklearn.metrics import r2_score
import openturns as ot
from batman.functions import (Ishigami, Branin, G_Function,
                              db_Mascaret, Forrester)
from batman.space import Space
from batman.driver import Driver


class Datatest:
    """Wrap results."""

    def __init__(self, kwds):
        self.__dict__.update(kwds)


@pytest.fixture()
def seed():
    np.random.seed(123456)
    ot.RandomGenerator.SetSeed(123456)


@pytest.fixture(scope="module")
def tmp(tmpdir_factory):
    """Create a common temp directory."""
    return str(tmpdir_factory.mktemp('tmp_test'))


@pytest.fixture(scope='session')
def settings_ishigami():
    return {
        "space": {
            "corners": [
                [-np.pi, -np.pi, -np.pi],
                [np.pi, np.pi, np.pi]
            ],
            "sampling": {
                "init_size": 150,
                "method": "halton"
            },
            "resampling": {
                "delta_space": 0.08,
                "resamp_size": 1,
                "method": "sigma",
                "q2_criteria": 0.9
            }
        },
        "pod": {
            "dim_max": 100,
            "tolerance": 0.99,
            "server": None,
            "type": "static"
        },
        "snapshot": {
            "max_workers": 10,
            "plabels": ["x1", "x2", "x3"],
            "flabels": ["F"],
            "provider": {
                "type": "function",
                "module": "batman.tests.plugins",
                "function": "f_ishigami"
            },
            "io": {
                "space_fname": "sample-space.json",
                "space_format": "json",
                "data_fname": "sample-data.json",
                "data_format": "json",
            }
        },
        "surrogate": {
            "predictions": [[0, 2, 1]],
            "method": "kriging"
        },
        "uq": {
            "sample": 2000,
            "test": "Ishigami",
            "pdf": ["Uniform(-3.1415, 3.1415)",
                    "Uniform(-3.1415, 3.1415)",
                    "Uniform(-3.1415, 3.1415)"],
            "type": "aggregated",
            "method": "sobol"
        }
    }


@pytest.fixture(scope='module')
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
    data['point'] = [2.20, 1.57, 3]
    data['target_point'] = data['func'](data['point'])
    data['space'] = Space(settings_ishigami['space']['corners'],
                          settings_ishigami['space']['sampling']['init_size'],
                          settings_ishigami['space']['resampling']['resamp_size'],
                          settings_ishigami['snapshot']['plabels'])
    data['space'].sampling(150, settings_ishigami['space']['sampling']['method'])
    data['target_space'] = data['func'](data['space'])
    return Datatest(data)


@pytest.fixture(scope="session")
def branin_data(settings_ishigami):
    data = {}
    data['func'] = Branin()
    data['dists'] = [ot.Uniform(-5, 10), ot.Uniform(0, 15)]
    data['point'] = [2., 2.]
    data['target_point'] = data['func'](data['point'])
    data['space'] = Space([[-7, 0], [10, 15]],
                          settings_ishigami['space']['sampling']['init_size'],
                          settings_ishigami['space']['resampling']['resamp_size'],
                          ['x1', 'x2'])
    data['space'].sampling(10, kind='halton', discrete=0)
    data['target_space'] = data['func'](data['space'])
    return Datatest(data)


@pytest.fixture(scope='session')
def g_function_data(settings_ishigami):
    data = {}
    data['func'] = G_Function()
    data['dists'] = [ot.Uniform(0, 1)] * 4
    data['point'] = [0.5, 0.2, 0.7, 0.1]
    data['target_point'] = data['func'](data['point'])
    data['space'] = Space([[0, 0, 0, 0], [1, 1, 1, 1]],
                          settings_ishigami['space']['sampling']['init_size'],
                          settings_ishigami['space']['resampling']['resamp_size'],
                          ['x1', 'x2', 'x3', 'x4'])
    data['space'].sampling(10, kind='halton', discrete=2)
    data['target_space'] = data['func'](data['space'])
    return Datatest(data)


@pytest.fixture(scope='session')
def mascaret_data(settings_ishigami):
    data = {}
    fun = db_Mascaret()
    data['func'] = lambda x: fun(x).reshape(-1, 14)[:, 0:3]
    data['func'].x = fun.x[0:3]
    x1 = ot.Uniform(15., 60.)
    x2 = ot.Normal(4035., 400.)
    data['dists'] = [x1, x2]
    data['point'] = [31.54, 4237.025]
    data['target_point'] = data['func'](data['point'])[0]
    data['space'] = Space([[15.0, 2500.0], [60, 6000.0]],
                          settings_ishigami['space']['sampling']['init_size'],
                          settings_ishigami['space']['resampling']['resamp_size'],
                          ['Ks', 'Q'])
    data['space'].sampling(50, settings_ishigami['space']['sampling']['method'])
    data['target_space'] = data['func'](data['space'])
    return Datatest(data)


@pytest.fixture(scope='session')
def mufi_data(settings_ishigami):
    data = {}
    f_e = Forrester('e')
    f_c = Forrester('c')
    data['dists'] = [ot.Uniform(0.0, 1.0)]
    data['point'] = [0.65]
    data['target_point'] = f_e(data['point'])
    data['space'] = Space([[0.0], [1.0]],
                          10,
                          settings_ishigami['space']['resampling']['resamp_size'],
                          ['fidelity', 'x'], multifidelity=[5.1, 13.0])
    data['space'].sampling(10, 'halton')

    working_space = np.array(data['space'])

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
