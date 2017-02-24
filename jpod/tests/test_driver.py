# coding: utf8
import pytest
import os
import copy
import openturns as ot
from jpod import Driver
from jpod.tests.conftest import sklearn_q2


def test_driver_init(driver_init):
    pass


def test_driver_chain(driver_init, tmp, ishigami_data):
    driver = driver_init
    driver.write()
    if not os.path.isdir(os.path.join(tmp, 'surrogate')):
        assert False

    driver.read()
    pred, _ = driver.prediction(write=True)
    if not os.path.isdir(os.path.join(tmp, 'predictions/Newsnap0000')):
        assert False

    data = ishigami_data
    f_ishigami = data[0]
    target_point = f_ishigami([0, 2, 1])
    assert pred[0].data == pytest.approx(target_point, 0.1)


def test_no_pod(ishigami_data, tmp, settings_ishigami):
    test_settings = copy.copy(settings_ishigami)
    test_settings.pop('pod')
    driver = Driver(test_settings, tmp)
    driver.sampling()

    f_3d, dists, model, point, target_point, space, target_space = ishigami_data

    pred, _ = driver.prediction(write=True, points=point)
    assert pred[0].data == pytest.approx(target_point, 0.1)
    if not os.path.isdir(os.path.join(tmp, 'predictions/Newsnap0000')):
        assert False

    def wrap_surrogate(x):
        evaluation, _ = driver.prediction(points=x)
        return [evaluation[0].data]
    surrogate_ot = ot.PythonFunction(3, 1, wrap_surrogate)
    q2 = sklearn_q2(dists, model, surrogate_ot)
    assert q2 == pytest.approx(1, 0.1)


def test_provider_dict(tmp, settings_ishigami):
    path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(path)
    test_settings = copy.copy(settings_ishigami)
    test_settings['space']['sampling']['init_size'] = 4
    test_settings['snapshot']['provider'] = {
        "command": "bash", "timeout": 30, "context": "data",
        "script": "data/script.sh", "clean": False, "private-directory": "jpod-data",
        "data-directory": "cfd-output-data", "restart": "False"}
    driver = Driver(test_settings, tmp)
    driver.sampling()
    driver.write()

    pred, _ = driver.prediction(write=True)
    if not os.path.isdir(os.path.join(tmp, 'predictions/Newsnap0000')):
        assert False


def test_resampling(tmp, settings_ishigami):
    test_settings = copy.copy(settings_ishigami)
    test_settings['space']['sampling']['init_size'] = 4
    driver = Driver(test_settings, tmp)
    driver.sampling()
    driver.resampling()


def test_uq(driver_init, tmp):
    driver = driver_init
    driver.uq()

    tmp = os.path.join(tmp, 'uq')

    if not os.path.isdir(tmp):
        assert False

    if not os.path.isfile(os.path.join(tmp, 'sensitivity.dat')):
        assert False

    if not os.path.isfile(os.path.join(tmp, 'moment.dat')):
        assert False

    if not os.path.isfile(os.path.join(tmp, 'pdf.dat')):
        assert False

    if not os.path.isfile(os.path.join(tmp, 'sensitivity_aggregated.dat')):
        assert False
