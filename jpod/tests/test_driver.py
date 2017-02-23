# coding: utf8
import pytest
import os
import numpy as np
import openturns as ot
from jpod import Driver
from jpod.functions import Ishigami
from test_models import (ot_q2, ishigami_data)

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


@pytest.fixture(scope="session")
def driver_init(tmpdir_factory):
    output = str(tmpdir_factory.mktemp('tmp_test'))
    driver = Driver(settings, output)
    driver.sampling()
    return output, driver


def test_driver_init(driver_init):
    pass


def test_driver_chain(driver_init):
    output, driver = driver_init
    driver.write()
    if not os.path.isdir(os.path.join(output, 'surrogate')):
        assert False

    driver.read()
    pred, _ = driver.prediction(write=True)
    if not os.path.isdir(os.path.join(output, 'predictions/Newsnap0000')):
        assert False

    target_point = f_ishigami([0, 2, 1])
    assert pred[0].data == pytest.approx(target_point, 0.1)


def test_resampling(driver_init):
    output, driver = driver_init
    driver.resampling()


def test_no_pod(ishigami_data, driver_init):
    output, _ = driver_init
    settings.pop('pod')
    driver = Driver(settings, output)
    driver.sampling()

    f_3d, dists, model, point, target_point, space, target_space = ishigami_data

    pred, _ = driver.prediction(write=True, points=point)
    assert pred[0].data == pytest.approx(target_point, 0.1)
    if not os.path.isdir(os.path.join(output, 'predictions/Newsnap0000')):
        assert False

    def wrap_surrogate(x):
        evaluation, _ = driver.prediction(points=x)
        return [evaluation[0].data]
    surrogate_ot = ot.PythonFunction(3, 1, wrap_surrogate)
    q2 = ot_q2(dists, model, surrogate_ot)
    assert q2 == pytest.approx(1, 0.1)


def test_provider_dict(driver_init):
    path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(path)
    settings['space']['sampling']['init_size'] = 4
    settings['snapshot']['provider'] = {
        "command": "bash", "timeout": 30, "context": "data",
        "script": "data/script.sh", "clean": False, "private-directory": "jpod-data",
        "data-directory": "cfd-output-data", "restart": "False"}
    output, _ = driver_init
    driver = Driver(settings, output)
    driver.sampling()
    driver.write()

    pred, _ = driver.prediction(write=True)
    if not os.path.isdir(os.path.join(output, 'predictions/Newsnap0000')):
        assert False


def test_uq(driver_init):
    output, driver = driver_init
    driver.uq()

    output = os.path.join(output, 'uq')

    if not os.path.isdir(output):
        assert False

    if not os.path.isfile(os.path.join(output, 'sensitivity.dat')):
        assert False

    if not os.path.isfile(os.path.join(output, 'moment.dat')):
        assert False

    if not os.path.isfile(os.path.join(output, 'pdf.dat')):
        assert False

    if not os.path.isfile(os.path.join(output, 'sensitivity_aggregated.dat')):
        assert False
