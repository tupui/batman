# coding: utf8
import pytest
import os
import shutil
import numpy as np
import numpy.testing as npt
from jpod import Driver
from jpod.functions import Ishigami

output = './tmp_test'

f_ishigami = Ishigami()
def f(x):
    X1, X2, X3 = x
    return f_ishigami([X1, X2, X3])

settings = {
    "space": {
        "corners": [[-np.pi, -np.pi, -np.pi],[np.pi, np.pi, np.pi]],
        "sampling": {"init_size": 100,"method": "halton"},
        "resampling": {"delta_space": 0.08, "resamp_size": 1,
            "method": "sigma", "q2_criteria": 0.9}},
    "pod": { "dim_max": 100, "tolerance": 0.99, "server": None, "type": "static"},
    "snapshot": {"max_workers": 10,
        "io": {"shapes": {"0": [[1]]}, "format": "fmt_tp_fortran",
            "variables": ["F"], "point_filename": "header.py",
            "filenames": {"0": ["function.dat"]}, "template_directory": None,
            "parameter_names": ["x1", "x2", "x3"]},
        "provider": f},
    "surrogate": {"predictions": [[0, 2, 1]], "method": "kriging"},
    "uq": {
        "sample": 2000, "test": "Ishigami",
        "pdf": ["Uniform(-3.1415, 3.1415)", "Uniform(-3.1415, 3.1415)", "Uniform(-3.1415, 3.1415)"],
        "type": "aggregated","method": "sobol"}}


@pytest.fixture(autouse=True, scope="function")
def clean_output():
    try:
        shutil.rmtree(output)
    except OSError:
        pass


@pytest.fixture(scope="session")
def driver_init():
    driver = Driver(settings, output)
    driver.sampling()
    return driver


def test_driver_chain(driver_init):
    driver = driver_init
    driver.write()
    if not os.path.isdir(os.path.join(output, 'surrogate')):
        assert False

    driver.read()
    pred, _ = driver.prediction(write=True)
    if not os.path.isdir(os.path.join(output, 'predictions/Newsnap0000')):
        assert False

    target_point = f_ishigami([0, 2, 1])
    assert pred[0].data == pytest.approx(target_point, 0.1)


def test_provider_dict():   
    settings['space']['sampling']['init_size'] = 10 
    settings['snapshot']['provider'] = {
        "command": "bash", "timeout": 10, "context": "data",
        "script": "data/script.sh", "clean": False, "private-directory": "jpod-data",
        "data-directory": "cfd-output-data", "restart": "False"}
    driver = Driver(settings, output)
    driver.sampling()
    driver.write()


def test_resampling(driver_init):
    driver = driver_init
    driver.write()
    driver.resampling()
    driver.write()
