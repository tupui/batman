# coding: utf8
import pytest
import os
from jpod.uq import UQ
from jpod.surrogate import SurrogateModel
from jpod.tasks import Snapshot
from test_models import (ishigami_data, clean_output)
from test_driver import settings

output = './tmp_test'


def test_indices(ishigami_data, clean_output):
    os.mkdir(output)
    _, _, _, _, _, space, target_space = ishigami_data

    Snapshot.initialize(settings['snapshot']['io'])
    surrogate = SurrogateModel('kriging', space.corners)
    surrogate.fit(space, target_space)

    analyse = UQ(surrogate, settings, output)

    indices = analyse.sobol()

    errors = analyse.error_model(indices, 'Ishigami')
    assert errors[0] == pytest.approx(1, 0.1)
    # 2nd order
    assert errors[2] == pytest.approx(0.1, 0.1)
    # 1st order
    assert errors[3] == pytest.approx(0.1, 0.1)
    # total order
    assert errors[4] == pytest.approx(0.1, 0.1)
