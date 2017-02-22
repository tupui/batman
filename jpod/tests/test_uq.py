# coding: utf8
import pytest
from jpod.uq import UQ
from jpod.surrogate import SurrogateModel
from jpod.tasks import Snapshot
from test_models import ishigami_data
from test_driver import settings


def test_indices(tmpdir_factory, ishigami_data):
    output = str(tmpdir_factory.mktemp('tmp_test'))
    _, _, _, _, _, space, target_space = ishigami_data

    Snapshot.initialize(settings['snapshot']['io'])
    surrogate = SurrogateModel('kriging', space.corners)
    surrogate.fit(space, target_space)

    analyse = UQ(surrogate, settings, output)

    indices = analyse.sobol()

    errors = analyse.error_model(indices, 'Ishigami')
    assert errors[0] == pytest.approx(1, 0.1)
    # 2nd order
    assert errors[2] == pytest.approx(0.2, 0.5)
    # 1st order
    assert errors[3] == pytest.approx(0.1, 0.1)
    # total order
    assert errors[4] == pytest.approx(0.1, 0.1)
