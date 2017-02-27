# coding: utf8
import pytest
from jpod.uq import UQ
from jpod.surrogate import SurrogateModel
from jpod.tasks import Snapshot


def test_indices(tmp, ishigami_data, settings_ishigami):
    _, _, _, _, _, space, target_space = ishigami_data

    Snapshot.initialize(settings_ishigami['snapshot']['io'])
    surrogate = SurrogateModel('kriging', space.corners)
    surrogate.fit(space, target_space)

    analyse = UQ(surrogate, settings_ishigami, tmp)

    indices = analyse.sobol()

    errors = analyse.error_model(indices, 'Ishigami')
    assert errors[0] == pytest.approx(1, 0.1)
    # 2nd order
    assert errors[2] == pytest.approx(0.1, abs=0.2)
    # 1st order
    assert errors[3] == pytest.approx(0.05, abs=0.05)
    # total order
    assert errors[4] == pytest.approx(0.05, abs=0.05)
