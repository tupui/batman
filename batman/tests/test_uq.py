# coding: utf8
import pytest
from batman.uq import UQ
from batman.surrogate import SurrogateModel
from batman.tasks import Snapshot


def test_indices(tmp, ishigami_data, settings_ishigami):
    Snapshot.initialize(settings_ishigami['snapshot']['io'])
    surrogate = SurrogateModel('kriging', ishigami_data.space.corners)
    surrogate.fit(ishigami_data.space, ishigami_data.target_space)

    analyse = UQ(settings_ishigami, surrogate, tmp)

    indices = analyse.sobol()

    errors = analyse.error_model(indices, 'Ishigami')
    assert errors[0] == pytest.approx(1, 0.1)
    # 2nd order
    assert errors[2] == pytest.approx(0.1, abs=0.2)
    # 1st order
    assert errors[3] == pytest.approx(0.05, abs=0.05)
    # total order
    assert errors[4] == pytest.approx(0.05, abs=0.05)
