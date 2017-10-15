# coding: utf8
import copy
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
    assert errors[0] == pytest.approx(1, 0.2)
    # 2nd order
    assert errors[2] == pytest.approx(0.1, abs=0.2)
    # 1st order
    assert errors[3] == pytest.approx(0.05, abs=0.05)
    # total order
    assert errors[4] == pytest.approx(0.05, abs=0.05)


def test_block(tmp, mascaret_data, settings_ishigami):
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['uq']['type'] = 'block'
    test_settings['uq'].pop('test')
    test_settings['snapshot']['io']['shape'] = {"0": [[14]]}
    test_settings['snapshot']['io']['parameter_names'] = ['Ks', 'Q']

    Snapshot.initialize(test_settings['snapshot']['io'])
    surrogate = SurrogateModel('rbf', mascaret_data.space.corners)
    surrogate.fit(mascaret_data.space, mascaret_data.target_space)

    analyse = UQ(test_settings, surrogate, tmp)

    indices = analyse.sobol()
    print(indices)
