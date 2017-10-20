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

    analyse = UQ(surrogate, nsample=settings_ishigami['uq']['sample'],
                 pdf=settings_ishigami['uq']['pdf'],
                 p_lst=settings_ishigami['snapshot']['io']['parameter_names'],
                 method=settings_ishigami['uq']['method'],
                 indices=settings_ishigami['uq']['type'],
                 test=settings_ishigami['uq']['test'])

    indices = analyse.sobol()
    errors = analyse.error_model(indices, 'Ishigami')
    assert errors[0] == pytest.approx(1, 0.2)
    # 2nd order
    assert errors[2] == pytest.approx(0.1, abs=0.2)
    # 1st order
    assert errors[3] == pytest.approx(0.05, abs=0.05)
    # total order
    assert errors[4] == pytest.approx(0.05, abs=0.05)


def test_block(mascaret_data, settings_ishigami):
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['uq']['type'] = 'block'
    test_settings['uq'].pop('test')
    test_settings['snapshot']['io']['shape'] = {"0": [[14]]}
    test_settings['snapshot']['io']['parameter_names'] = ['Ks', 'Q']

    Snapshot.initialize(test_settings['snapshot']['io'])
    surrogate = SurrogateModel('rbf', mascaret_data.space.corners)
    surrogate.fit(mascaret_data.space, mascaret_data.target_space)

    analyse = UQ(surrogate, nsample=test_settings['uq']['sample'],
                 pdf=['Uniform(15., 60.)', 'Normal(4035., 400.)'],
                 p_lst=test_settings['snapshot']['io']['parameter_names'],
                 method=test_settings['uq']['method'],
                 indices=test_settings['uq']['type'])

    indices = analyse.sobol()
    print(indices)
