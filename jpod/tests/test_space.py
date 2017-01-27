# coding: utf8

import pytest
from jpod.space import *

settings = {
    "space": {
        "corners": [[1.0, 1.0, 1.0], [3.1415, 3.1415, 3.1415]],
        "sampling": {
            "init_size": 10,
            "method": "halton"
        },
        "resampling": {
            "delta_space": 0.08,
            "resamp_size": 6,
            "method": "sigma",
            "hybrid": [["sigma", 4], ["loo_sobol", 2]],
            "q2_criteria": 0.8
        }
    }
}

def test_space():

    space = Space(settings)

    space += (1, 2, 3)
    assert space[:] == [(1, 2, 3)]

    space.empty()
    assert space[:] == []

    space += [(1, 2, 3), (1, 1, 3)]
    assert space[:] == [(1, 2, 3), (1, 1, 3)]

    s1 = space.sampling(10)
    space2 = Space(settings)
    s2 = space2.sampling(10, kind='lhsc')
    assert s1[:] != s2[:]

    try:
        space += (1, 2, 3)
        space += (1, 2, 3)
    except UnicityError:
        assert True
    else:
        assert False

    try:
        space += (1, 7, 3)
    except AlienPointError:
        assert True
    else:
        assert False

    try:
        space.sampling(17)
    except FullSpaceError:
        assert True
    else:
        assert False

if __name__ == '__main__':
    test_space()