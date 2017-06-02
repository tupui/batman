# coding: utf8
import pytest
from batman.functions import MascaretApi
from io import StringIO


config_user = {
    "init_cst": {
        "Q_cst": 0.0,
        "Z_cst": 0.0
    },
    "Q_BC": {
        "idx": 0,
        "value": 2345.0
    },
    "misc": {
        "info_bc": true,
        "index_outstate": 25
    }
}


def test_Single_Run():
    f = MascaretApi('config_canal.json', StringIO(config_user))
    assert f() == pytest.approx(16.12553589, 0.0001)

    ks = {"Ks": {"zone": True,"value": 25.0,"ind_zone": 0}}
    config_user.update(ks)
    f = MascaretApi('config_canal.json', StringIO(config_user))
    assert f() == pytest.approx(16.82152109, 0.0001)
