from batman.functions import (el_nino, tahiti, mascaret, marthe)
import numpy.testing as npt


def test_el_nino():
    data = el_nino()
    assert len(data) == 58
    assert data.shape == (58, 13)
    assert data.data[6][4] == 23.24
    assert tuple(data.space[6]) == (1956,)


def test_tahiti():
    data = tahiti()
    assert len(data) == 66
    assert data.shape == (66, 13)
    assert data.data[6][4] == 12.3
    assert tuple(data.space[6]) == (1957,)


def test_mascaret():
    data = mascaret()
    assert len(data) == 100000
    assert data.shape == (100000, 16)
    assert data.data[6][4] == 23.058688963373736
    npt.assert_almost_equal(data.space[6].tolist(), [55.6690593, 3550.8123906])
    assert data.dataframe['data']['21925'].values[4] == 25.384495603739079

    data = mascaret(multizone=True)
    assert len(data) == 5000
    assert data.shape == (5000, 467)
    assert data.data[5][4] == 27.907469930354466
    npt.assert_almost_equal(data.space[5].tolist(), [34.5856891, 36.695892, 33.8615593, 4592.2619857])
    print(data.dataframe['data']['13750.0'])
    assert data.dataframe['data']['13750.0'].values[5][0] == 27.86339870997002


def test_marthe():
    data = marthe()
    assert len(data) == 300
    assert data.shape == (300, 30)
    assert data.data[6][4] == 1.2749
    out = [9.0116, 13.849, 12.705, 5.9253, 6.6665, 9.8553, 10.598, 0.68069,
           0.7728, 1.6404, 0.061930538, 0.01932541, 0.057965174, 10.09, 16.667,
           45.725, 0.33342, 6.42E-05, 0.005401919, 0.069934838]
    npt.assert_almost_equal(data.space[6].tolist(), out)
