import pytest
from batman.functions import (Data, el_nino, tahiti, mascaret, marthe)
import numpy.testing as npt
import numpy as np


def test_data():
    # Only data
    data = np.array([1, 2, 3, 4]).reshape(-1, 1)
    dataset = Data(data=data, desc='toto')

    assert len(dataset) == len(data)
    assert dataset.shape == np.array(data).shape
    assert dataset[2] == data[2]
    npt.assert_almost_equal(dataset.data, data)
    npt.assert_almost_equal([i for i in dataset], data)

    # Using sample
    sample = np.array([3, 6, 8, 9]).reshape(-1, 1)

    with pytest.raises(SystemError):
        dataset = Data(data=data, desc='toto', sample=sample[:-1])

    dataset = Data(data=data, desc='toto', sample=sample)

    assert dataset[2] == (8, 3)
    npt.assert_almost_equal([i for i in dataset], list(zip(sample, data)))

    # with array_like data
    data = np.array([[3, 4, 8], [1, 2, 9], [5, 7, 10], [12, 3, 0]])
    dataset = Data(data=data, desc='toto', sample=sample)

    assert dataset[2][0] == 8
    npt.assert_almost_equal(dataset[2][1], [5, 7, 10])
    npt.assert_almost_equal([i[1] for i in dataset], data)

    # Using names
    plabels = ['Temp']
    flabels = ['Min', 'Mean', 'Max']
    dataset = Data(data=data, desc='toto', sample=sample,
                   plabels=plabels, flabels=flabels)

    npt.assert_almost_equal(dataset.data['Min'], [3, 1, 5, 12])
    npt.assert_almost_equal(dataset.sample['Temp'], [3, 6, 8, 9])

    # Back to regular arrays
    dataset.toarray()
    npt.assert_almost_equal(dataset[2][1], [5, 7, 10])
    npt.assert_almost_equal([i[1] for i in dataset], data)

    with pytest.raises(IndexError):
        npt.assert_almost_equal(dataset.data['Min'], [3, 1, 5, 12])


def test_el_nino():
    data = el_nino()
    assert len(data) == 58
    assert data.shape == ((58, 1), (58, 12))
    assert data.data[6][4] == 23.24
    assert tuple(data.sample[6]) == (1956,)


def test_tahiti():
    data = tahiti()
    assert len(data) == 66
    assert data.shape == ((66, 1), (66, 12))
    assert data.data[6][4] == 12.3
    assert tuple(data.sample[6]) == (1957,)


def test_mascaret():
    data = mascaret()
    assert len(data) == 100000
    assert data.shape == ((100000, 2), (100000, 14))
    assert data.data[6][4] == 23.058688963373736
    npt.assert_almost_equal(data.sample[6].tolist(), [55.6690593, 3550.8123906])
    assert data.data['21925'][4] == 25.384495603739079


def test_marthe():
    data = marthe()
    assert len(data) == 300
    assert data.shape == ((300, 20), (300, 10))
    assert data.data[6][4] == 1.2749
    out = [9.0116, 13.849, 12.705, 5.9253, 6.6665, 9.8553, 10.598, 0.68069,
           0.7728, 1.6404, 0.061930538, 0.01932541, 0.057965174, 10.09, 16.667,
           45.725, 0.33342, 6.42E-05, 0.005401919, 0.069934838]
    npt.assert_almost_equal(data.sample[6].tolist(), out)
