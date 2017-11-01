import pytest
from batman.functions import Data
import numpy.testing as npt
import numpy as np


def test_data():
    # Only data
    data = [1, 2, 3, 4]
    dataset = Data(data=data, desc='toto')

    assert len(dataset) == len(data)
    assert dataset.shape == np.array(data).shape
    assert dataset[2] == data[2]
    npt.assert_almost_equal(dataset.data, data)
    npt.assert_almost_equal([i for i in dataset], data)

    # Using sample
    sample = [3, 6, 8, 9]

    with pytest.raises(SystemError):
        dataset = Data(data=data, desc='toto', sample=sample[:-1])

    dataset = Data(data=data, desc='toto', sample=sample)

    assert dataset[2] == (8, 3)
    npt.assert_almost_equal([i for i in dataset], list(zip(sample, data)))

    # with array_like data
    data = [[3, 4, 8], [1, 2, 9], [5, 7, 10], [12, 3, 0]]
    dataset = Data(data=data, desc='toto', sample=sample)

    assert dataset[2][0] == 8
    npt.assert_almost_equal(dataset[2][1], [5, 7, 10])
    npt.assert_almost_equal([i[1] for i in dataset], data)
