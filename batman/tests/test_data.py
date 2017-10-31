import pytest
from batman.functions import Data
import numpy.testing as npt
import numpy as np


def test_data():
    data = [1, 2, 3, 4]
    dataset = Data(data=[1, 2, 3, 4], desc='toto')

    assert len(dataset) == len(data)
    assert dataset.shape == np.array(data).shape
    assert dataset[2] == data[2]
    npt.assert_almost_equal(dataset.data, data)
    npt.assert_almost_equal([i for i in dataset], data)

    data = [1, 2, 3, 4]
    sample = [3, 6, 8, 9]

    with pytest.raises(SystemError):
        dataset = Data(data=[1, 2, 3, 4], desc='toto', sample=sample[:-1])

    dataset = Data(data=[1, 2, 3, 4], desc='toto', sample=sample)

    import pdb; pdb.set_trace()  # breakpoint 89b0c7c8 //

