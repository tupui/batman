import os
import pytest
import numpy.testing as npt
import numpy as np
from batman.input_output import formater


@pytest.fixture(scope="module", params=["1D", "2D", "3D"])
def dataset(request):
    shape = {
        '1D': (21,),
        '2D': (42, 3),
        '3D': (7, 8, 9),
    }
    return {'data': np.random.rand(*shape[request.param]),
            'labels': ['x{}'.format(i) for i in range(shape[request.param][-1])]}

@pytest.fixture(scope="module", params=["json", "csv", "npz"])
def fmt_documented(request):
    return request.param

@pytest.fixture(scope="module", params=["npy"])
def fmt_undocumented(request):
    return request.param

def test_documented_format(tmp, fmt_documented, dataset):
    fmt = formater(fmt_documented)
    indata = dataset['data']
    labels = dataset['labels']

    # write
    filepath = os.path.join(tmp, 'testio.unknown')
    fmt.write(filepath, indata, labels)
    indata = indata.reshape(-1, indata.shape[-1])

    # read
    outdata = fmt.read(filepath, labels)
    npt.assert_array_equal(outdata, indata)

    # read out-of-order
    outdata = fmt.read(filepath, labels[::-1])
    npt.assert_array_equal(outdata, indata[:, ::-1])

    # read subset
    ncol = len(labels) // 2
    start = len(labels) // 4
    outdata = fmt.read(filepath, labels[start:start+ncol])
    npt.assert_array_equal(outdata, indata[:, start:start+ncol])


def test_undocumented_format(tmp, fmt_undocumented, dataset):
    fmt = formater(fmt_undocumented)
    indata = dataset['data']
    labels = dataset['labels']

    # write
    filepath = os.path.join(tmp, 'testio.unknown')
    fmt.write(filepath, indata, labels)
    indata = indata.reshape(-1, indata.shape[-1])

    # read
    outdata = fmt.read(filepath, labels)
    npt.assert_array_equal(outdata, indata)

    # read out-of-order
    outdata = fmt.read(filepath, labels[::-1])
    npt.assert_array_equal(outdata, indata)
