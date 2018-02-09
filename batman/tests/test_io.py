# coding: utf8
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
    sizes = {
        '1D': [5, 6, 7, 3],
        '2D': [1, 2],
        '3D': [9] * 8,
    }
    return {'data': np.random.rand(*shape[request.param]),
            'labels': ['x{}'.format(i) for i in range(len(sizes[request.param]))],
            'sizes': sizes[request.param]}


@pytest.mark.parametrize('fmt_name', ['json', 'csv', 'npz'])
def test_documented_format(tmp, fmt_name, dataset):
    fmt = formater(fmt_name)
    indata = dataset['data']
    labels = dataset['labels']
    sizes = dataset['sizes']
    offsets = np.append(0, np.cumsum(sizes)[:-1])

    # write
    filepath = os.path.join(tmp, 'testio.unknown')
    fmt.write(filepath, indata, labels, sizes)
    indata = np.atleast_3d(indata)
    indata = indata.reshape(indata.shape[0], -1)

    # read
    outdata = fmt.read(filepath, labels)
    npt.assert_array_equal(outdata, indata)

    # read out-of-order
    order = sum([list(range(ofs, ofs+size)) for ofs, size
                 in zip(offsets[::-1], sizes[::-1])], [])
    outdata = fmt.read(filepath, labels[::-1])
    npt.assert_array_equal(outdata, indata[:, order])

    # read subset
    ncol = len(labels) // 2
    start = len(labels) // 4
    outdata = fmt.read(filepath, labels[start:start+ncol])
    order = sum([list(range(ofs, ofs+size)) for ofs, size
                 in zip(offsets[start:start+ncol], sizes[start:start+ncol])], [])
    npt.assert_array_equal(outdata, indata[:, order])


@pytest.mark.parametrize('fmt_name', ['npy'])
def test_undocumented_format(tmp, fmt_name, dataset):
    fmt = formater(fmt_name)
    indata = dataset['data']
    labels = dataset['labels']
    sizes = dataset['sizes']
    offsets = np.append(0, np.cumsum(sizes)[:-1])

    # write
    filepath = os.path.join(tmp, 'testio.unknown')
    fmt.write(filepath, indata, labels, sizes)
    indata = np.atleast_3d(indata)
    indata = indata.reshape(indata.shape[0], -1)

    # read
    outdata = fmt.read(filepath, labels)
    npt.assert_array_equal(outdata, indata)

    # read out-of-order
    outdata = fmt.read(filepath, labels[::-1])
    npt.assert_array_equal(outdata, indata)
