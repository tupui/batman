from copy import deepcopy
import os
import pytest
import itertools
import numpy as np
import numpy.testing as npt
from batman.space import Sample


NSAMPLE = [0, 42]
SPACE_SPEC = [
    {'nfeature': 0, 'ncomponent': 0, 'none': ['labels', 'sizes']},
    {'nfeature': 2, 'ncomponent': 1, 'none': ['labels']},
    {'nfeature': 1, 'ncomponent': 3, 'none': ['sizes']},
    {'nfeature': 3, 'ncomponent': [2, 7], 'none': []},
]
DATA_SPEC = [
    {'nfeature': 0, 'ncomponent': 0, 'none': ['labels', 'sizes']},
    {'nfeature': 7, 'ncomponent': 1, 'none': []},
    {'nfeature': 2, 'ncomponent': 3, 'none': []},
    {'nfeature': 1, 'ncomponent': [2, 7], 'none': ['labels']},
]
PARAMS = list(itertools.product(NSAMPLE, SPACE_SPEC, DATA_SPEC))


def build_dataset(nsample, nfeature, ncomponent, template='X'):
    if nfeature == 0:
        return np.empty((0, 0)), [], []

    labels = ['{}{}'.format(template, i) for i in range(nfeature)]
    if isinstance(ncomponent, list):
        sizes = np.random.randint(*ncomponent, size=nfeature).tolist()
        shape = (nsample, sum(sizes))
    else:
        sizes = np.repeat(ncomponent, nfeature).tolist()
        shape = (nsample, nfeature, ncomponent)

    data = np.random.rand(*shape)
    return (data, labels, sizes)


@pytest.fixture(scope='module', params=PARAMS)
def sample_case(request):
    nsample, space_spec, data_spec = request.param
    args = {}

    # space
    space, plabels, psizes = build_dataset(nsample,
                                           space_spec['nfeature'],
                                           space_spec['ncomponent'],
                                           'p')
    args['space'] = space if space.size > 0 else None
    args['plabels'] = plabels if 'labels' not in space_spec['none'] else None
    args['psizes'] = psizes if 'sizes' not in space_spec['none'] else None

    # data
    data, flabels, fsizes = build_dataset(nsample,
                                          data_spec['nfeature'],
                                          data_spec['ncomponent'],
                                          'f')
    args['data'] = data if data.size > 0 else None
    args['flabels'] = flabels if 'labels' not in data_spec['none'] else None
    args['fsizes'] = fsizes if 'sizes' not in data_spec['none'] else None

    # case
    sample = Sample(**args)

    if args['space'] is None:
        if (args['plabels'] is None) and (args['psizes'] is None):
            plabels = []
            psizes = []
        elif args['psizes'] is None:
            psizes = [1] * len(plabels)
    space = space.reshape(nsample, sum(psizes))
    if args['data'] is None:
        if (args['flabels'] is None) and (args['fsizes'] is None):
            flabels = []
            fsizes = []
        elif args['fsizes'] is None:
            fsizes = [1] * len(flabels)
    data = data.reshape(nsample, sum(fsizes))
    values = np.append(space, data, axis=1)
    if values.size == 0:
        nsample = 0
        values = values.reshape(0, values.shape[1])
        space = space.reshape(0, space.shape[1])
        data = data.reshape(0, data.shape[1])

    expected = {
        'len': nsample,
        'shape': values.shape,
        'values': values,
        'space': space,
        'plabels': plabels,
        'psizes': psizes,
        'data': data,
        'flabels': flabels,
        'fsizes': fsizes,
    }

    return sample, expected


def test_api(tmp, sample_case):
    sample, expected = sample_case

    # properties
    assert len(sample) == expected['len']
    npt.assert_array_equal(sample.shape, expected['shape'])
    npt.assert_array_equal(sample.plabels, expected['plabels'])
    npt.assert_array_equal(sample.flabels, expected['flabels'])
    npt.assert_array_equal(sample.psizes, expected['psizes'])
    npt.assert_array_equal(sample.fsizes, expected['fsizes'])
    npt.assert_array_equal(sample.values, expected['values'])
    npt.assert_array_equal(sample.space, expected['space'])
    npt.assert_array_equal(sample.data, expected['data'])

    # I/Os
    sample_work = deepcopy(sample)
    sample_work.empty()

    fname_space = os.path.join(tmp, 'sample-space.json')
    fname_data = os.path.join(tmp, 'sample-data.json')
    sample.write(fname_space, fname_data)

    sample_work.read(fname_space, fname_data)
    npt.assert_array_equal(sample_work.values, sample.values)
    # test read append entries to existing samples
    sample_work.read(fname_space, fname_data)
    npt.assert_array_equal(sample_work.values[:len(sample)], sample.values)
    npt.assert_array_equal(sample_work.values[len(sample):], sample.values)


def test_container(sample_case):
    sample, _ = sample_case
    sample_work = deepcopy(sample)
    snapshots = np.random.rand(10, sample.shape[1])
    expected = np.append(sample.values, snapshots, axis=0)

    # __getitem___ and __iter__
    assert isinstance(sample[:], Sample)
    npt.assert_array_equal(sample[:].values, sample.values)
    for i, point in enumerate(sample):
        npt.assert_array_equal(point, sample.values[i])
        npt.assert_array_equal(point, sample[i])
        assert point in sample

    # append/pop 1 by 1
    for snap in snapshots:
        sample_work.append(snap)
    npt.assert_array_equal(sample_work.values, expected)
    for snap in snapshots[::-1, :]:
        point = sample_work.pop()
        assert point not in sample
        npt.assert_array_equal(point, snap)
    npt.assert_array_equal(sample_work.values, sample.values)

    # append all at once
    sample_work.append(snapshots)
    npt.assert_array_equal(sample_work.values, expected)
    # __setitem__
    sample_work[:] = 0
    npt.assert_array_equal(sample_work.values, np.zeros(expected.shape))
    sample_work[:len(sample)] = sample.values
    npt.assert_array_equal(sample_work[:len(sample)].values, sample.values)
    npt.assert_array_equal(sample_work[len(sample):].values, np.zeros(snapshots.shape))

    # deletion
    sample_work.empty()
    assert len(sample_work) == 0
    npt.assert_array_equal(sample_work.plabels, sample.plabels)
    npt.assert_array_equal(sample_work.flabels, sample.flabels)
    npt.assert_array_equal(sample_work.psizes, sample.psizes)
    npt.assert_array_equal(sample_work.fsizes, sample.fsizes)

    # append another sample
    sample_work.append(sample)
    npt.assert_array_equal(sample_work.values, sample.values)
    expected = sample.values.copy()
    # del 1 by 1
    while len(sample_work) > 0:
        mid = len(sample_work) // 2
        expected = np.delete(expected, mid, axis=0)
        del sample_work[mid]
        npt.assert_array_equal(sample_work.values, expected)

    # append a dataframe
    sample_work.append(sample.dataframe)
    npt.assert_array_equal(sample_work.values, sample.values)
    expected = sample.values.copy()
    # del by slices
    while len(sample_work) > 0:
        mid = (len(sample_work) + 1) // 2
        expected = np.delete(expected, slice(mid), axis=0)
        del sample_work[:mid]
        npt.assert_array_equal(sample_work.values, expected)

    # add operators
    sample_work += sample
    assert len(sample_work) == len(sample)
    if len(sample) > 0:
        sample_work += sample[0]
        assert len(sample_work) == len(sample) + 1
        npt.assert_array_equal(sample_work[0], sample_work[-1])
        sample_other = sample_work + sample
        assert len(sample_work) == len(sample) + 1
        assert len(sample_other) == 2 * len(sample) + 1
        del sample_other[0]
        assert len(sample_work) == len(sample) + 1
        assert len(sample_other) == 2 * len(sample)

    if sample.space.shape[1] == 0:
        with pytest.raises(ValueError):
            sample_space = Sample(space=sample.space, plabels=sample.plabels, psizes=sample.psizes)
    if sample.data.shape[1] == 0:
        with pytest.raises(ValueError):
            sample_data = Sample(data=sample.data, flabels=sample.flabels, fsizes=sample.fsizes)

    if (sample.data.shape[1] > 0) and (sample.space.shape[1] > 0):
        sample_space = Sample(space=sample.space, plabels=sample.plabels, psizes=sample.psizes)
        sample_data = Sample(data=sample.data, flabels=sample.flabels, fsizes=sample.fsizes)
        # append new features from sample
        sample_work = deepcopy(sample_space)
        sample_work.append(sample_data, axis=1)
        npt.assert_array_equal(sample_work.values, sample.values)
        # append new features from dataframe
        sample_work = deepcopy(sample_space)
        sample_work.append(sample_data.dataframe, axis=1)
        npt.assert_array_equal(sample_work.values, sample.values)
        # append new features from dataframe with a bad formated index
        sample_work = deepcopy(sample_space)
        with pytest.raises(AssertionError):
            sample_work.append(sample_data.dataframe['data'], axis=1)
        # append new features from numpy array
        sample_work = deepcopy(sample_space)
        with pytest.raises(ValueError):
            sample_work.append(sample_data.values, axis=1)


def test_sample_incomplete(tmp):
    args = {}

    # space
    space, plabels, psizes = build_dataset(10, 2, 1, 'p')
    args['space'] = space
    args['plabels'] = plabels
    args['psizes'] = psizes

    # data
    data, flabels, fsizes = build_dataset(7, 2, 1, 'f')
    args['data'] = data
    args['flabels'] = flabels
    args['fsizes'] = fsizes

    # case
    sample = Sample(**args)
    assert np.count_nonzero(~np.isnan(sample.data)) == 14
    assert sample.space.shape == (10, 2)

    fname_space = os.path.join(tmp, 'sample-space.json')
    fname_data = os.path.join(tmp, 'sample-data.json')
    sample.write(fname_space, fname_data)

    sample_work = deepcopy(sample)
    sample_work.empty()
    sample_work.read(fname_space, fname_data)

    assert sample_work.data.shape == (7, 2)
    assert sample_work.space.shape == (7, 2)
