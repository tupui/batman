import os
import pytest
from concurrent import futures
import copy
import numpy as np
import numpy.testing as npt
from paramiko.ssh_exception import NoValidConnectionsError
from batman.tasks.sample_cache import SampleCache
from batman.tasks import (ProviderFunction, ProviderFile, ProviderJob)
from batman.input_output import formater


@pytest.fixture(scope='module')
def sample_spec():
    return {
        'plabels': ['X1', 'X2'],
        'psizes': [1, 1],
        'flabels': ['F1', 'F2', 'F3'],
        'fsizes': [1, 1, 2],
        'space_fname': 'sample-space.json',
        'space_format': 'json',
        'data_fname': 'sample-data.csv',
        'data_format': 'csv',
    }


def test_samplecache(tmp, sample_spec):
    space_fmt = formater(sample_spec['space_format'])
    data_fmt = formater(sample_spec['data_format'])
    savedir = os.path.join(tmp, 'snapshots')
    datadir = os.path.join(os.path.dirname(__file__), 'data', 'snapshots')

    cache = SampleCache(savedir=savedir, **sample_spec)

    # test init --> is empty with proper labels
    assert len(cache) == 0

    # test discover --> load every existing snapshots
    cache.discover(os.path.join(datadir, '*'))
    assert len(cache) == 9
    space_file = sample_spec['space_fname']
    plabels = sample_spec['plabels']
    result_space = np.concatenate([
        space_fmt.read(os.path.join(datadir, '1', space_file), plabels),
        space_fmt.read(os.path.join(datadir, '3', space_file), plabels),
        space_fmt.read(os.path.join(datadir, '5', space_file), plabels),
    ])
    data_file = sample_spec['data_fname']
    flabels = sample_spec['flabels']
    result_data = np.concatenate([
        data_fmt.read(os.path.join(datadir, '1', data_file), flabels),
        data_fmt.read(os.path.join(datadir, '3', data_file), flabels),
        data_fmt.read(os.path.join(datadir, '5', data_file), flabels),
    ])
    npt.assert_array_equal(result_space, cache.space)
    npt.assert_array_equal(result_data, cache.data)

    # test save --> write to file (and reload)
    cache.save()
    assert os.path.isfile(os.path.join(savedir, space_file))
    assert os.path.isfile(os.path.join(savedir, data_file))
    result_space = space_fmt.read(os.path.join(savedir, space_file), plabels)
    result_data = data_fmt.read(os.path.join(savedir, data_file), flabels)
    npt.assert_array_equal(cache.space, result_space)
    npt.assert_array_equal(cache.data, result_data)

    cache.save(tmp)
    assert os.path.isfile(os.path.join(tmp, space_file))
    assert os.path.isfile(os.path.join(tmp, data_file))

    # test locate --> return proper location for existing and new points
    points = cache.space[:4] * np.reshape([1, -1, -1, 1], (-1, 1))
    index = cache.locate(points)
    npt.assert_array_equal([0, 9, 10, 3], index)


def test_provider_function(tmp, sample_spec):
    space_fmt = formater(sample_spec['space_format'])
    data_fmt = formater(sample_spec['data_format'])
    space_file = sample_spec['space_fname']
    plabels = sample_spec['plabels']
    data_file = sample_spec['data_fname']
    flabels = sample_spec['flabels']
    datadir = os.path.join(os.path.dirname(__file__), 'data', 'snapshots')

    provider = ProviderFunction(module='batman.tests.plugins', function='f_snapshot',
                                discover_pattern=os.path.join(datadir, '*'),
                                **sample_spec)

    # test return existing
    points = space_fmt.read(os.path.join(datadir, '3', space_file), plabels)
    data = data_fmt.read(os.path.join(datadir, '3', data_file), flabels)
    sample = provider.require_data(points)
    npt.assert_array_equal(points, sample.space)
    npt.assert_array_equal(data, sample.data)

    # test return new
    points *= -1
    data = np.tile([42, 87, 74, 74], (len(points), 1))
    sample = provider.require_data(points)
    npt.assert_array_equal(points, sample.space)
    npt.assert_array_equal(data, sample.data)


def test_provider_file(sample_spec):
    space_fmt = formater(sample_spec['space_format'])
    data_fmt = formater(sample_spec['data_format'])
    space_file = sample_spec['space_fname']
    plabels = sample_spec['plabels']
    data_file = sample_spec['data_fname']
    flabels = sample_spec['flabels']
    datadir = os.path.join(os.path.dirname(__file__), 'data', 'snapshots')

    filedir = os.path.join(datadir, 'sample')
    files = [(os.path.join(filedir, 'toto.json'), os.path.join(filedir, 'tata.csv')),
             (os.path.join(filedir, 'bibi.json'), os.path.join(filedir, 'bubu.csv'))]
    provider = ProviderFile(file_pairs=files,
                            discover_pattern=os.path.join(datadir, '?'),
                            **sample_spec)

    # test 9 from discover_patter (1, 3, 5) and 3 from file_pairs
    assert len(provider._cache.space) == 12
    assert len(provider._cache.data) == 12

    # test return existing
    points = np.vstack((space_fmt.read(os.path.join(datadir, '3', space_file), plabels),
                        space_fmt.read(files[0][0], plabels)))
    data = np.vstack((data_fmt.read(os.path.join(datadir, '3', data_file), flabels),
                      data_fmt.read(files[0][1], flabels)))
    sample = provider.require_data(points)
    npt.assert_array_equal(points, sample.space)
    npt.assert_array_equal(data, sample.data)

    # test raise new
    with pytest.raises(ValueError):
        sample = provider.require_data(-points)


def test_provider_job(sample_spec):
    space_fmt = formater(sample_spec['space_format'])
    data_fmt = formater(sample_spec['data_format'])
    space_file = sample_spec['space_fname']
    plabels = sample_spec['plabels']
    data_file = sample_spec['data_fname']
    flabels = sample_spec['flabels']
    datadir = os.path.join(os.path.dirname(__file__), 'data', 'snapshots')

    pool = futures.ThreadPoolExecutor(max_workers=2)
    provider = ProviderJob(command='python script.py',
                           context_directory=os.path.join(datadir, 'job'),
                           coupling={'coupling_directory': 'coupling-dir'},
                           discover_pattern=os.path.join(datadir, '*'),
                           pool=pool,
                           **sample_spec)

    # test loading existing
    assert len(provider._cache) == 9

    # test return existing
    points = space_fmt.read(os.path.join(datadir, '3', space_file), plabels)
    data = data_fmt.read(os.path.join(datadir, '3', data_file), flabels)
    sample = provider.require_data(points)
    npt.assert_array_equal(points, sample.space)
    npt.assert_array_equal(data, sample.data)

    # test return new
    points *= -1
    data = np.tile([42, 87, 74, 74], (len(points), 1))
    sample = provider.require_data(points)
    npt.assert_array_equal(points, sample.space)


@pytest.mark.xfail(raises=NoValidConnectionsError, reason='Cannot connect to localhost')
def test_remote_job(tmp, sample_spec):
    space_fmt = formater(sample_spec['space_format'])
    data_fmt = formater(sample_spec['data_format'])
    space_file = sample_spec['space_fname']
    plabels = sample_spec['plabels']
    data_file = sample_spec['data_fname']
    flabels = sample_spec['flabels']
    datadir = os.path.join(os.path.dirname(__file__), 'data', 'snapshots')

    host = [
        {
            'hostname': 'localhost',
            'remote_root': 'TTA'
        },
        {
            'hostname': 'localhost',
            'remote_root': 'TTB'
        }
    ]
    pool = futures.ThreadPoolExecutor(max_workers=3)

    sample_spec = copy.deepcopy(sample_spec)
    sample_spec.update(dict(command='bash remote_script.sh',
                            context_directory=os.path.join(datadir, 'job'),
                            coupling={'coupling_directory': 'coupling-dir'},
                            discover_pattern=os.path.join(datadir, '*'),
                            pool=pool, hosts=host, save_dir=tmp))

    provider = ProviderJob(**sample_spec)

    # test return existing
    points = space_fmt.read(os.path.join(datadir, '3', space_file), plabels)
    data = data_fmt.read(os.path.join(datadir, '3', data_file), flabels)
    sample = provider.require_data(points)
    npt.assert_array_equal(points, sample.space)
    npt.assert_array_equal(data, sample.data)

    # test return new
    points *= -1
    data = np.tile([42, 87, 74, 74], (len(points), 1))
    sample = provider.require_data(points)
    npt.assert_array_equal(points, sample.space)
    npt.assert_array_equal(data, sample.data)
    npt.assert_array_equal(data, sample.data)

    # test with weights
    sample_spec['hosts'][0]['weight'] = 0.3
    sample_spec['hosts'][1]['weight'] = 0.8
    sample_spec['hosts'][0]['remote_root'] = 'TTC'
    sample_spec['hosts'][1]['remote_root'] = 'TTD'
    sample_spec['save_dir'] = os.path.join(tmp, 'new')
    provider = ProviderJob(**sample_spec)

    sample = provider.require_data(points)
    npt.assert_array_equal(points, sample.space)
    npt.assert_array_equal(data, sample.data)
    npt.assert_array_equal(data, sample.data)
