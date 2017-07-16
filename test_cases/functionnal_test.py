# coding: utf8

import pytest
import mock
import batman.ui
import batman.misc
import os
import shutil
import sys
from batman.tests.conftest import tmp

path = os.path.dirname(os.path.realpath(__file__))
schema = os.path.join(path, '../batman/misc/schema.json')

def check_output(tmp):
    if not os.path.isfile(os.path.join(tmp, 'surrogate/DOE.pdf')):
        assert False
    if not os.path.isfile(os.path.join(tmp, 'surrogate/surrogate.dat')):
        assert False
    if not os.path.isfile(os.path.join(tmp, 'surrogate/space.dat')):
        assert False
    if not os.path.isfile(os.path.join(tmp, 'surrogate/data.dat')):
        assert False


def init_case(tmp, case, output=True, force=False):
    os.chdir(os.path.join(path, case))
    sys.argv = ['batman', 'settings.json', '-o', tmp]
    run = True

    if force or (os.listdir(tmp) == []):
        shutil.rmtree(tmp)
    else:
        run = False

    if run:
        batman.ui.main()
        check_output(tmp)
    if not output:
        try:
            shutil.rmtree(os.path.join(tmp, 'surrogate/pod'))
        except:
            pass


# Use Michalewicz: 2D -> 1D
def test_init(tmp, case='Michalewicz'):
    init_case(tmp, case, force=True)
    check_output(tmp)


def test_no_pod(tmp, case='Michalewicz'):
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-n', '-o', tmp]
    batman.ui.main()
    check_output(tmp)
    if not os.path.isfile(os.path.join(tmp, 'surrogate/pod/pod.npz')):
        assert False


def test_no_model_pred(tmp, case='Michalewicz'):
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-ns', '-o', tmp]
    batman.ui.main()
    check_output(tmp)
    if not os.path.isdir(os.path.join(tmp, 'predictions')):
        assert False


def test_quality(tmp, case='Michalewicz'):
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-nq', '-o', tmp]
    batman.ui.main()
    check_output(tmp)


def test_uq(tmp, case='Michalewicz'):
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-nu', '-o', tmp]
    batman.ui.main()
    check_output(tmp)
    if not os.path.isdir(os.path.join(tmp, 'uq')):
        assert False


def test_checks(tmp, case='Michalewicz'):
    """Check answers to questions if there is an output folder."""
    init_case(tmp, case)

    # Restart from snapshots
    with mock.patch('builtins.input', side_effect=['', '']):
        batman.ui.main()

    check_output(tmp)

    # Remove files and restart
    with mock.patch('builtins.input', side_effect=['yes', 'yes']):
        batman.ui.main()

    check_output(tmp)

    # Exit without doing anything
    with mock.patch('builtins.input', side_effect=['no', 'no']):
        with pytest.raises(SystemExit):
            batman.ui.main()


def test_restart_pod(tmp, case='Michalewicz'):
    """Test all restart options."""
    # Restart POD from existing one and continue with resample
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-r', '-o', tmp]
    options = batman.ui.parse_options()
    settings = batman.misc.import_config(options.settings, schema)
    settings['space']['resampling']['resamp_size'] = 1
    batman.ui.run(settings, options)
    check_output(tmp)
    if not os.path.isdir(os.path.join(tmp, 'snapshots/4')):
        assert False

    init_case(tmp, case, force=True)
    # Restart from snapshots and read a template directory
    settings['snapshot']['io']['template_directory'] = os.path.join(tmp, 'snapshots/0/batman-data')
    batman.ui.run(settings, options)
    check_output(tmp)

    init_case(tmp, case, force=True)
    # Restart from 4 and add 2 points continuing the DOE sequence
    settings['space']['resampling']['resamp_size'] = 0
    settings['space']['sampling']['init_size'] = 6
    batman.ui.run(settings, options)
    check_output(tmp)
    if not os.path.isdir(os.path.join(tmp, 'snapshots/5')):
        assert False


def test_resampling(tmp, case='Michalewicz'):
    """Assess all resampling methods."""
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-o', tmp]
    options = batman.ui.parse_options()
    settings = batman.misc.import_config(options.settings, schema)
    settings['space']['sampling']['init_size'] = 10
    settings['space']['resampling']['resamp_size'] = 2

    for method in ['loo_sigma', 'extrema']:
        shutil.rmtree(tmp)
        settings['space']['resampling']['method'] = method
        if method == 'extrema':
            settings['space']['resampling']['resamp_size'] = 4
        batman.ui.run(settings, options)
        check_output(tmp)
        if not os.path.isdir(os.path.join(tmp, 'snapshots/11')):
            assert False

# Ishigami: 3D -> 1D
# Oakley & O'Hagan: 1D -> 1D
# Channel_Flow: 2D -> 400D
@pytest.mark.parametrize('name', [
    ('G_Function'),
    ('Basic_function'),
    ('Channel_Flow'),
])
def test_cases(tmp, name):
    test_init(tmp, case=name)
    test_quality(tmp, case=name)
    test_uq(tmp, case=name)
    test_restart_pod(tmp, case=name)


def test_simple_settings(tmp):
    init_case(tmp, 'Ishigami', output=False)
    sys.argv = ['batman', 'settings.json', '-o', tmp]
    options = batman.ui.parse_options()
    settings = batman.misc.import_config(options.settings, schema)
    settings['space'].pop('resampling')
    settings.pop('pod')
    settings.pop('surrogate')
    settings.pop('uq')
    shutil.rmtree(tmp)
    batman.ui.run(settings, options)
