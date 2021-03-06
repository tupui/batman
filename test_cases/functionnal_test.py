"""Functionnal testing suite."""
import os
import shutil
import sys
import json
import re
import copy
import pytest
import mock
import batman.ui
import batman.misc

PATH = os.path.dirname(os.path.realpath(__file__))
SCHEMA = os.path.join(PATH, '../batman/misc/schema.json')

USER_INPUT = 'builtins.input'


def check_output(tmp):
    assert os.path.isfile(os.path.join(tmp, 'space/DOE.pdf'))
    assert os.path.isfile(os.path.join(tmp, 'space/space.dat'))
    assert os.path.isfile(os.path.join(tmp, 'surrogate/surrogate.dat'))
    assert os.path.isfile(os.path.join(tmp, 'surrogate/data.dat'))


def init_case(tmp, case, output=True, force=False):
    os.chdir(os.path.join(PATH, case))
    run = True

    if force or (os.listdir(tmp) == []):
        shutil.rmtree(tmp)
    else:
        run = False

    if run:
        sys.argv = ['batman', 'settings.json', '-o', tmp]
        batman.ui.main()
    if not output:
        try:
            shutil.rmtree(os.path.join(tmp, 'surrogate/pod'))
        except OSError:
            pass


# Use Michalewicz: 2D -> 1D
def test_empty_output(tmp, case='Michalewicz'):
    os.mkdir(os.path.join(tmp, 'snapshots'))
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-nv', '-o', tmp]
    with pytest.raises(SystemExit):
        batman.ui.main()


def test_init(tmp, case='Michalewicz'):
    init_case(tmp, case, force=True)
    if case != 'Channel_Flow':
        check_output(tmp)


def test_no_model(tmp, case='Michalewicz'):
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-n', '-o', tmp]
    batman.ui.main()
    check_output(tmp)
    assert os.path.isfile(os.path.join(tmp, 'surrogate/pod/pod.npz'))


def test_no_model_pred(tmp, case='Michalewicz'):
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-ns', '-o', tmp]
    batman.ui.main()
    check_output(tmp)
    assert os.path.isdir(os.path.join(tmp, 'predictions'))


def test_quality(tmp, case='Michalewicz'):
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-nq', '-o', tmp]
    batman.ui.main()


def test_uq(tmp, case='Michalewicz'):
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-nu', '-o', tmp]
    batman.ui.main()
    assert os.path.isdir(os.path.join(tmp, 'uq'))


def test_checks(tmp, case='Michalewicz'):
    """Check answers to questions if there is an output folder."""
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-o', tmp]

    # Restart from snapshots, first enter something incorrect
    with mock.patch(USER_INPUT, side_effect=['nope', '', '']):
        batman.ui.main()

    check_output(tmp)

    # Remove files and restart
    with mock.patch(USER_INPUT, side_effect=['yes', 'yes']):
        batman.ui.main()

    check_output(tmp)

    # Exit without doing anything
    with mock.patch(USER_INPUT, side_effect=['no', 'no']):
        with pytest.raises(SystemExit):
            batman.ui.main()

    # Exit because no snapshot folder
    shutil.rmtree(os.path.join(tmp, 'snapshots'))
    with mock.patch(USER_INPUT, side_effect=['no', 'no']):
        with pytest.raises(SystemExit):
            batman.ui.main()


def test_restart_pod(tmp, case='Michalewicz'):
    """Test all restart options."""
    # Restart POD from existing one and continue with resample
    init_case(tmp, case, force=True)
    sys.argv = ['batman', 'settings.json', '-r', '-o', tmp]
    options = batman.ui.parse_options()
    settings = batman.misc.import_config(options.settings, SCHEMA)
    settings['space']['resampling']['resamp_size'] = 1
    batman.ui.run(settings, options)
    assert os.path.isdir(os.path.join(tmp, 'snapshots/4'))

    # Restart from 5 and add 2 points continuing the DOE sequence
    # DoE from 4 to 6 so max points is 7
    ns = 12 if case == 'G_Function' else 6
    try:
        settings['space']['sampling']['init_size'] = ns
    except TypeError:  # Case with list instead of dict
        settings['space']['sampling'] = {'init_size': ns, 'method': 'halton'}
    batman.ui.run(settings, options)
    assert os.path.isdir(os.path.join(tmp, 'snapshots/6'))


def test_resampling(tmp, case='Michalewicz'):
    """Assess all resampling methods."""
    init_case(tmp, case)
    sys.argv = ['batman', 'settings.json', '-o', tmp]
    options = batman.ui.parse_options()
    settings = batman.misc.import_config(options.settings, SCHEMA)
    settings['space']['resampling']['resamp_size'] = 2

    for method in ['loo_sigma', 'hybrid']:
        shutil.rmtree(tmp)
        settings['space']['resampling']['method'] = method
        if method == 'hybrid':
            settings['space']['resampling']['resamp_size'] = 4
        batman.ui.run(settings, options)
        check_output(tmp)
        assert os.path.isdir(os.path.join(tmp, 'snapshots/5'))


# Ishigami: 3D -> 1D
# Oakley & O'Hagan: 1D -> 1D
# Channel_Flow: 2D -> nD
@pytest.mark.parametrize('name', [
    ('G_Function'),
    ('Basic_function'),
    ('Channel_Flow'),
    ('Multifidelity'),
])
def test_cases(tmp, name):
    test_init(tmp, case=name)
    if name != 'Multifidelity':
        test_quality(tmp, case=name)
    test_uq(tmp, case=name)
    if name not in ['Channel_Flow', 'Multifidelity']:
        test_restart_pod(tmp, case=name)


def test_simple_settings(tmp, case='Ishigami'):
    os.chdir(os.path.join(PATH, case))
    sys.argv = ['batman', 'settings.json', '-o', tmp]
    options = batman.ui.parse_options()
    settings = batman.misc.import_config(options.settings, SCHEMA)
    settings['space'].pop('resampling')
    settings.pop('pod')
    settings.pop('surrogate')
    settings.pop('uq')
    tmp_settings_path = os.path.join(tmp, 'simple_settings.json')
    with open(tmp_settings_path, 'w') as f:
        json.dump(settings, f, indent=4)
    simple_settings = batman.misc.import_config(tmp_settings_path, SCHEMA)
    shutil.rmtree(tmp)
    batman.ui.run(simple_settings, options)


def test_only_surrogate(tmp, case='Michalewicz'):
    os.chdir(os.path.join(PATH, case))
    sys.argv = ['batman', 'settings.json', '-o', tmp]
    options = batman.ui.parse_options()
    settings = batman.misc.import_config(options.settings, SCHEMA)
    settings['space'].pop('resampling')
    settings.pop('pod')
    settings.pop('uq')
    tmp_settings_path = os.path.join(tmp, 'only_surrogate_settings.json')
    with open(tmp_settings_path, 'w') as f:
        json.dump(settings, f, indent=4)
    only_surrogate_settings = batman.misc.import_config(tmp_settings_path, SCHEMA)
    shutil.rmtree(tmp)
    clean_settings = copy.deepcopy(only_surrogate_settings)
    batman.ui.run(only_surrogate_settings, options)

    # Restart from snapshots
    with mock.patch(USER_INPUT, side_effect=['', '']):
        batman.ui.run(clean_settings, options)

    check_output(tmp)


def test_only_surrogate_settings(tmp, case='Basic_function'):
    os.chdir(os.path.join(PATH, case))
    sys.argv = ['batman', 'settings.json', '-o', tmp]
    options = batman.ui.parse_options()
    settings = batman.misc.import_config(options.settings, SCHEMA)
    settings['space'].pop('resampling')
    settings.pop('uq')
    # Kriging settings
    settings['surrogate'].update({
        'kernel': "ConstantKernel() + "
                  "Matern(length_scale=0.5, nu=0.5)",
        'noise': 1,
        'global_optimizer': False})
    shutil.rmtree(tmp)
    batman.ui.run(settings, options)

    # SklearnRegressor
    settings['surrogate'] = {"method": "RandomForestRegressor()"}
    shutil.rmtree(tmp)
    batman.ui.run(settings, options)


def test_uq_no_surrogate(tmp, case='Ishigami'):
    os.chdir(os.path.join(PATH, case))
    sys.argv = ['batman', 'settings.json', '-u', '-o', tmp]
    options = batman.ui.parse_options()
    settings = batman.misc.import_config(options.settings, SCHEMA)
    settings['space']['sampling']['method'] = 'saltelli'
    settings['space']['sampling']['init_size'] = 80
    settings['space'].pop('resampling')
    settings.pop('pod')
    settings.pop('surrogate')
    tmp_settings_path = os.path.join(tmp, 'uq_no_surrogate_settings.json')
    with open(tmp_settings_path, 'w') as f:
        json.dump(settings, f, indent=4)
    uq_no_surrogate_settings = batman.misc.import_config(tmp_settings_path, SCHEMA)
    shutil.rmtree(tmp)
    batman.ui.run(uq_no_surrogate_settings, options)


def test_doe_as_list(tmp, case='Ishigami'):
    os.chdir(os.path.join(PATH, case))
    sys.argv = ['batman', 'settings.json', '-o', tmp]
    options = batman.ui.parse_options()
    settings = batman.misc.import_config(options.settings, SCHEMA)
    settings['space'].pop('resampling')
    settings.pop('pod')
    settings.pop('surrogate')
    settings['space']['sampling'] = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    tmp_settings_path = os.path.join(tmp, 'doe_as_list_settings.json')
    with open(tmp_settings_path, 'w') as f:
        json.dump(settings, f, indent=4)
    doe_as_list_settings = batman.misc.import_config(tmp_settings_path, SCHEMA)
    shutil.rmtree(tmp)
    batman.ui.run(doe_as_list_settings, options)


def test_wrong_settings(tmp, case='Ishigami'):
    os.chdir(os.path.join(PATH, case))

    # First check some correct settings
    sys.argv = ['batman', 'settings.json', '-c', '-o', tmp]
    with pytest.raises(SystemExit):
        batman.ui.main()

    sys.argv = ['batman', 'settings.json', '-o', tmp]
    options = batman.ui.parse_options()
    settings = batman.misc.import_config(options.settings, SCHEMA)

    # Invalid settings
    settings['space']['sampling'] = {'init_size': 150, 'method': 'wrong'}

    wrong_path = os.path.join(tmp, 'wrong_settings.json')
    with open(wrong_path, 'w') as f:
        json.dump(settings, f, indent=4)

    with pytest.raises(SyntaxError):
        batman.misc.import_config(wrong_path, SCHEMA)

    # Invalid JSON file
    with open('settings.json', 'rb') as ws:
        file = ws.read().decode('utf8')
        exp = re.search('(\"space\")(:)', file, re.MULTILINE)
        file = file.replace(exp.group(2), ',')

    with open(wrong_path, 'wb') as ws:
        ws.write(file.encode('utf8'))

    with pytest.raises(SyntaxError):
        batman.misc.import_config(wrong_path, SCHEMA)
