# coding: utf8

import pytest
import mock
import jpod.ui
import jpod.misc
import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
schema = path + "/../jpod/misc/schema.json"

def check_output():
    if not os.path.isdir('output/pod'):
        assert False
    if not os.path.isfile('output/pod/model'):
        assert False
    if not os.path.isfile('output/pod/points.dat'):
        assert False
    if not os.path.isfile('output/pod/pod.npz'):
        assert False


def init_case(case, output=True, force=False):
    os.chdir(path + case)
    sys.argv = ['jpod', 'settings.json']
    run = True

    if force:
        os.system('rm -rf output')
    elif os.path.isdir('output'):
        run = False
    
    if run:
        jpod.ui.main()
        check_output()
    if not output:
        os.system('rm -rf output/pod')


# Use Michalewicz: 2D -> 1D
def test_init(case='/Michalewicz'):
    init_case(case, force=True)
    check_output()


def test_no_pod(case='/Michalewicz'):
    init_case(case)
    sys.argv = ['jpod', 'settings.json', '-n']
    jpod.ui.main()
    check_output()


def test_no_model_pred(case='/Michalewicz'):
    init_case(case)
    sys.argv = ['jpod', 'settings.json', '-ps']
    jpod.ui.main()
    check_output()
    if not os.path.isdir('output/predictions'):
        assert False


def test_quality(case='/Michalewicz'):
    init_case(case)
    sys.argv = ['jpod', 'settings.json', '-pq']
    jpod.ui.main()
    check_output()


def test_uq(case='/Michalewicz'):
    init_case(case)
    sys.argv = ['jpod', 'settings.json', '-pu']
    jpod.ui.main()
    check_output()
    if not os.path.isdir('output/uq'):
        assert False


def test_checks(case='/Michalewicz'):
    """Check answers to questions if there is an output folder."""
    init_case(case, output=False)

    # Restart from snapshots
    with mock.patch.object(jpod.misc, 'check_yes_no', lambda prompt, default: '\n'):
        jpod.ui.main()

    check_output()

    # Remove files and restart
    with mock.patch.object(jpod.misc, 'check_yes_no', lambda prompt, default: 'y'):
        jpod.ui.main()

    check_output()

    # Exit without doing anything
    with mock.patch.object(jpod.misc, 'check_yes_no', lambda prompt, default: 'n'):
        jpod.ui.main()

    check_output()


def test_restart_pod(case='/Michalewicz'):
    """Test all restart options."""
    # Restart POD from existing one and continue with resample
    init_case(case)
    sys.argv = ['jpod', 'settings.json', '-r']
    options = jpod.ui.parse_options()
    settings = jpod.misc.import_config(options.settings, schema)
    settings["space"]["resampling"]["resamp_size"] = 1

    jpod.ui.run(settings, options)
    check_output()

    if not os.path.isdir('output/snapshots/4'):
        assert False

    init_case(case, force=True)
    # Restart from 4 and add 2 points continuing the DOE sequence
    settings["space"]["resampling"]["resamp_size"] = 0
    settings["space"]["sampling"]["init_size"] = 6

    jpod.ui.run(settings, options)
    check_output()

    if not os.path.isdir('output/snapshots/5'):
        assert False


def test_resampling(case='/Michalewicz'):
    """Assess all resampling methods."""
    sys.argv = ['jpod', 'settings.json']
    options = jpod.ui.parse_options()
    settings = jpod.misc.import_config(options.settings, schema)
    settings["space"]["sampling"]["init_size"] = 10
    settings["space"]["resampling"]["resamp_size"] = 2

    for method in ["loo_sigma", "loo_sobol", "extrema"]:
        print("Method: ", method)
        os.system('rm -rf output')
        settings["space"]["resampling"]["method"] = method
        if method == "extrema":
            settings["space"]["resampling"]["resamp_size"] = 4
        jpod.ui.run(settings, options)
        check_output()
        if not os.path.isdir('output/snapshots/11'):
            assert False

# Ishigami: 3D -> 1D
# Oakley & O'Hagan: 1D -> 1D
# Channel_Flow: 2D -> 400D
@pytest.mark.parametrize("name", [
    ('/Ishigami'),
    ('/Basic_function'),
    ('/Channel_Flow'),
])
def test_cases(name):
    test_init(case=name)
    test_quality(case=name)
    test_uq(case=name)
    test_restart_pod(case=name)
