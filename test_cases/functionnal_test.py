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

    if os.path.isdir('output'):
        run = False
    if force:
        os.system('rm -rf output')
        run = True
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
    settings["space"]["size_max"] = 5

    jpod.ui.run(settings, options)
    check_output()

    if not os.path.isdir('output/snapshots/4'):
        assert False

    init_case(case, force=True)
    # Restart from 4 and add 2 points continuing the DOE sequence
    settings["space"]["size_max"] = 6
    settings["space"]["provider"]["size"] = 6

    jpod.ui.run(settings, options)
    check_output()

    if not os.path.isdir('output/snapshots/5'):
        assert False


def test_resampling(case='/Michalewicz'):
    """Assess all resampling methods."""
    init_case(case)
    sys.argv = ['jpod', 'settings.json']
    options = jpod.ui.parse_options()
    settings = jpod.misc.import_config(options.settings, schema)
    settings["space"]["size_max"] = 6

    for method in ["loo_mse", "loo_sobol", "extrema"]:
        os.system('rm -rf output')
        settings["pod"]["resample"] = method
        if method == "extrema":
            settings["space"]["size_max"] = 8
        jpod.ui.run(settings, options)
        check_output()
        if not os.path.isdir('output/snapshots/5'):
            assert False


# Ishigami: 3D -> 1D
def test_ishigami():
    test_init(case='/Ishigami')
    test_quality(case='/Ishigami')
    test_uq(case='/Ishigami')
    test_restart_pod(case='/Ishigami')


# Oakley & O'Hagan: 1D -> 1D
def test_basic():
    test_init(case='/Basic_function')
    test_quality(case='/Basic_function')
    test_uq(case='/Basic_function')
    test_restart_pod(case='/Basic_function')


# Channel_Flow: 2D -> 400D
def test_channel_flow():
    test_init(case='/Channel_Flow')
    test_quality(case='/Channel_Flow')
    test_uq(case='/Channel_Flow')
    test_restart_pod(case='/Channel_Flow')
