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


def init_case(case, output=True):
    os.chdir(path + case)
    os.system('rm -rf output')
    sys.argv = ['jpod', 'settings.json']
    jpod.ui.main()
    check_output()
    if not output:
        os.system('rm -rf output/pod')


# Use Ishigami
def test_init():
    init_case('/Ishigami')
    check_output()


def test_no_pod():
    init_case('/Ishigami')
    sys.argv = ['jpod', 'settings.json', '-n']
    jpod.ui.main()
    check_output()


def test_checks():
    """Check answers to questions if there is an output folder."""
    init_case('/Ishigami', output=False)

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


def test_restart_pod():
    """Test all restart options."""
    # Restart POD from existing one and continue with resample
    init_case('/Ishigami')
    sys.argv = ['jpod', 'settings.json', '-r']
    options = jpod.ui.parse_options()
    settings = jpod.misc.import_config(options.settings, schema)
    settings["space"]["size_max"] = 5

    jpod.ui.run(settings, options)
    check_output()

    if not os.path.isdir('output/snapshots/4'):
        assert False

    init_case('/Ishigami')
    # Restart from 4 and add 2 points continuing the DOE sequence
    settings["space"]["size_max"] = 6
    settings["space"]["provider"]["size"] = 6

    jpod.ui.run(settings, options)
    check_output()

    if not os.path.isdir('output/snapshots/5'):
        assert False


def test_resampling():
    """Assess all resampling methods."""
    sys.argv = ['jpod', 'settings.json']
    options = jpod.ui.parse_options()
    settings = jpod.misc.import_config(options.settings, schema)
    settings["space"]["size_max"] = 6

    for method in ["MSE", "loo_mse", "loo_sobol", "extrema"]:
        os.system('rm -rf output')
        settings["pod"]["resample"] = method
        if method == ["extrema"]:
            settings["space"]["size_max"] = 8
        jpod.ui.run(settings, options)
        check_output()
        if not os.path.isdir('output/snapshots/5'):
            assert False
