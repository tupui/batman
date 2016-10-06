"""A command line interface to jpod."""

import logging
from logging.config import dictConfig
import argparse
import os
import json

from jpod import __version__, __branch__, __commit__
from jpod import import_file
from jpod import Driver
from jpod import mpi

description_message = '''
JPOD creates a surrogate model using POD+Kriging and perform UQ.
'''

jpod_banner = r"""
    _____  _______    ______   _______
   |     \|       \  /      \ |       \
    \$$$$$| $$$$$$$\|  $$$$$$\| $$$$$$$\
      | $$| $$__/ $$| $$  | $$| $$  | $$
 __   | $$| $$    $$| $$  | $$| $$  | $$
|  \  | $$| $$$$$$$ | $$  | $$| $$  | $$
| $$__| $$| $$      | $$__/ $$| $$__/ $$
 \$$    $$| $$       \$$    $$| $$    $$
  \$$$$$$  \$$        \$$$$$$  \$$$$$$$
"""

path = os.path.dirname(os.path.realpath(__file__)) + '/misc/logging.json'
with open(path, 'r') as file:
    logging_config = json.load(file)


def run(settings, options):
    """Run the driver along."""
    dictConfig(logging_config)
    if options.verbose:
        console = logging.getLogger().handlers[0]
        console.setLevel(logging.DEBUG)
        logging.getLogger().removeHandler('console')
        logging.getLogger().addHandler(console)

    logger = logging.getLogger('JPOD main')

    logger.info(jpod_banner)
    logger.info("Branch: {}\n\
        Last commit: {}".format(__branch__, __commit__))

    # clean up output directory
    if not options.restart and not options.no_pod and not options.pred:
        mpi.clean_makedirs(options.output)
        # tell that the output directory has previously been clean
        logger.debug('cleaning : {}'.format(options.output))

    driver = Driver(settings, options.script, options.output)

    update = settings.pod['type'] != 'static'

    if not options.no_pod and not options.pred:
        # the pod will be computed
        if options.restart:
            driver.restart()
            update = True

        driver.sampling_pod(update)

        if settings.pod['resample'] is not None:
            driver.resampling_pod()

        driver.write_pod()

    elif options.no_pod or options.pred:
        # just read the existing pod
        try:
            driver.read_pod()
        except IOError:
            logger.exception(
                "POD need to be computed: \
                check output folder or re-try without -n")
            raise SystemExit

    if not options.pred:
        driver.prediction(write=options.save_snapshots)
        driver.write_model()
    else:
        driver.prediction_without_computation(write=True)
        logger.info('Prediction without model building')

    logger.info(driver.pod)

    if options.q2:
        driver.pod.estimate_quality()

    if options.uq:
        driver.uq()


def abs_path(value):
    """Get absolute path."""
    return os.path.abspath(value)


def main():
    """Parse and check options, and then call run()."""
    # parser
    parser = argparse.ArgumentParser(prog="JPOD",
                                     description=description_message)
    parser.add_argument('--version',
                        action='version',
                        version="%(prog)s {}".format(__version__))

    # Positionnal arguments
    parser.add_argument(
        'settings',
        help='path to settings file')

    # Optionnal arguments
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='set verbosity from WARNING to DEBUG, [default: %(default)s]')

    parser.add_argument(
        '-s', '--save-snapshots',
        action='store_true',
        default=False,
        help='save the snapshots to disk when using a function,\
             [default: %(default)s]')

    parser.add_argument(
        '-o', '--output',
        type=abs_path,
        default='./output',
        help='path to output directory, [default: %(default)s]')

    parser.add_argument(
        '-r', '--restart',
        action='store_true',
        default=False,
        help='restart pod, [default: %(default)s]')

    parser.add_argument(
        '-n', '--no-pod',
        action='store_true',
        default=False,
        help='do not compute pod but read it from disk,\
             [default: %(default)s]')

    parser.add_argument(
        '-u', '--uq',
        action='store_true',
        default=False,
        help='Uncertainty Quantification study, [default: %(default)s].')

    parser.add_argument(
        '-p', '--pred',
        action='store_true',
        default=False,
        help='compute prediction and write it on disk, [default: %(default)s]')

    parser.add_argument(
        '-q', '--q2',
        action='store_true',
        default=False,
        help='estimate Q2 and find the point with max MSE,\
             [default: %(default)s]')

    # parse command line
    options = parser.parse_args()

    settings = import_file(options.settings)

    # store settings absolute file path
    options.script = os.path.abspath(options.settings)

    run(settings, options)


if __name__ == "__main__":
    main()
