"""A command line interface to jpod."""

import logging
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler
import argparse
import os
import json
import mpi
import numpy as N
from misc import import_file
from driver import Driver
from pod import Snapshot


description_message = '''
JPOD creates a surrogate model using POD+Kriging and perform UQ.
'''
__version__ = 1.2

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
    
    # clean up output directory
    if not options.restart \
       and not options.no_pod and not options.pred:
        mpi.clean_makedirs(options.output)
        # tell that the output directory has previously been clean
        logger.debug('cleaning : %s', options.output)

    driver = Driver(settings.snapshot, settings.space, options.output)

    driver.init_pod(settings, options.script)
    update = settings.pod['type'] != 'static'

    if not options.no_pod and not options.pred:
        # the pod will be computed
        if options.restart:
            driver.restart()
            update = True

        driver.fixed_sampling_pod(update)

        if settings.pod['type'] == 'auto':
            driver.automatic_resampling_pod()

        driver.write_pod()

    elif options.no_pod or options.pred:
        # just read the existing pod
        driver.read_pod()

    if not options.pred:
        snapshots = driver.prediction(settings.prediction, write=options.save_snapshots)
        driver.write_model()
    else:
        snapshots = driver.prediction_without_computation(
            settings.prediction,
            write=True)
        logger.info('Prediction without model building')

    if options.uq:
        driver.uq(settings)

    logger.info(driver.pod)

    if False and driver.provider.is_function:
        error = N.zeros(N.asarray(settings.prediction['points']).shape[0])

        for i, n in enumerate(snapshots):
            n = Snapshot.convert(n)
            p = settings.prediction['points'][i]
            error[i] = 100 * \
                ((n.data - driver.provider(p)) / driver.provider(p))[0]
        if mpi.myid == 0:
            logger.info('Relative error (inf norm) = %g',
                        N.linalg.norm(error, N.inf))


def abs_path(value):
    """Get absolute path."""
    return os.path.abspath(value)


def parse_command_line_and_run():
    """Parse and check options, and then call run()."""
    # parser
    parser = argparse.ArgumentParser(prog="JPOD", description=description_message)
    parser.add_argument('--version', action='version', version="%(prog)s {}".format(__version__))

    # Positionnal arguments
    parser.add_argument(
        'task',
        help='path to the task to run')

    # Optionnal arguments
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='Set verbosity from WARNING to DEBUG, [default: %(default)s].')

    parser.add_argument(
        '-s', '--save-snapshots',
        action='store_true',
        default=False,
        help='save the snapshots to disk when using a function, [default: %(default)s].')

    parser.add_argument(
        '-o', '--output',
        type=abs_path,
        default='./',
        help='path to output directory, [default: %(default)s].')

    parser.add_argument(
        '--set',
        action='append',
        default=[],
        help='jpod settings to override the file ones, pass "setting_name[key1]...[keyN]=value", [default: none]')

    parser.add_argument(
        '-r', '--restart',
        action='store_true',
        default=False,
        help='restart pod, [default: %(default)s].')

    parser.add_argument(
        '-n', '--no-pod',
        action='store_true',
        default=False,
        help='do not compute pod but read it from disk, [default: %(default)s].')

    parser.add_argument(
        '-u', '--uq',
        action='store_true',
        default=False,
        help='Uncertainty Quantification study')

    parser.add_argument(
        '-p',
        action='store_true',
        default=False,
        dest='pred',
        help='compute prediction and write it from disk, [default: %(default)s].')

    # parse command line
    options = parser.parse_args()

    settings = import_file(options.task)

    # store settings absolute file path
    options.script = os.path.abspath(options.task)

    # override input script settings from command line
    for s in options.set:
        logger.warn('overriding setting : %s', s)
        exec s in settings.__dict__

    #try:
    run(settings, options)
    #except:
    #    logger.exception('Exception caught on cpu %i' % mpi.myid)


if __name__ == "__main__":
    parse_command_line_and_run()
