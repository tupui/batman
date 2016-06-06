"""A command line interface to jpod."""

import logging
from optparse import OptionParser
import os
import sys

from driver import Driver
from misc import import_file
from misc import logging_conf
import mpi
import numpy as N
from pod import Snapshot


help_message = '''
jpod settings_file.py
'''
__version__ = 2

logger = logging.getLogger('ui')


def run(settings, options):
    """Run the driver along."""
    # clean up output directory
    if not options.restart \
       and not options.no_pod and not options.pred:
        mpi.clean_makedirs(options.output)
    	# tell that the output directory has previously been clean
    	logger.info('cleaning : %s', options.output)

    # setup logging, after directory creation
    logging_conf.setup(options.output, 'driver')

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
        snapshots = driver.prediction(settings.prediction,
                                  write=options.save_snapshots)
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


def output_option(option, opt, value, parser):
    """Get absolute path."""
    parser.values.output = os.path.abspath(value)


def parse_command_line_and_run(argv=None):
    """Parse and check options, and then call XXX()."""

    if argv is None:
        argv = sys.argv  # [1:]

    parser = OptionParser(usage=help_message, version=__version__)

    # command line options
    parser.add_option(
        '-s', '--save-snapshots',
        action='store_true',
        default=False,
        help='save the snapshots to disk when using a function, [default: %default].')

    parser.add_option(
        '-o', '--output',
        type='string',
        # action='store',
        action='callback',
        callback=output_option,
        default='.',
        help='path to output directory, [default: %default].')

    parser.add_option(
        '--set',
        action='append',
        default=[],
        help='jpod settings to override the file ones, pass "setting_name[key1]...[keyN]=value", [default: none]')

    parser.add_option(
        '-r', '--restart',
        action='store_true',
        default=False,
        help='restart pod, [default: %default].')

    parser.add_option(
        '-n', '--no-pod',
        action='store_true',
        default=False,
        help='do not compute pod but read it from disk, [default: %default].')

    parser.add_option(
        '-u', '--uq',
        action='store_true',
        default=False,
        help='Uncertainty Quantification study')

    parser.add_option(
        '-p',
        action='store_true',
        default=False,
        dest='pred',
        help='compute prediction and write it from disk, [default: %default].')

    # parse command line
    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error("incorrect number of arguments")
    settings = import_file(args[0])

    # store settings absolute file path
    options.script = os.path.abspath(args[0])

    # override input script settings from command line
    for s in options.set:
        logger.info('overriding setting : %s', s)
        exec s in settings.__dict__

    try:
        run(settings, options)
        return 0
    except:
        logger.exception('Exception caught on cpu %i' % mpi.myid)
        return 1


if __name__ == "__main__":
    sys.exit(parse_command_line_and_run())
