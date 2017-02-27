# coding: utf8
"""A command line interface to jpod."""

import logging
from logging.config import dictConfig
import argparse
import os
import shutil
import json

from jpod import __version__, __branch__, __commit__
from jpod import Driver
from jpod import misc

description_message = '''
BATMAN creates a surrogate model using POD+Kriging and perform UQ.
'''

banner = r"""
 /$$$$$$$   /$$$$$$  /$$$$$$$$ /$$      /$$  /$$$$$$  /$$   /$$
| $$__  $$ /$$__  $$|__  $$__/| $$$    /$$$ /$$__  $$| $$$ | $$
| $$  \ $$| $$  \ $$   | $$   | $$$$  /$$$$| $$  \ $$| $$$$| $$
| $$$$$$$ | $$$$$$$$   | $$   | $$ $$/$$ $$| $$$$$$$$| $$ $$ $$
| $$__  $$| $$__  $$   | $$   | $$  $$$| $$| $$__  $$| $$  $$$$
| $$  \ $$| $$  | $$   | $$   | $$\  $ | $$| $$  | $$| $$\  $$$
| $$$$$$$/| $$  | $$   | $$   | $$ \/  | $$| $$  | $$| $$ \  $$
|_______/ |__/  |__/   |__/   |__/     |__/|__/  |__/|__/  \__/
BAyesian Tool for Modelling and uncertainty ANalysis
"""

path = os.path.dirname(os.path.realpath(__file__))
with open(path + '/misc/logging.json', 'r') as file:
    logging_config = json.load(file)

dictConfig(logging_config)


def run(settings, options):
    """Run the driver along."""
    if options.verbose:
        console = logging.getLogger().handlers[0]
        console.setLevel(logging.DEBUG)
        logging.getLogger().removeHandler('console')
        logging.getLogger().addHandler(console)

    logger = logging.getLogger('BATMAN main')

    logger.info(banner)
    logger.info("Branch: {}\n\
        Last commit: {}".format(__branch__, __commit__))

    # clean up output directory or re-use it
    if not options.restart and not options.no_surrogate and not options.pred:
        delete = True
        # check if output is empty and ask for confirmation
        if os.path.isdir(options.output):
            prompt = "Output folder exists, delete it? [y/N] > "
            delete = misc.check_yes_no(prompt, default='no')
            if not delete:
                prompt = "Re-use output results? [Y/n] > "
                use_output = misc.check_yes_no(prompt, default='yes')
                root = os.path.join(options.output, 'snapshots')

                if not os.path.isdir(root):
                    logger.warning("No folder snapshots in output folder")
                    raise SystemExit

                def key(arg):
                    return int(os.path.basename(
                        os.path.dirname(os.path.normpath(arg))))
                settings['snapshot']['provider'] = sorted([os.path.join(
                    root, d, 'jpod-data')
                    for d in os.listdir(root)],
                    key=key)
                settings['snapshot']['io']['template_directory'] = \
                    os.path.join(root, '0', 'jpod-data')
                settings['snapshot']['io']['shapes'] = None

                if not use_output:
                    logger.warning(
                        'Stopped to prevent deletion. Change options')
                    raise SystemExit
        if delete:
            try:
                shutil.rmtree(options.output)
            except OSError:
                pass
            os.makedirs(options.output)
            logger.debug('cleaning : {}'.format(options.output))

    driver = Driver(settings, options.output)

    if 'pod' in settings:
        update = True if settings['pod']['type'] != 'static' else False

    if not options.no_surrogate and not options.pred:
        # the surrogate [and POD] will be computed
        if options.restart:
            driver.restart()
            update = True

        driver.sampling(update=update)
        driver.write()

        try:
            if settings['space']['resampling']['resamp_size'] != 0:
                driver.resampling()
                driver.write()
        except KeyError:
            pass

    elif options.no_surrogate or options.pred:
        # just read the existing pod
        try:
            driver.read()
        except IOError:
            logger.exception(
                "Surrogate need to be computed: \
                check output folder or re-try without -n")
            raise SystemExit

    if 'predictions' in settings["surrogate"]:
        driver.prediction(write=options.save_snapshots)

    if 'pod' in settings:
        logger.info(driver.pod)

    if options.q2:
        driver.pod.estimate_quality()

    if options.uq:
        driver.uq()


def parse_options():
    """Parse options."""
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
        '-c', '--check',
        action='store_true',
        default=False,
        help='check settings, [default: %(default)s]')

    parser.add_argument(
        '-s', '--save-snapshots',
        action='store_true',
        default=False,
        help='save the snapshots to disk when using a function,\
             [default: %(default)s]')

    parser.add_argument(
        '-o', '--output',
        type=misc.abs_path,
        default='./output',
        help='path to output directory, [default: %(default)s]')

    parser.add_argument(
        '-r', '--restart',
        action='store_true',
        default=False,
        help='restart pod, [default: %(default)s]')

    parser.add_argument(
        '-n', '--no-surrogate',
        action='store_true',
        default=False,
        help='do not compute surrogate but read it from disk,\
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

    if options.check:
        raise SystemExit

    return options


def main():
    """Parse options, import and check settings then call run()"""
    options = parse_options()
    schema = path + "/misc/schema.json"
    settings = misc.import_config(options.settings, schema)

    run(settings, options)


if __name__ == "__main__":
    main()
