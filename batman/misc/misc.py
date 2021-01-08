# coding: utf8
"""Misc functions.

Implements functions:

- :func:`misc.clean_path`,
- :func:`misc.check_yes_no`,
- :func:`misc.ask_path`,
- :func:`misc.abs_path`,
- :func:`misc.import_config`,
- :class:`misc.ProgressBar`

"""
import os
import sys
import logging
import json
import time

import numpy as np
from pathos.multiprocessing import cpu_count
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed

from .schema import Settings


def clean_path(path):
    """Return an absolute and normalized path."""
    return os.path.abspath(os.path.normpath(path))


def check_yes_no(prompt, default):
    """Ask user for delete confirmation.

    :param str prompt: yes-no question
    :param str default: default value
    :returns: true if yes
    :rtype: boolean
    """
    logger = logging.getLogger('User checking')
    while True:
        try:
            value = input(prompt)
        except ValueError:
            logger.error("Sorry, I didn't understand that.")
            continue

        value = value.lower()
        if not all(x in 'yesno ' for x in value.lower()):
            logger.error('Sorry, your response must be yes, or no.')
            continue
        elif value == '':
            value = default
            break
        else:
            break

    answer = True if value.strip()[0] == 'y' else False

    return answer


def ask_path(prompt, default, root):
    """Ask user for a folder path.

    :param str prompt: Ask.
    :param str default: default value.
    :param str root: root path.
    :returns: path if folder exists.
    :rtype: str.
    """
    logger = logging.getLogger('User checking')
    while True:
        try:
            path = input(prompt)
        except ValueError:
            logger.error("Sorry, I didn't understand that.")
            continue

        if path == '':
            path = default

        if not os.path.isdir(os.path.join(root, path)):
            logger.error("Output folder not found: {}".format(path))
            continue
        else:
            break

    return path


def abs_path(value):
    """Get absolute path."""
    return os.path.abspath(value)


def import_config(fname):
    """Import a configuration file."""
    with open(fname) as json_file:
        settings = json.load(json_file)

    return Settings(**settings)


class ProgressBar:
    """Print progress bar in console."""

    def __init__(self, total):
        """Create a bar.

        :param int total: number of iterations
        """
        self.total = total
        self.calls = 1
        self.progress = 0.

        sys.stdout.write("Progress | " +
                         " " * 50 +
                         " |" + "0.0% ")

        self.init_time = time.time()

    def __call__(self):
        """Update bar."""
        self.progress = (self.calls) / float(self.total) * 100

        eta, vel = self.compute_eta()
        self.show_progress(eta, vel)

        self.calls += 1

    def compute_eta(self):
        """Compute ETA.

        Compare current time with init_time.

        :return: eta, vel
        :rtype: str
        """
        end_time = time.time()
        iter_time = (end_time - self.init_time) / self.calls

        eta = (self.total - self.calls) * iter_time
        eta = time.strftime("%H:%M:%S", time.gmtime(eta))

        vel = str(1. / iter_time)

        return eta, vel

    def show_progress(self, eta=None, vel=None):
        """Print bar and ETA if relevant.

        :param str eta: ETA in H:M:S
        :param str vel: iteration/second
        """
        p_bar = int(np.floor(self.progress / 2))
        sys.stdout.write("\rProgress | " +
                         "-" * (p_bar - 1) + "~" +
                         " " * (50 - p_bar - 1) +
                         " |" + str(self.progress) + "% ")

        if self.progress == 100:
            sys.stdout.write('\n')
            del self
        elif eta and vel:
            sys.stdout.write("| ETA: " + eta + " (at " + vel + " it/s) ")

        sys.stdout.flush()


def cpu_system():
    """Number of physical CPU of system."""
    try:
        try:
            import psutil
            n_cpu_system = psutil.cpu_count(logical=False)
        except ImportError:
            n_cpu_system = cpu_count()
    except NotImplementedError:
        n_cpu_system = os.sysconf('SC_NPROCESSORS_ONLN')

    return 1 if (n_cpu_system == 0) or (n_cpu_system is None) else n_cpu_system


def optimization(bounds, discrete=None):
    """Perform a discret or a continuous/discrete optimization.

    If a variable is discrete, the decorator allows to find the optimum by
    doing an optimization per discrete value and then returns the optimum.

    :param array_like bounds: bounds for optimization ([min, max], n_features).
    :param int discrete: index of the discrete variable.
    """
    def optimize(fun):
        """Compute several optimizations."""
        def combinatory_optimization(i, bounds=bounds, discrete=discrete):
            """One optimization.

            :param int i: index of the optimization plan.
            :param array_like bounds: bounds for optimization ([min, max], n_features).
            :param int discrete: index of the discrete variable.
            :returns: min_x, min_fun.
            :rtype: floats.
            """
            bounds[discrete] = [i, i]
            with np.errstate(divide='ignore', invalid='ignore'):
                results = differential_evolution(fun, bounds)
            min_x = results.x
            min_fun = results.fun
            return min_x, min_fun

        def wrapper_fun_obj():
            """Two behaviour depending on discrete."""
            if discrete is not None:
                start = int(np.ceil(bounds[discrete, 0]))
                end = int(np.ceil(bounds[discrete, 1]))
                discrete_range = range(start, end) if end > start else [start]

                results = Parallel(n_jobs=cpu_system())(delayed(combinatory_optimization)(i)
                                                        for i in discrete_range)
                results = list(results)

                min_x, min_fun = zip(*results)

                # Find best results
                min_idx = np.argmin(min_fun)
                min_fun = min_fun[min_idx]
                min_x = min_x[min_idx]
            else:
                results = differential_evolution(fun, bounds)
                min_x = results.x
                min_fun = results.fun
            return min_x, min_fun
        return wrapper_fun_obj
    return optimize
