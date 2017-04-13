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
import re
import json
import jsonschema
import numpy as np
import time


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
            try:
                value = raw_input(prompt)
            except NameError:
                value = input(prompt)
        except ValueError:
            logger.error("Sorry, I didn't understand that.")
            continue

        value = value.lower()
        if not all(x in "yesno " for x in value.lower()):
            logger.error("Sorry, your response must be yes, or no.")
            continue
        elif value is '':
            value = default
            break
        else:
            break

    answer = True if value.strip()[0] is 'y' else False

    return answer


def ask_path(prompt, default, root):
    """Ask user for a folder path.

    :param str prompt: Ask 
    :param str default: default value
    :param str root: root path
    :returns: path if folder exists
    :rtype: str
    """
    logger = logging.getLogger('User checking')
    while True:
        try:
            try:
                path = raw_input(prompt)
            except NameError:
                path = input(prompt)
        except ValueError:
            logger.error("Sorry, I didn't understand that.")
            continue

        if path is '':
            path = default

        if not os.path.isdir(os.path.join(root, path)):
            logger.error("Output folder not found: ".format(path))
            continue
        else:
            break

    return path


def abs_path(value):
    """Get absolute path."""
    return os.path.abspath(value)


def import_config(path_config, path_schema):
    """Import a configuration file."""
    logger = logging.getLogger('Settings Validation')

    def minify_comments(file, **kwargs):
        """Minify comments in JSON file.

        Deserialize `file` to a Python object using `commentjson <https://pypi.python.org/pypi/commentjson>`_ package.

        :param file: serialized JSON string with or without comments.
        :param kwargs: all the arguments that `json.loads <http://docs.python.org/
                       2/library/json.html#json.loads>`_ accepts.
        :raises: Parsing Exception from ``json.loads``.
        :returns: dict or list.
        """
        file = file.read().decode('utf8')
        regex = r'\s*(#|\/{2}).*$'
        regex_inline = r'(:?(?:\s)*([A-Za-z\d\.{}]*)|((?<=\").*\"),?)(?:\s)*(((#|(\/{2})).*)|)$'
        lines = file.split('\n')

        for index, line in enumerate(lines):
            if re.search(regex, line):
                if re.search(r'^' + regex, line, re.IGNORECASE):
                    lines[index] = ""
                elif re.search(regex_inline, line):
                    lines[index] = re.sub(regex_inline, r'\1', line)

        try:
            return json.loads('\n'.join(lines), encoding="utf-8", **kwargs)
        except:
            logger.exception("Connot load configuration file: json error")
            raise SystemExit

    with open(path_config, 'rb') as file:
        settings = minify_comments(file)

    with open(path_schema, 'rb') as file:
        schema = minify_comments(file)

    error = False
    try:
        validator = jsonschema.Draft4Validator(schema)
        for error in sorted(validator.iter_errors(settings), key=str):
            logger.error("Error: {}\n\tOrigin: {}"
                         .format(error.message, error.path))
            error = True
    except jsonschema.ValidationError as e:
        logger.exception(e.message)

    if not error:
        logger.info("Settings successfully imported and checked")
    else:
        logger.error("Error were found in configuration file")
        raise SystemExit

    return settings


class ProgressBar(object):

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
        bar = int(np.floor(self.progress / 2))
        sys.stdout.write("\rProgress | " +
                         "-" * (bar - 1) + "~" +
                         " " * (50 - bar - 1) +
                         " |" + str(self.progress) + "% ")

        if self.progress == 100:
            sys.stdout.write('\n')
            del self
        elif (eta and vel):
            sys.stdout.write("| ETA: " + eta + " (at " + vel + " it/s) ")

        sys.stdout.flush()