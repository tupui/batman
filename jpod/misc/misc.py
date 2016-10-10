"""Misc functions.

Implements functions:

- :func:`misc.clean_path`,
- :func:`misc.check_yes_no`,
- :func:`misc.abs_path`,
- :func:`misc.import_config`.

"""
import os
import logging
import json
import jsonschema


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


def abs_path(value):
    """Get absolute path."""
    return os.path.abspath(value)


def import_config(path_config, path_schema):
    """Import a configuration file."""
    logger = logging.getLogger('Settings Validation')

    with open(path_config, 'r') as file:
        settings = json.load(file)

    with open(path_schema, 'r') as file:
        schema = json.load(file)

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
