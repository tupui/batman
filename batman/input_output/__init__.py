"""
IO module
*********

Provides Formater objects to deal with I/Os.

Every formaters have the same interface, exposing the two methods **read** and **write**.

:Example: Using **json** formater

::

    >> from input_output import formater
    >> varnames = ['x1', 'x2', 'x3']
    >> data = [[1, 2, 3], [87, 74, 42]]
    >> fmt = formater('json')
    >> fmt.write('file.json', data, varnames)
    {'x1': [1, 87], 'x2': [2, 74], 'x3': [3, 42]}
    >> # can load a subset of variables, in a different order (unavailable for format 'npy')
    >> fmt.read('file.json', ['x2', 'x1'])
    array([[2, 1], [74, 87]])

"""
from copy import copy
from .formater import FORMATER as BUILTIN_FORMATER
from .antares_wrappers import ANTARES_FORMATER


__all__ = ['formater', 'available_formats']


FORMATER = copy(ANTARES_FORMATER)
FORMATER.update(BUILTIN_FORMATER)  # highest priority to built-in formaters


def available_formats():
    """Return the list of available format names."""
    return copy(FORMATER.keys())


def formater(format_name):
    """Return a Formater."""
    return FORMATER[format_name]
