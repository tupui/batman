"""
IO module
*********

Provides Formater objects to deal with I/Os.

Each formater exposes a reader and a writer method.
Datasets must be numpy structured arrays with named fields.
"""
from copy import copy
from .formater import FORMATER as DEFAULT_FORMATER
from .antares_wrappers import ANTARES_FORMATER


__all__ = ['FORMATER']

FORMATER = copy(ANTARES_FORMATER)
FORMATER.update(DEFAULT_FORMATER)  # highest priority to built-in formaters

