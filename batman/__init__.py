"""
BATMAN package
**************
"""

from .driver import Driver
from . import mpi
from . import misc

__all__ = ['Driver', 'mpi', 'misc', 'pod']

__version__ = '1.5'
__branch__ = 'heads/feature-evofusion'
__commit__ = 'Oswald-47-ge4b4d6a'
