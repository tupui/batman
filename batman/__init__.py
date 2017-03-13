"""
BATMAN package
**************
"""

from .driver import Driver
from . import mpi
from . import misc

__all__ = ['Driver', 'mpi', 'misc', 'pod']

__version__ = '1.5'
__branch__ = 'heads/release-Oswald'
__commit__ = '1.4-203-gd8427ed'
