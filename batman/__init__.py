"""
BATMAN package
**************
"""

from .driver import Driver
from . import mpi
from . import misc

__all__ = ['Driver', 'mpi', 'misc', 'pod']

__version__ = '1.5'
__branch__ = 'heads/run_mascaret'
__commit__ = '1.4-204-g7a3982b'
