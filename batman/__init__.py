"""
BATMAN package
**************
"""

from .driver import Driver
from . import mpi
from . import misc

__all__ = ['Driver', 'mpi', 'misc']

__version__ = '1.5'
__branch__ = 'heads/release-Oswald'
__commit__ = ''
