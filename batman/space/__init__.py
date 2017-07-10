"""
Space module
************
"""

from .space import Space, FullSpaceError, AlienPointError, UnicityError
from .sampling import Doe
from .gp_1d_sampler import Gp1dSampler
from .point import Point
from .refiner import Refiner

__all__ = ["Space", "Doe", "Gp1dSampler", "Point", "Refiner",
           "FullSpaceError", "AlienPointError", "UnicityError"]
