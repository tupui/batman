"""
Surrogate model module
**********************
"""

from .RBFnet import RBFnet
from .kriging import Kriging

__all__ = ['RBFnet', 'Kriging']
