"""
Surrogate model module
**********************
"""

from .RBFnet import RBFnet
from .kriging import Kriging
from .polynomial_chaos import PC
from .surrogate_model import SurrogateModel
from .multifidelity import Evofusion

__all__ = ['RBFnet', 'Kriging', 'PC', 'SurrogateModel', 'Evofusion']
