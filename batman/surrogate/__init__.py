"""
Surrogate model module
**********************
"""

from .RBFnet import RBFnet
from .kriging import Kriging
from .sk_interface import SklearnRegressor
from .polynomial_chaos import PC
from .surrogate_model import SurrogateModel
from .multifidelity import Evofusion
from .mixture import Mixture

__all__ = ['RBFnet', 'Kriging', 'SklearnRegressor', 'PC', 'SurrogateModel',
           'Evofusion', 'Mixture']
