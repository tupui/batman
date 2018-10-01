"""
Functions module
================

Defines some analytical functions, utilities as well as the Mascaret API.
"""
from .data import (el_nino, tahiti, mascaret, marthe)
from .analytical import (SixHumpCamel, Branin, Michalewicz, Rosenbrock,
                         Rastrigin, Ishigami, G_Function, Forrester,
                         ChemicalSpill, Channel_Flow, Manning)
from .db_mascaret import db_Mascaret
from .db_generic import DbGeneric
from .utils import (output_to_sequence)

__all__ = ['SixHumpCamel', 'Branin', 'Michalewicz', 'Rosenbrock', 'Ishigami',
           'Rastrigin', 'G_Function', 'Forrester', 'ChemicalSpill',
           'Channel_Flow', 'Manning', 'db_Mascaret', 'DbGeneric',
           'output_to_sequence',
           'el_nino', 'tahiti', 'mascaret', 'marthe']
