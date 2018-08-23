"""
Functions module
================

Defines some analytical functions, utilities as well as the Mascaret API.
"""
from .data import (el_nino, tahiti, mascaret, marthe)
from .analytical import (SixHumpCamel, Branin, Michalewicz, Rosenbrock,
                         Rastrigin, Ishigami, G_Function,
                         Forrester, Channel_Flow, Manning, ChemicalSpill)
from .db_mascaret import db_Mascaret
from .utils import (output_to_sequence)

__all__ = ['SixHumpCamel', 'Branin', 'Michalewicz', 'Rosenbrock', 'Ishigami',
           'Rastrigin', 'G_Function', 'Forrester', 'Channel_Flow', 'Manning',
           'db_Mascaret', 'ChemicalSpill',
           'output_to_sequence',
           'el_nino', 'tahiti', 'mascaret', 'marthe']
