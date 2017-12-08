"""
Functions module
================

Defines some analytical functions, utilities as well as the Mascaret API.
"""
from .data import (Data, el_nino, tahiti, mascaret, marthe)
from .analytical import (SixHumpCamel, Branin, Michalewicz, Rosenbrock,
                         Rastrigin, Ishigami, G_Function,
                         Forrester, Channel_Flow, Manning, ChemicalSpill)
from .telemac_mascaret import (Mascaret, MascaretApi)
from .utils import (output_to_sequence)

__all__ = ['SixHumpCamel', 'Branin', 'Michalewicz', 'Rosenbrock', 'Ishigami',
           'Rastrigin', 'G_Function', 'Forrester', 'Channel_Flow', 'Manning',
           'Mascaret', 'MascaretApi', 'ChemicalSpill',
           'output_to_sequence',
           'Data', 'el_nino', 'tahiti', 'mascaret', 'marthe']
