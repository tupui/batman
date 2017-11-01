"""
Functions module
================

Defines some analytical functions, utilities as well as the Mascaret API.
"""
from .data import (Data, el_nino, tahiti)
from .analytical import (SixHumpCamel, Branin, Michalewicz, Rosenbrock,
                         Rastrigin, Ishigami, G_Function,
                         Forrester, Channel_Flow, Manning, ChemicalSpill)
from .mascaret import (Mascaret, MascaretApi)
from .utils import (multi_eval, output_to_sequence)

__all__ = ['SixHumpCamel', 'Branin', 'Michalewicz', 'Rosenbrock', 'Ishigami',
           'Rastrigin', 'G_Function', 'Forrester', 'Channel_Flow', 'Manning',
           'Mascaret', 'MascaretApi', 'ChemicalSpill',
           'multi_eval', 'output_to_sequence', Data, el_nino, tahiti]
