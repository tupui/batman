from .analytical import (Michalewicz, Rosenbrock,
                   Ishigami, G_Function, Channel_Flow)
from .mascaret import (Mascaret)

__all__ = ['Michalewicz', 'Rosenbrock', 'Ishigami',
           'G_Function', 'Channel_Flow', 'Mascaret']

dispatcher = {
    "Michalewicz": Michalewicz,
    "Ishigami": Ishigami,
    "Rosenbrock": Rosenbrock,
    "G_Function": G_Function,
    "Channel_Flow": Channel_Flow,
    "Mascaret": Mascaret
}
