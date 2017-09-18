"""
Visualization module
********************
"""

from .kiviat import Kiviat3D
from .hdr import HdrBoxplot
from .uncertainty import pdf
from matplotlib import pyplot as plt

__all__ = ['Kiviat3D', 'HdrBoxplot', 'pdf', 'reshow']


def reshow(fig):
    """Create a dummy figure and use its manager to display :attr:`fig`.

    :param fig: Matplotlib figure instance
    """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    return dummy
