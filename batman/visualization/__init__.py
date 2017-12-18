"""
Visualization module
********************
"""
import warnings
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
from .kiviat import Kiviat3D
from .tree import Tree
from .hdr import HdrBoxplot
from .uncertainty import (kernel_smoothing, pdf, sobol, corr_cov)
from .doe import doe
from .response_surface import response_surface

__all__ = ['Kiviat3D', 'Tree', 'HdrBoxplot', 'kernel_smoothing', 'pdf',
           'sobol', 'corr_cov', 'reshow', 'save_show', 'response_surface',
           'doe']


def reshow(fig):
    """Create a dummy figure and use its manager to display :attr:`fig`.

    :param fig: Matplotlib figure instance
    """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    return dummy


def save_show(fname, figures):
    """Either show or save the figure[s].

    If :attr:`fname` is `None` the figure will show.

    :param str fname: wether to export to filename or display the figures.
    :param list(Matplotlib figure instance) figures: Figures to handle.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

    if fname is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(fname)
        for fig in figures:
            pdf.savefig(fig, transparent=True, bbox_inches='tight')
        pdf.close()
    else:
        plt.show()
    plt.close('all')
