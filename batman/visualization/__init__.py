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
from .uncertainty import (kernel_smoothing, pdf, sensitivity_indices, corr_cov)
from .density import (cusunoro, moment_independent)
from .doe import (doe, doe_ascii, pairplot)
from .response_surface import response_surface
from .mesh_2D import mesh_2D

__all__ = ['Kiviat3D', 'Tree', 'HdrBoxplot', 'kernel_smoothing', 'pdf',
           'sensitivity_indices', 'corr_cov', 'cusunoro', 'moment_independent',
           'reshow', 'save_show', 'response_surface', 'doe', 'doe_ascii',
           'pairplot', 'mesh_2D']


def reshow(fig):
    """Create a dummy figure and use its manager to display :attr:`fig`.

    :param fig: Matplotlib figure instance.
    """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    return dummy


def save_show(fname, figures, **kwargs):
    """Either show or save the figure[s].

    If :attr:`fname` is `None` the figure will show.

    :param str fname: whether to export to filename or display the figures.
    :param list(Matplotlib figure instance) figures: Figures to handle.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fig in figures:
            try:
                fig.tight_layout()
            except ValueError:
                pass

    if fname is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(fname)
        for fig in figures:
            pdf.savefig(fig, transparent=True, bbox_inches='tight', **kwargs)
        pdf.close()
    else:
        plt.show()
    plt.close('all')
