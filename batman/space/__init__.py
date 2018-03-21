"""
Space module
************
"""

from .space import Space
from .sampling import Doe
from .sample import Sample
from .refiner import Refiner

__all__ = ["Space", "Doe", "Sample", "Refiner", "dists_to_ot", "kernel_to_ot"]


def dists_to_ot(dists):
    """Convert distributions to openTURNS.

    The list of distribution is converted to openTURNS objects.

    :Example:

    ::

        >> from batman.space import dists_to_ot
        >> dists = dists_to_ot(['Uniform(12, 15)', 'Normal(400, 10)'])

    :param list(str) dists: Distributions available in openTURNS.
    :return: List of openTURNS distributions.
    :rtype: list(:class:`openturns.Distribution`)
    """
    try:
        dists = [eval('ot.' + dist, {'__builtins__': None},
                      {'ot': __import__('openturns')})
                 for dist in dists]
    except (TypeError, AttributeError):
        raise AttributeError('OpenTURNS distribution unknown.')

    return dists


def kernel_to_ot(kernel):
    """Convert kernel to openTURNS.

    The kernel is converted to openTURNS objects.

    :Example:

    ::

        >> from batman.space import kernels_to_ot
        >> kernel = kernel_to_ot("AbsoluteExponential([0.5], 1.0)")

    :param str kernel: Kernel available in openTURNS.
    :return: openTURNS kernel.
    :rtype: list(:class:`openturns.Kernel`)
    """
    try:
        kernel = eval('ot.' + kernel, {'__builtins__': None},
                      {'ot': __import__('openturns')})
    except (TypeError, AttributeError):
        raise AttributeError('OpenTURNS kernel unknown.')

    return kernel
