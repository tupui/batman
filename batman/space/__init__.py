"""
Space module
************
"""

from .space import Space
from .sampling import Doe
from .sample import Sample
from .refiner import Refiner
from .point import Point

__all__ = ["Space", "Doe", "Sample", "Refiner", "dists_to_ot", "Point"]


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
