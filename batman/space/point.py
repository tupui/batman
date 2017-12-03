# coding: utf8
"""
A simple class of points in space.

A point is a tuple of coordinates, thus this class derives from the tuple class.
A point can be created either from integer or float coordinates, they are
stored internally as float values.
Two points can be compared with the usual `==` or `!=`, it can be approximative
in the sense that two close points can be equals. The comparison is done by
computing the distance of the points and checking it against a predefined threshold.
"""
import logging
import numpy as np


class Point(tuple):
    """Point class."""

    logger = logging.getLogger(__name__)

    threshold = 0.
    '''Maximum distance error when comparing 2 points.'''

    @classmethod
    def set_threshold(cls, threshold):
        """Set the threshold for comparing points."""
        try:
            threshold = float(threshold)
            if threshold < 0:
                raise ValueError
        except ValueError:
            cls.logger.exception("Threshold must be a positive real number.")
            raise ValueError
        else:
            cls.threshold = threshold

    def __new__(cls, coordinates):
        """Create a new object from `coordinates`.

        The argument `coordinates` must be a collection that can be converted
        to a tuple.
        """
        # check and eventually convert int to float
        coords = []
        for c in coordinates:
            try:
                coords += [float(c)]
            except ValueError:
                cls.logger.exception("Coordinate values must be real numbers: {}"
                                     .format(c))
                raise ValueError
        return super(Point, cls).__new__(cls, coords)

    def __eq__(self, other):
        """Compare using the euclidian distance"""
        return np.linalg.norm(np.array(self) - np.array(other)) \
            <= self.__class__.threshold

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(self))
