# coding: utf-8
"""
This module defines the Snapshot Class.
"""
from ..space import Point


class Snapshot(object):
    """A snapshot container.

    Its very basic interface is just used for binding
    a dataset to a sample point.
    """

    def __init__(self, point, data):
        """ Initialize a snapshot.

        :param point: a point in parameter space.
        :param data: either a dataset or a :class:`concurrent.futures.Future` to a dataset.

        :type point: :class:`batman.space.Point`
        """

        self._point = Point(point)
        self._data = data

    @property
    def point(self):
        """Snapshot point coordinates."""
        return self._point

    @property
    def data(self):
        """Snapshot data."""
        try:
            self._data = self._data.result()
        except AttributeError:
            pass
        return self._data
