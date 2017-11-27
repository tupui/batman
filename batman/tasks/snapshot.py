# coding: utf-8
"""
This module defines the Snapshot Class.

author: Cyril Fournier
"""
from ..space import Point


class Snapshot(object):
    """
    A snapshot container.

    Its very basic interface is just used for binding
    a dataset to a sample point.
    """

    def __init__(self, point, data):
        self._point = Point(point)
        self._data = data

    @property
    def point(self):
        return self._point

    @property
    def data(self):
        """
        Returns snapshot data.
        Waits for completion if data is produced by a job.
        """
        try:
            self._data = self._data.result()
        except AttributeError:
            pass
        return self._data
