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
    a dataset to a point.
    """

    def __init__(self, point, future_data):
        self.point = Point(point)
        self._future = future_data

    @property
    def data(self):
        return self._future.result()
