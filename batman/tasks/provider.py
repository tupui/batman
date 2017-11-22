# coding: utf-8
"""
This module defines an abstract Provider class.

author: Cyril Fournier
"""
from abc import ABCMeta, abstractmethod, abstractproperty


class AbstractProvider(object):
    """
    The Provider abstract interface.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, executor, settings):
        """
        :param executor: an executor pool with submit methode that returns a future.
        """
        pass

    @abstractproperty
    def known_points(self):
        """
        Returns the list of known point as a dict {point: 'path'}
        """
        pass

    @abstractmethod
    def snapshot(self, point, *kwargs):
        """
        Submit an asynchronous task that produce data at the given point.
        :returns: a snapshot bound to the result of a task.
        """
        pass
