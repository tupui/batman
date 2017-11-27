# coding: utf-8
"""
This module defines a specialized Provider class.

The ProviderPlugin class handles jobs that consist in
executing a python function from user-provided plugin.

author: Cyril Fournier
"""
import logging
import importlib

from .provider import AbstractProvider
from .snapshot import Snapshot


class ProviderPlugin(AbstractProvider):
    """
    A Provider that build snapshost whose data come from a python function.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, executor, io_manager, settings):
        plugin = importlib.import_module(settings['module'])
        self._function = getattr(plugin, settings['function'])
        self._executor = executor
        self._io = io_manager

    @property
    def known_points(self):
        """
        Returns an empty mapping: never remember a snapshot.
        """
        return {}

    def snapshot(self, point, *ignored):
        """
        Returns a snapshot bound to an asynchronous job that execute the provided plugin function.
        """
        self.logger.debug('Request snapshot for point {}'.format(point))
        return Snapshot(point, self._executor.submit(self._function, point))
