# coding: utf-8
"""
This module defines a specialized Provider class.

The ProviderPlugin class handles jobs that consist in
executing a python function from user-provided plugin.
"""
import logging
import importlib
import sys
import os
from .snapshot import Snapshot


class ProviderPlugin(object):
    """A Provider that build snapshost whose data come from a python function."""

    logger = logging.getLogger(__name__)

    def __init__(self, executor, io_manager, plug_settings):
        """Initialize the provider.

        :param executor: a task pool executor.
        :param io_manager: defines snapshots as files.
        :param plug_settings: specify how to load a plugin with the following:

            - **module** (str): python module to load.
            - **function** (str): function in `module` to execute when a
              snapshot is required.

        :type executor: :class:`concurrent.futures.Executor`
        :type io_manager: :class:`SnapshotIO`
        :type plug_settings: dict
        """
        sys.path.append(os.path.abspath('.'))
        plugin = importlib.import_module(plug_settings['module'])
        self._function = getattr(plugin, plug_settings['function'])
        self._executor = executor
        self._io = io_manager

    @property
    def known_points(self):
        """Dictionnary binding known snapshots with their location."""
        return {}  # never remember a snapshot

    def snapshot(self, point, *ignored):
        """Snapshot bound to an asynchronous job.

        It execute the provided plugin function.

        :param point: the point in parameter space at which to provide a snapshot.
        :type point: :class:`batman.space.Point`
        :return: A Snapshot.
        :rtype: :class:`Snapshot`
        """
        self.logger.debug('Request snapshot for point {}'.format(point))
        return Snapshot(point, self._executor.submit(self._function, point))
