# coding: utf8
"""NestedPool class.

This class is used when nested process pool are needed.
It modify the ``daemon`` attribute to allow this subprocessing.

"""
import pathos
import multiprocess


class NoDaemonProcess(multiprocess.Process):
    """NoDaemonProcess class.

    Inherit from :class:`multiprocessing.Process`.
    The ``daemon`` attribute always returns False.
    """

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NestedPool(pathos.multiprocessing.Pool):
    """NestedPool class.

    Inherit from :class:`pathos.multiprocessing.Pool`.
    Enable nested process pool.
    """

    Process = NoDaemonProcess
