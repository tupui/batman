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

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NestedPool(pathos.multiprocessing.Pool):
    """NestedPool class.

    Inherit from :class:`pathos.multiprocessing.Pool`.
    Enable nested process pool.
    """

    def Process(self, *args, **kwds):
        proc = pathos.multiprocessing.Pool.Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess

        return proc
