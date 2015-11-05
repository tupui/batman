from concurrent import futures as _futures
import mpi

__dummy = None

class do_nothing_ThreadPoolExecutor(object):
    def __init__(self, *args, **kwargs):
        pass
    def submit(self, *args, **kwargs):
        pass


# dummy futures for debugging purposes
class dummy_ThreadPoolExecutor(object):
    def __init__(self, **kwargs):
        pass
    def submit(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class dummy_Futures(object):
    def __init__(self, arg):
        self.arg = arg
    def result(self):
        return self.arg


def dummy_as_completed(fs, timeout=None):
    return [dummy_Futures(f) for f in fs]


def SnapshotProviderManager(dummy=False, *args, **kwargs):
    """Dispatch SnapshotProviderManager instance according to kind."""
    global __dummy
    __dummy = dummy
    if dummy:
        return dummy_ThreadPoolExecutor(*args, **kwargs)
    elif mpi.myid == 0:
        return _futures.ThreadPoolExecutor(*args, **kwargs)
    else:
        return do_nothing_ThreadPoolExecutor(*args, **kwargs)


def mpi_as_completed(*arg, **kwargs):
    """MPI safe as_completed, only master drives things up."""
    if mpi.myid == 0:
        for f in _futures.as_completed(*arg, **kwargs):
            result = f.result()
            mpi.bcast(result)
            yield result
    else:
        result = mpi.bcast()
        yield result


def available_snapshots(*arg, **kwargs):
    """Dispatch SnapshotProviderManager instance according to the context."""
    if __dummy:
        return dummy_as_completed(*arg, **kwargs)
    elif mpi.size == 1:
        return _futures.as_completed(*arg, **kwargs)
    else:
        return mpi_as_completed(*arg, **kwargs)

