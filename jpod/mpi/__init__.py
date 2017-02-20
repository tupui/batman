import os
import shutil

if os.getenv('MPI_STUB') is not None:
    sum = None
    myid = 0
    size = 1
    def Allreduce(*args, **kwargs):
        pass
    def allreduce(x, **kwargs):
        return x
    def barrier():
        pass
    def bcast(*args):
        return args
    def gather(*args):
        return args
else:
    from mpi4py import MPI
    myid      = MPI.COMM_WORLD.Get_rank()
    size      = MPI.COMM_WORLD.Get_size()
    Allreduce = MPI.COMM_WORLD.Allreduce
    allreduce = MPI.COMM_WORLD.allreduce
    barrier   = MPI.COMM_WORLD.barrier
    bcast     = MPI.COMM_WORLD.bcast
    gather    = MPI.COMM_WORLD.gather
    sum       = MPI.SUM


def makedirs(path):
    if myid == 0 and not os.path.isdir(path):
        os.makedirs(path)
    barrier()
