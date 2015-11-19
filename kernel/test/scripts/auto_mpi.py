from auto import *
import functions
import snapshot_io
from mpi4py import MPI

snapshot['io'] = snapshot_io.io_mpi
if MPI.COMM_WORLD.Get_rank() == 0:
    snapshot['provider'] = functions.partial(functions.f1, 3)
elif MPI.COMM_WORLD.Get_rank() == 1:
    def f(p):
        v = functions.f1(5,p)
        return v[3:5]
    snapshot['provider'] = f
