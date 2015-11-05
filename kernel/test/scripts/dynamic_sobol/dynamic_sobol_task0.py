from dynamic_sobol import *
import snapshot_provider
import snapshot_io
snapshot['provider'] = snapshot_provider.job_mpi
snapshot['io'] = snapshot_io.io_mpi
