import os
from auto_server import *
from pod_server_settings import pod_server
import snapshot_io

pod['server'] = pod_server
pod['server']['python'] = 'openmpirun -n 2 python'

root = '/Users/dechaume/Codes/pod/jpod/jpod2/test/output/auto_mpi/snapshots'
def key(arg):
    return int(os.path.basename(os.path.normpath(arg)))
snapshot['provider'] = sorted([os.path.join(root, d) for d in os.listdir(root)], key=key)

snapshot['io']['template_directory'] = os.path.join(root, '0')
snapshot['io']['shapes'] = None
snapshot['io'] = snapshot_io.io_mpi
