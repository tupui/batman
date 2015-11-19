from dynamic_sobol import *
import os

root = 'output/dynamic_sobol_server/snapshots'
def key(arg):
    return int(os.path.basename(os.path.normpath(arg)))
snapshot['provider'] = sorted([os.path.join(root, d) for d in os.listdir(root)], key=key)

snapshot['io']['template_directory'] = os.path.join(root, '0')
snapshot['io']['shapes'] = None
