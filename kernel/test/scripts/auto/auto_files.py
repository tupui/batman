from auto import *
import os

root = 'output/auto_server/snapshots'
def key(arg):
    return int(os.path.basename(os.path.normpath(arg)))
snapshot['provider'] = sorted([os.path.join(root, d) for d in os.listdir(root)], key=key)
