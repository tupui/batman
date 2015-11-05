'''Same as static_sobol_server but with snapshots computed as task run from a bash script, 1 task at a time.
'''
from static_sobol_server import *
import snapshot_provider
snapshot['provider'] = snapshot_provider.job
