'''Same as static_sobol_server_task_0 but with several tasks at a time, snapshot order is not garanteed.
'''
from static_sobol_server_task_0 import *

snapshot['max_workers'] = 10
